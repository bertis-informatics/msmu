"""MuData access and bookkeeping helpers.

Contract: every msmu call receives a MuData whose MuData-level bookkeeping (global ``obs``/``var``,
``obsmap``/``varmap``, ``obsm``/``varm`` masks) is finished, meaning mudata's ``update()`` would leave
it unchanged, and returns one in the same state. The provenance hash covers that bookkeeping.

- ``copy_mudata`` copies the parts without rebuilding them (``MuData.copy()`` reruns ``update()``).
- ``add_modality`` / ``replace_modality`` insert or swap a modality and finish the bookkeeping.
- ``refresh_global_obs`` finishes the obs side for calls that only relabel observations.

These helpers read mudata's private attributes, so mudata is pinned to a minor in pyproject.
"""

import copy as copy_module
from collections.abc import MutableMapping
from hashlib import sha1
from typing import Any, TypeAlias, cast

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
from anndata._core.aligned_mapping import AxisArrays, PairwiseArrays
from mudata._core.file_backing import MuDataFileManager
from mudata._core.mudata import ModalityMapAxisArrays, ModDict

from .._core._provenance import log_provenance
from ..logging_utils import get_logger

MutableMuDataMod: TypeAlias = MutableMapping[str, ad.AnnData | md.MuData]

# dtype mudata gives obsmap/varmap: 0 marks "not in this modality", 1.. the position in the modality.
MODALITY_MAP_DTYPE = np.uint32

# Aligned-mapping attributes of a MuData, with the class and axis mudata builds them with. They hold a
# reference to their parent, so a copy is constructed for the new parent instead of deep-copied.
_ALIGNED_MAPPING_CLASSES: dict[str, tuple[type, int]] = {
    "_obsm": (AxisArrays, 0),
    "_varm": (AxisArrays, 1),
    "_obsp": (PairwiseArrays, 0),
    "_varp": (PairwiseArrays, 1),
    "_obsmap": (ModalityMapAxisArrays, 0),
    "_varmap": (ModalityMapAxisArrays, 1),
}

logger = get_logger(__name__)


# ---------------------------------------------------------------------------------------------------
# Typed access
# ---------------------------------------------------------------------------------------------------


def get_anndata_mod(mdata: md.MuData, mod_name: str) -> ad.AnnData:
    """Return an AnnData modality and fail explicitly if the modality is nested MuData."""
    mod = mdata.mod[mod_name]
    if not isinstance(mod, ad.AnnData):
        raise TypeError(f"Expected AnnData modality '{mod_name}', got {type(mod).__name__}.")
    return mod


def get_mudata_mod(mdata: md.MuData, mod_name: str) -> md.MuData:
    """Return a nested MuData modality and fail explicitly if the modality is AnnData."""
    mod = mdata.mod[mod_name]
    if not isinstance(mod, md.MuData):
        raise TypeError(f"Expected MuData modality '{mod_name}', got {type(mod).__name__}.")
    return mod


def get_anndata(mdata: md.MuData, modality: str) -> ad.AnnData:
    """Return the modality-specific AnnData object with proper typing."""
    return get_anndata_mod(mdata, modality)


def get_mudata(mdata: Any) -> md.MuData:
    """Return the MuData object with proper typing."""
    return cast(md.MuData, mdata)


def mutable_mudata_mod(mdata: md.MuData) -> MutableMuDataMod:
    return cast(MutableMuDataMod, mdata.mod)


# ---------------------------------------------------------------------------------------------------
# Copying
# ---------------------------------------------------------------------------------------------------


def _copy_part(value: Any) -> Any:
    """Copy one stored part: pandas, numpy and sparse containers by their own ``copy()``, everything else deeply."""
    if isinstance(value, dict):
        return copy_module.deepcopy(value)
    if hasattr(value, "copy"):
        return value.copy()
    return copy_module.deepcopy(value)


def copy_mudata(mdata: md.MuData) -> md.MuData:
    """Copy ``mdata`` without rebuilding its bookkeeping.

    The source is not modified. A container edited by hand (``mdata.mod[name] = adata`` without
    ``update()``) is copied as it is. Views and backed objects go through ``MuData.copy()``.
    """
    if mdata.is_view or mdata.isbacked:
        return mdata.copy()
    copied = md.MuData.__new__(md.MuData)
    for attribute_name, value in vars(mdata).items():
        if attribute_name == "_mod":
            modalities = ModDict()
            for modality_name, modality in value.items():
                modalities[modality_name] = modality.copy()
            copied_value: Any = modalities
        elif attribute_name in _ALIGNED_MAPPING_CLASSES:
            mapping_class, axis = _ALIGNED_MAPPING_CLASSES[attribute_name]
            copied_value = mapping_class(
                copied, axis=axis, store={key: _copy_part(item) for key, item in value.items()}
            )
        elif isinstance(value, MuDataFileManager):
            copied_value = MuDataFileManager()  # the copy is not backed by the source file
        else:
            copied_value = _copy_part(value)
        copied.__dict__[attribute_name] = copied_value
    return copied


# ---------------------------------------------------------------------------------------------------
# Finishing the bookkeeping after a modality changed
# ---------------------------------------------------------------------------------------------------


def _has_plain_global_var(mdata: md.MuData) -> bool:
    """Whether the global var is the layout msmu readers produce: ``axis=0``, no global var columns, no
    varm beyond the modality masks and no varp.
    """
    return (
        mdata.axis == 0
        and mdata._var.shape[1] == 0
        and set(mdata._varm.keys()) <= set(mdata.mod.keys())
        and len(mdata._varp) == 0
    )


def _concatenate_var_names(modalities: list[ad.AnnData]) -> pd.Index:
    return pd.Index([]).append([modality.var_names for modality in modalities])


def _rebuild_global_var(mdata: md.MuData, global_var_names: pd.Index | None = None) -> None:
    """Rebuild the global var, ``varmap`` and ``varm`` masks in place.

    The plain layout with unique var names is built directly: the modalities' var names in modality
    order, with contiguous 1..n blocks. Any other container goes through ``update_var()``, which keeps
    the existing rows in place and appends a renamed modality's new names last; either state is one
    ``update()`` leaves unchanged. ``global_var_names`` is the concatenated var names when the caller
    already built them.
    """
    modalities = list(mdata.mod.items())
    modality_names = [name for name, _ in modalities]
    if global_var_names is None:
        global_var_names = _concatenate_var_names([modality for _, modality in modalities])
    if not (_has_plain_global_var(mdata) and global_var_names.is_unique):
        mdata.update_var()
        return

    global_var_names.name = mdata._var.index.name
    # mudata's global var has no columns and its empty columns Index is string-typed (the label dtype
    # of the modality columns it drops); pandas' default RangeIndex would change the provenance hash.
    mdata._var = pd.DataFrame(index=global_var_names, columns=pd.Index([], dtype="str"))
    for removed_name in [name for name in list(mdata._varmap.keys()) if name not in modality_names]:
        del mdata._varmap[removed_name]
    block_start = 0
    for modality_name, modality in modalities:
        varmap = np.zeros(len(global_var_names), dtype=MODALITY_MAP_DTYPE)
        varmap[block_start : block_start + modality.n_vars] = np.arange(
            1, modality.n_vars + 1, dtype=MODALITY_MAP_DTYPE
        )
        mdata._varmap[modality_name] = varmap
        mdata._varm[modality_name] = varmap > 0
        block_start += modality.n_vars
    # mudata caches a sha1 of every modality's var index and columns to skip update_var() when nothing
    # changed; leave it as an update would. (The sha1 runs over the array of string pointers, so with
    # arrow-backed indexes it never hits and mudata re-aligns on every copy, which is why copy_mudata
    # exists.)
    mdata._varhash = {
        modality_name: (
            sha1(np.ascontiguousarray(modality.var.index.values)).hexdigest(),
            sha1(np.ascontiguousarray(modality.var.columns.values)).hexdigest(),
        )
        for modality_name, modality in modalities
    }


def refresh_global_obs(mdata: md.MuData) -> None:
    """Rebuild the global obs, ``obsmap`` and ``obsm`` masks in place with ``update_obs()``; mudata's change
    cache is cleared first so observations relabelled in place are not skipped.
    """
    mdata.__dict__.pop("_obshash", None)
    mdata.update_obs()


def _check_observations_within_container(mdata: md.MuData, adata: ad.AnnData) -> None:
    if not mdata.obs.index.is_unique:
        raise ValueError("mdata.obs.index should be unique.")
    if not adata.obs.index.is_unique:
        raise ValueError("adata.obs.index should be unique.")
    missing_obs_names = adata.obs.index.difference(mdata.obs.index).tolist()
    if missing_obs_names:
        raise ValueError(
            f"adata.obs.index contains observations not present in mdata.obs.index: {missing_obs_names}"
        )


def add_modality(mdata: md.MuData, adata: ad.AnnData, mod_name: str) -> md.MuData:
    """Insert ``adata`` as modality ``mod_name`` and finish the bookkeeping; returns ``mdata`` itself.

    ``adata`` may cover part of the container's observations but none outside ``mdata.obs.index``.
    Global obs columns are pushed onto the new modality.
    """
    _check_observations_within_container(mdata, adata)
    mutable_mudata_mod(mdata)[mod_name] = adata
    refresh_global_obs(mdata)  # gives the new modality its obsmap, which push_obs aligns through
    mdata.push_obs()
    _rebuild_global_var(mdata)
    return mdata


def replace_modality(mdata: md.MuData, mod_name: str, adata: ad.AnnData) -> md.MuData:
    """Replace modality ``mod_name`` with ``adata`` and finish the bookkeeping.

    ``adata`` may drop or rename features but may not add observations outside ``mdata.obs.index``.
    Whenever the rebuild would go through ``update_var()`` (a layout that is not plain, or a var name
    shared with another modality) the old var rows are sliced away first: ``update_var()`` refuses a
    modality renamed and filtered at once, and the slice makes the replacement an insertion to it. The
    result is then a new object. Global obs columns are pushed onto the new modality, as in
    ``add_modality``.
    """
    if mod_name not in mdata.mod:
        raise KeyError(f"Modality '{mod_name}' not found in mdata.")
    _check_observations_within_container(mdata, adata)
    global_var_names = _concatenate_var_names(
        [adata if name == mod_name else modality for name, modality in mdata.mod.items()]
    )
    if not (_has_plain_global_var(mdata) and global_var_names.is_unique):
        old_var_names = mdata.mod[mod_name].var_names
        mdata = mdata[:, ~mdata.var_names.isin(old_var_names)].copy()
    mutable_mudata_mod(mdata)[mod_name] = adata
    _rebuild_global_var(mdata, global_var_names)
    refresh_global_obs(mdata)
    mdata.push_obs()
    return mdata


# ---------------------------------------------------------------------------------------------------
# Public utilities
# ---------------------------------------------------------------------------------------------------


@log_provenance
def reindex_obs(
    mdata: md.MuData,
    column: str,
) -> md.MuData:
    """Reindex the observation (`.obs`) of the `MuData` object to ensure consistency across modalities.

    Parameters:
        mdata: `MuData` with the requested column in both global `.obs` and every modality `.obs`.
        column: Column whose string-converted values become observation names.

    Returns:
        A copied `MuData` with global and modality observation indices replaced. The old index is retained as a column by `reset_index`.

    Notes:
        Missing columns raise `KeyError`. Use identifiers consistent across modalities; the input is unchanged.

    Examples:
        ```python
        import msmu as mm
        renamed = mm.utils.reindex_obs(mdata, column="sample")
        ```
    """
    mdata = copy_mudata(mdata)
    if column not in mdata.obs.columns:
        msg = f"Column '{column}' not found in mdata.obs."
        logger.error(msg)
        raise KeyError(msg)

    new_index = mdata.obs[column].astype(str)
    mdata.obs.reset_index(drop=False, inplace=True)
    mdata.obs.set_index(new_index, inplace=True, drop=False)
    for mod in mdata.mod.keys():
        adata = get_anndata_mod(mdata, mod)
        mod_obs = adata.obs
        if not isinstance(mod_obs, pd.DataFrame):
            msg = f"Expected DataFrame-backed obs for mdata.mod['{mod}'], got {type(mod_obs).__name__}."
            logger.error(msg)
            raise TypeError(msg)

        if column not in mod_obs.columns:
            msg = f"Column '{column}' not found in mdata.mod['{mod}'].obs."
            logger.error(msg)
            raise KeyError(msg)

        mod_new_index = mod_obs[column].astype(str)
        mod_obs.reset_index(drop=False, inplace=True)
        mod_obs.set_index(mod_new_index, inplace=True, drop=False)

    return mdata


__all__ = [
    "MutableMuDataMod",
    "add_modality",
    "copy_mudata",
    "get_anndata",
    "get_anndata_mod",
    "get_mudata",
    "get_mudata_mod",
    "mutable_mudata_mod",
    "refresh_global_obs",
    "reindex_obs",
    "replace_modality",
]
