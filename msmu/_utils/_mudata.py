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
from numpy.typing import NDArray

from .._core._provenance import log_provenance
from ..logging_utils import get_logger

MutableMuDataMod: TypeAlias = MutableMapping[str, ad.AnnData | md.MuData]
MutableMuDataObsMap: TypeAlias = MutableMapping[str, NDArray[np.integer]]

# Aligned-mapping attributes of a MuData with the class and axis mudata builds them with. They hold a
# reference to their parent, so a copy is constructed for the new parent instead of deep-copied.
_ALIGNED_MAPPING_CLASSES: dict[str, tuple[type, int]] = {
    "_obsm": (AxisArrays, 0),
    "_varm": (AxisArrays, 1),
    "_obsp": (PairwiseArrays, 0),
    "_varp": (PairwiseArrays, 1),
    "_obsmap": (ModalityMapAxisArrays, 0),
    "_varmap": (ModalityMapAxisArrays, 1),
}
# dtype mudata gives obsmap/varmap in update(); 0 marks "not in this modality", 1.. the modality position.
MODALITY_MAP_DTYPE = np.uint32

logger = get_logger(__name__)


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


def get_mudata_mod_as_mutable(mdata: md.MuData) -> MutableMuDataMod:
    """Return the mutable modality mapping with MuData's runtime value types."""
    return mutable_mudata_mod(mdata)


def mutable_mudata_obsmap(mdata: md.MuData) -> MutableMuDataObsMap:
    return cast(MutableMuDataObsMap, mdata.obsmap)


# ---------------------------------------------------------------------------------------------------
# Copying a MuData without mudata's per-copy global var realignment.
#
# ``MuData.copy()`` rebuilds the object through ``MuData.__init__()``, which always calls ``update()``.
# ``update_var()`` re-aligns the global var with pandas ``isin`` over every var name; with pandas 3
# arrow strings that is a Python loop, so in psm containers every msmu step paid O(n_var) for a copy
# whose bookkeeping was already correct. The helpers below copy the parts as they are and rebuild
# only what msmu's own calls can leave stale, in exactly the layout mudata produces (verified
# bit-identical in provenance hashes and replay against mudata 0.4.1). They read mudata's private
# attributes, so mudata is pinned to its minor in pyproject and tests/test_utils_mudata_copy.py
# compares every rebuilt part against mudata's ``update()`` so a layout change breaks tests first.
# ---------------------------------------------------------------------------------------------------


def _copy_part(value: Any) -> Any:
    """Copy one stored part: pandas/numpy/sparse containers by their own copy(), everything else deeply."""
    if isinstance(value, dict):
        return copy_module.deepcopy(value)
    if hasattr(value, "copy"):
        return value.copy()
    return copy_module.deepcopy(value)


def _copy_mudata_parts(mdata: md.MuData) -> md.MuData:
    """Deep-copy every part of ``mdata`` into a new MuData without running ``MuData.__init__``.

    The global obs/var, the modality maps and the membership masks are copied as they are, so the
    copy carries the same bookkeeping as ``mdata`` (``copy_mudata`` then repairs the parts msmu can
    have left stale). Aligned mappings are re-created for the new parent; the file manager is fresh
    because the copy is not backed by the source file.
    """
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
            copied_value = MuDataFileManager()
        else:
            copied_value = _copy_part(value)
        copied.__dict__[attribute_name] = copied_value
    return copied


def _is_global_var_stale(mdata: md.MuData) -> bool:
    """O(n_mod) check that the global var still mirrors the modalities: total length, map keys and
    the first/last name of every modality block. Replacing a modality in place (``mdata.mod[name] =
    adata``) changes all three, which is what msmu's filters and summarisers do."""
    modalities = list(mdata.mod.items())
    if len(mdata._var) != sum(modality.n_vars for _, modality in modalities):
        return True
    if set(mdata._varmap.keys()) != {name for name, _ in modalities}:
        return True
    block_start = 0
    for _, modality in modalities:
        if modality.n_vars:
            block_end = block_start + modality.n_vars - 1
            if (
                mdata._var.index[block_start] != modality.var_names[0]
                or mdata._var.index[block_end] != modality.var_names[-1]
            ):
                return True
        block_start += modality.n_vars
    return False


def realign_global_var(mdata: md.MuData) -> None:
    """Rebuild the global var, ``varmap`` and ``varm`` membership masks from the modalities in place.

    In msmu containers (``axis=0``, no pulled global var columns, var names unique across
    modalities) mudata's ``update_var()`` result is simply the modality var names concatenated in
    modality order with contiguous 1..n blocks in ``varmap``; this builds exactly that without the
    O(n_var) ``isin`` passes. Any other container falls back to ``update_var()``. ``add_modality``
    calls it after inserting a modality and ``copy_mudata`` calls it on the copy when the global var
    is stale. Do not add it to a call that used to return a stale global var: that would change the
    call's recorded output hash and break replay of existing histories.
    """
    modalities = list(mdata.mod.items())
    modality_names = [name for name, _ in modalities]
    global_var_names = pd.Index([]).append([modality.var_names for _, modality in modalities])
    can_rebuild_directly = (
        mdata.axis == 0
        and mdata._var.shape[1] == 0
        and global_var_names.is_unique
        # Extra varm keys (user annotations, masks of removed modalities) and varp matrices would
        # have to be re-indexed; only mudata's update knows how.
        and set(mdata._varm.keys()) <= set(modality_names)
        and len(mdata._varp) == 0
    )
    if not can_rebuild_directly:
        mdata.update_var()
        return

    global_var_names.name = mdata._var.index.name
    # mudata's global var has no columns and its empty columns Index is string-typed (the label dtype
    # of the modality columns it drops); pandas' default RangeIndex would change the provenance hash.
    mdata._var = pd.DataFrame(index=global_var_names, columns=pd.Index([], dtype="str"))
    for stale_name in [name for name in list(mdata._varmap.keys()) if name not in modality_names]:
        del mdata._varmap[stale_name]
    block_start = 0
    for modality_name, modality in modalities:
        varmap = np.zeros(len(global_var_names), dtype=MODALITY_MAP_DTYPE)
        varmap[block_start : block_start + modality.n_vars] = np.arange(
            1, modality.n_vars + 1, dtype=MODALITY_MAP_DTYPE
        )
        mdata._varmap[modality_name] = varmap
        mdata._varm[modality_name] = varmap > 0
        block_start += modality.n_vars
    # mudata caches a sha1 of every modality's var index/columns to skip update_var() when nothing
    # changed. Refresh it as mudata would after an update. The sha1 runs over the object array of
    # string pointers, so it only hits while the same Python strings are reused (python-backed
    # indexes); arrow-backed indexes hand out new strings per access and never hit, which is why
    # mudata's own update_var() re-aligns on every copy in the first place.
    mdata._varhash = {
        modality_name: (
            sha1(np.ascontiguousarray(modality.var.index.values)).hexdigest(),
            sha1(np.ascontiguousarray(modality.var.columns.values)).hexdigest(),
        )
        for modality_name, modality in modalities
    }


def copy_mudata(mdata: md.MuData) -> md.MuData:
    """Copy ``mdata`` as ``MuData.copy()`` would, without re-aligning the global var from scratch.

    Returns a new MuData with copied modalities, global annotations and bookkeeping. The global var
    is rebuilt only when it no longer mirrors the modalities; the global obs is refreshed with
    mudata's own ``update_obs()`` (it is sample-sized, so this is cheap) with its change cache
    cleared, so the copy ends in the same state ``MuData.copy()`` produces. The source object is
    never modified. Views and backed objects go through ``MuData.copy()``.
    """
    if mdata.is_view or mdata.isbacked:
        return mdata.copy()
    copied = _copy_mudata_parts(mdata)
    if _is_global_var_stale(copied):
        realign_global_var(copied)
    copied.__dict__.pop("_obshash", None)
    copied.update_obs()
    return copied


def add_modality(
    mdata: md.MuData,
    adata: ad.AnnData,
    mod_name: str,
    parent_mods: list[str] | None = None,
) -> md.MuData:
    """
    Add a modality to MuData while keeping obs/var mappings consistent.

    This helper is shared across readers and preprocessors so modality insertion
    follows one contract. The new modality may cover all or part of the existing
    MuData observations, but it may not introduce observations outside
    ``mdata.obs.index``.
    """
    if parent_mods is not None:
        if not parent_mods:
            raise ValueError("parent_mods should not be empty.")

        obsmap_list = [mdata.obsmap[parent_mod] for parent_mod in parent_mods]
        merged_obsmap = obsmap_list[0].copy()
        for obsmap in obsmap_list[1:]:
            merged_obsmap = merged_obsmap + obsmap
        zero_indices = merged_obsmap == 0
        merged_obsmap = np.arange(1, len(merged_obsmap) + 1, dtype=int).reshape(-1, 1)
        merged_obsmap[zero_indices] = 0

        mutable_mudata_mod(mdata)[mod_name] = adata
        mutable_mudata_obsmap(mdata)[mod_name] = merged_obsmap

        mdata.push_obs()
        realign_global_var(mdata)

        return mdata

    if not mdata.obs.index.is_unique:
        raise ValueError("mdata.obs.index should be unique.")

    if not adata.obs.index.is_unique:
        raise ValueError("adata.obs.index should be unique.")

    mdata_obs_positions = {obs_name: index for index, obs_name in enumerate(mdata.obs.index)}
    missing_obs = [obs_name for obs_name in adata.obs.index if obs_name not in mdata_obs_positions]
    if missing_obs:
        raise ValueError(f"adata.obs.index contains observations not present in mdata.obs.index: {missing_obs}")

    new_obsmap = np.zeros((len(mdata.obs.index), 1), dtype=int)
    for adata_index, obs_name in enumerate(adata.obs.index, start=1):
        new_obsmap[mdata_obs_positions[obs_name], 0] = adata_index

    mutable_mudata_mod(mdata)[mod_name] = adata
    mutable_mudata_obsmap(mdata)[mod_name] = new_obsmap

    mdata.push_obs()
    realign_global_var(mdata)

    return mdata


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
    "MutableMuDataObsMap",
    "add_modality",
    "copy_mudata",
    "get_anndata",
    "get_anndata_mod",
    "get_mudata",
    "get_mudata_mod",
    "get_mudata_mod_as_mutable",
    "mutable_mudata_mod",
    "mutable_mudata_obsmap",
    "realign_global_var",
    "reindex_obs",
]
