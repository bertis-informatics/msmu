from collections.abc import MutableMapping
from typing import Any, TypeAlias, cast

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..logging_utils import get_logger

MutableMuDataMod: TypeAlias = MutableMapping[str, ad.AnnData | md.MuData]
MutableMuDataObsMap: TypeAlias = MutableMapping[str, NDArray[np.integer]]

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
        mdata.update_var()

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
    mdata.update_var()

    return mdata


def reindex_obs(
    mdata: md.MuData,
    column: str,
) -> md.MuData:
    """
    Reindex the observation (obs) of the MuData object to ensure consistency across modalities.
    """
    mdata = mdata.copy()
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
    "get_anndata",
    "get_anndata_mod",
    "get_mudata",
    "get_mudata_mod",
    "get_mudata_mod_as_mutable",
    "mutable_mudata_mod",
    "mutable_mudata_obsmap",
    "reindex_obs",
]
