import anndata as ad
import mudata as md
import numpy as np


def add_modality(mdata: md.MuData, adata: ad.AnnData, mod_name: str, parent_mods: list[str]) -> md.MuData:
    """
    Add a modality to MuData while keeping obs/var mappings consistent.

    This helper is shared across readers and preprocessors so modality insertion
    follows one contract.
    """
    if not parent_mods:
        raise ValueError("parent_mods should not be empty.")

    mdata.mod[mod_name] = adata

    obsmap_list = [mdata.obsmap[parent_mod] for parent_mod in parent_mods]
    merged_obsmap = sum(obsmap_list)

    zero_indices = merged_obsmap == 0
    merged_obsmap = np.arange(1, len(merged_obsmap) + 1, dtype=int).reshape(-1, 1)
    merged_obsmap[zero_indices] = 0

    mdata.obsmap[mod_name] = merged_obsmap
    mdata.push_obs()
    mdata.update_var()

    return mdata
