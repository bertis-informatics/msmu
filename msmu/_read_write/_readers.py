from pathlib import Path
from types import NoneType

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from ._diann_reader import DiannReader
from ._sage_reader import LfqSageReader, TmtSageReader


def read_sage(
    sage_output_dir: str | Path,
    sample_name: list[str],
    label: str,
    channel: list[str] | None = None,
    filename: list[str] | None = None,
) -> md.MuData:
    if label == "tmt":
        reader_cls = TmtSageReader
    elif label == "lfq":
        reader_cls = LfqSageReader
    else:
        raise ValueError("Argument label should be one of 'tmt', 'lfq'.")

    reader = reader_cls(
        sage_output_dir=sage_output_dir,
        sample_name=sample_name,
        channel=channel,
        filename=filename,
    )
    mdata = reader.read()

    return mdata


def read_diann():
    return NotImplementedError


def read_comet():
    return NotImplementedError


def read_protdiscov():
    return NotImplementedError


def read_maxquant():
    return NotImplementedError

def read_h5mu(h5mu_file: str | Path) -> md.MuData:
    mdata: md.MuData = md.read_h5mu(h5mu_file)

    return mdata

def merge_mudata(mdatas: dict[str, md.MuData]) -> md.MuData:
    adata_dict: dict = dict()
    peptide_list: list = list()
    #    protein_list: list = list() # for further feature
    #    ptm_list: list = list() # for further feature

    obs_ident: list = list()
    for name_, mdata in mdatas.items():
        for mod in mdata.mod_names:
            adata = mdata[mod].copy()
            if adata.uns["level"] == "psm":
                psm_prefix: str = "psm"
                psm_name: str = f"{psm_prefix}_{name_}"
                adata_dict[psm_name] = adata

                obs_ident_df = adata.obs.copy()
                obs_ident_df["set"] = name_
                obs_ident.append(obs_ident_df)

            elif adata.uns["level"] == "peptide":
                peptide_list.append(adata)

    if len(peptide_list) > 0:
        adata_pep = ad.concat(peptide_list, uns_merge="unique", join="outer")
        adata_dict["peptide"] = adata_pep

    obs_ident_df: pd.DataFrame = pd.concat(obs_ident)

    merged_mdata: md.MuData = md.MuData(adata_dict)
    merged_mdata.obs["set"] = obs_ident_df["set"]
    merged_mdata.push_obs()
    merged_mdata.update_var()

    return merged_mdata


def mask_obs(
    mdata: md.MuData,
    mask_type: str,
    prefix: str | None = None,
    suffix: str | None = None,
    masking_list: list[str] | None = None,
) -> md.MuData:

    assert mask_type in [
        "blank",
        "gis",
    ], 'Argument "type" must one of "blank", "IRS (Internal Reference Standard)"'

    obs_df = mdata.obs.copy()
    obs_df["_obs"] = obs_df.index

    mask_column_name = f"is_{mask_type}"
    obs_df[mask_column_name] = False

    if isinstance(prefix, NoneType) == False:
        obs_df.loc[obs_df["_obs"].str.startswith(prefix), mask_column_name] = True
    elif isinstance(suffix, NoneType) == False:
        obs_df.loc[obs_df["_obs"].str.endswith(suffix), mask_column_name] = True
    elif isinstance(masking_list, NoneType) == False:
        for mask in masking_list:
            obs_df.loc[obs_df["_obs"] == mask, mask_column_name] = True

    obs_df = obs_df.drop("_obs", axis=1)

    mdata.obs[mask_column_name] = obs_df[mask_column_name]
    mdata.push_obs()

    return mdata


def add_modality(
    mdata: md.MuData, adata: ad.AnnData, mod_name: str, parent_mods: list[str]
) -> md.MuData:
    mdata.mod[mod_name] = adata

    if parent_mods:
        obsmap_list: list = list()
        for parent_mod in parent_mods:
            obsmap: np.array[int] = mdata.obsmap[parent_mod]
            obsmap_list.append(obsmap)
        merged_obsmap: np.array[int] = sum(obsmap_list)
        zero_indices: np.array[bool] = merged_obsmap == 0
        merged_obsmap: np.array[int] = np.arange(1, len(merged_obsmap) + 1, dtype=int)
        merged_obsmap[zero_indices] = 0

        mdata.obsmap[mod_name] = merged_obsmap

    else:
        raise ValueError("parent_mods should not be empty.")

    mdata.push_obs()
    mdata.update_var()

    return mdata
