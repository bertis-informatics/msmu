from functools import reduce
from pathlib import Path
from typing import Literal

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from ._diann_reader import DiannReader
from ._sage_reader import LfqSageReader, TmtSageReader


def read_sage(
    sage_output_dir: str | Path,
    label: Literal["tmt", "lfq"],
    sample_name: list[str] | None = None,
    channel: list[str] | None = None,
    filename: list[str] | None = None,
    meta: pd.DataFrame | None = None,
    sample_col: str | None = None,
    channel_col: str | None = None,
    filename_col: str | None = None,
) -> md.MuData:
    """
    Reads Sage output and returns a MuData object.

    Args:
        sage_output_dir (str | Path): Path to the Sage output directory.
        label (Literal["tmt", "lfq"]): Label for the Sage output ('tmt' or 'lfq').
        sample_name (list[str] | None): List of sample names.
        channel (list[str] | None): List of TMT channels.
        filename (list[str] | None): List of filenames for LFQ.
        meta (pd.DataFrame | None): Metadata DataFrame.
        sample_col (str | None): Column name for sample names in metadata.
        channel_col (str | None): Column name for TMT channels in metadata.
        filename_col (str | None): Column name for filenames in metadata.

    Returns:
        md.MuData: A MuData object containing the Sage data.
    """
    if label == "tmt":
        reader_cls = TmtSageReader
    elif label == "lfq":
        reader_cls = LfqSageReader
    else:
        raise ValueError("Argument label should be one of 'tmt', 'lfq'.")

    if meta is not None:
        sample_name = meta[sample_col].tolist()
        if label == "tmt":
            channel = meta[channel_col].tolist()
        if label == "lfq":
            filename:list[str] = meta[filename_col].tolist()
            filename = [f if f.endswith(".mzML") else f"{f}.mzML" for f in filename]

    reader = reader_cls(
        sage_output_dir=sage_output_dir,
        sample_name=sample_name,
        channel=channel,
        filename=filename,
    )
    mdata = reader.read()

    if meta is not None:
        meta_col_add = [x for x in meta.columns if x not in mdata.obs.columns]
        meta_add = meta[meta_col_add].set_index(sample_col, drop=False)
        mdata.obs = mdata.obs.join(meta_add)
    elif channel is not None:
        mdata.obs["channel"] = mdata.obs.index.map(
            {i: c for i, c in zip(sample_name, channel)}
        )

    mdata.obs = to_categorical(mdata.obs)
    mdata.push_obs()

    return mdata


def to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts object-type columns in a DataFrame to categorical.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with object columns converted to categorical.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.Categorical(df[col], categories=df[col].unique())

    return df


def read_diann(
    diann_output_dir: str | Path,
    sample_name: list[str] | None = None,
    filename: list[str] | None = None,
    meta: pd.DataFrame | None = None,
    sample_col: str | None = None,
    filename_col: str | None = None,
) -> md.MuData:

    if meta is not None:
        sample_name = meta[sample_col].tolist()
        filename = meta[filename_col].tolist()

    mdata: md.MuData = DiannReader(
        diann_output_dir=diann_output_dir,
        sample_name=sample_name,
        filename=filename,
    ).read()

    mdata.obs = to_categorical(mdata.obs)
    mdata.push_obs()

    return mdata


def read_comet():
    return NotImplementedError


def read_protdiscov():
    return NotImplementedError


def read_maxquant():
    return NotImplementedError


def read_h5mu(h5mu_file: str | Path) -> md.MuData:
    """
    Reads an H5MU file and returns a MuData object.

    Args:
        h5mu_file (str | Path): Path to the H5MU file.

    Returns:
        md.MuData: A MuData object.
    """
    return md.read_h5mu(h5mu_file)


def merge_mudata(mdatas: dict[str, md.MuData]) -> md.MuData:
    """
    Merges multiple MuData objects into a single MuData object.

    Args:
        mdatas (dict[str, md.MuData]): Dictionary of MuData objects to merge.

    Returns:
        md.MuData: A merged MuData object.
    """
    adata_dict = {}
    peptide_list = []
    #    protein_list: list = list() # for further feature
    #    ptm_list: list = list() # for further feature
    obs_ident = []
    protein_info:pd.DataFrame = pd.DataFrame()
    for name_, mdata in mdatas.items():
        for mod in mdata.mod_names:
            adata = mdata[mod].copy()
            if adata.uns["level"] == "psm":
                psm_name = f"psm_{name_}"
                adata_dict[psm_name] = adata
                obs_ident_df = adata.obs.copy()
                obs_ident_df["set"] = name_
                obs_ident.append(obs_ident_df)
            elif adata.uns["level"] == "peptide":
                peptide_list.append(adata)

        protein_info = pd.concat([protein_info, mdata.uns['protein_info']]).drop_duplicates()

    if peptide_list:
        adata_dict["peptide"] = ad.concat(
            peptide_list, uns_merge="unique", join="outer"
        )

    obs_ident_df = pd.concat(obs_ident)

    merged_mdata = md.MuData(adata_dict)
    merged_mdata.obs = pd.concat(
        [
            merged_mdata.obs,
            pd.concat([merged_mdata[mod].obs for mod in merged_mdata.mod_names if mod != "peptide"]),
        ],
        axis=1,
    )
    merged_mdata.obs["set"] = obs_ident_df["set"]
    merged_mdata.obs = to_categorical(merged_mdata.obs)
    merged_mdata.uns['protein_info'] = protein_info.reset_index(drop=True)
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
    """
    Masks observations in a MuData object based on the specified criteria.

    Args:
        mdata (md.MuData): Input MuData object.
        mask_type (str): Type of mask ('blank' or 'gis').
        prefix (str | None): Prefix for masking.
        suffix (str | None): Suffix for masking.
        masking_list (list[str] | None): List of specific observations to mask.

    Returns:
        md.MuData: Updated MuData object with masked observations.
    """
    if mask_type not in ["blank", "gis"]:
        raise ValueError('Argument "mask_type" must be one of "blank" or "gis".')

    obs_df = mdata.obs.copy()
    obs_df["_obs"] = obs_df.index
    mask_column_name = f"is_{mask_type}"
    obs_df[mask_column_name] = False

    if prefix:
        obs_df.loc[obs_df["_obs"].str.startswith(prefix), mask_column_name] = True
    if suffix:
        obs_df.loc[obs_df["_obs"].str.endswith(suffix), mask_column_name] = True
    if masking_list:
        obs_df.loc[obs_df["_obs"].isin(masking_list), mask_column_name] = True

    obs_df = obs_df.drop("_obs", axis=1)
    mdata.obs[mask_column_name] = obs_df[mask_column_name]
    mdata.push_obs()

    return mdata


def add_modality(
    mdata: md.MuData, adata: ad.AnnData, mod_name: str, parent_mods: list[str]
) -> md.MuData:
    """
    Adds a new modality to a MuData object.

    Args:
        mdata (md.MuData): Input MuData object.
        adata (ad.AnnData): AnnData object to add as a modality.
        mod_name (str): Name of the new modality.
        parent_mods (list[str]): List of parent modalities.

    Returns:
        md.MuData: Updated MuData object with the new modality.
    """
    if not parent_mods:
        raise ValueError("parent_mods should not be empty.")

    mdata.mod[mod_name] = adata

    obsmap_list = [mdata.obsmap[parent_mod] for parent_mod in parent_mods]
    merged_obsmap = sum(obsmap_list)
    zero_indices = merged_obsmap == 0
    merged_obsmap = np.arange(1, len(merged_obsmap) + 1, dtype=int)
    merged_obsmap[zero_indices] = 0

    mdata.obsmap[mod_name] = merged_obsmap
    mdata.push_obs()
    mdata.update_var()

    return mdata
