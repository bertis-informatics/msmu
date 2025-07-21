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
    # sample_name: list[str] | None = None,
    # channel: list[str] | None = None,
    # filename: list[str] | None = None,
    # meta: pd.DataFrame | None = None,
    # sample_col: str | None = None,
    # channel_col: str | None = None,
    # filename_col: str | None = None,
) -> md.MuData:
    """
    Reads Sage output and returns a MuData object.

    Args:
        sage_output_dir (str | Path): Path to the Sage output directory.
        label (Literal["tmt", "lfq"]): Label for the Sage output ('tmt' or 'lfq').
        # sample_name (list[str] | None): List of sample names.
        # channel (list[str] | None): List of TMT channels.
        # filename (list[str] | None): List of filenames for LFQ.
        # meta (pd.DataFrame | None): Metadata DataFrame.
        # sample_col (str | None): Column name for sample names in metadata.
        # channel_col (str | None): Column name for TMT channels in metadata.
        # filename_col (str | None): Column name for filenames in metadata.

    Returns:
        md.MuData: A MuData object containing the Sage data.
    """
    if label == "tmt":
        reader_cls = TmtSageReader
    elif label == "lfq":
        reader_cls = LfqSageReader
    else:
        raise ValueError("Argument label should be one of 'tmt', 'lfq'.")

    # if meta is not None:
    #     sample_name = meta[sample_col].tolist()
    #     if label == "tmt":
    #         channel = meta[channel_col].tolist()
    #     if label == "lfq":
    #         filename: list[str] = meta[filename_col].tolist()
    #         filename = [f if f.endswith(".mzML") else f"{f}.mzML" for f in filename]

    reader = reader_cls(
        sage_output_dir=sage_output_dir,
        # sample_name=sample_name,
        # channel=channel,
        # filename=filename,
    )
    mdata = reader.read()

    # if meta is not None:
    #     meta_col_add = [x for x in meta.columns if x not in mdata.obs.columns]
    #     meta_add = meta[meta_col_add].set_index(sample_col, drop=False)
    #     mdata.obs = mdata.obs.join(meta_add)
    # elif channel is not None:
    #     mdata.obs["channel"] = mdata.obs.index.map({i: c for i, c in zip(sample_name, channel)})

    # mdata.obs = to_categorical(mdata.obs)
    # mdata.push_obs()

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
    # sample_name: list[str] | None = None,
    # filename: list[str] | None = None,
    # meta: pd.DataFrame | None = None,
    # sample_col: str | None = None,
    # filename_col: str | None = None,
    # fasta: str | Path | None = None,
) -> md.MuData:

    # if meta is not None:
    #     sample_name = meta[sample_col].tolist()
    #     filename = meta[filename_col].tolist()

    mdata: md.MuData = DiannReader(
        diann_output_dir=diann_output_dir,
        # sample_name=sample_name,
        # filename=filename,
        # fasta=fasta,
    ).read()

    # if meta is not None:
    #     meta_col_add = [x for x in meta.columns if x not in mdata.obs.columns]
    #     meta_add = meta[meta_col_add].set_index(sample_col, drop=False)
    #     mdata.obs = mdata.obs.join(meta_add)

    # mdata.obs = to_categorical(mdata.obs)
    # mdata.push_obs()

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
        md.MuData: Merged MuData object.
    """
    mdata_components = dict()
    adata_components = dict()
    for name_, mdata in mdatas.items():
        if not isinstance(mdata, md.MuData):
            raise TypeError(
                f"Expected MuData object, got {type(mdata)} for {name_}. "
                "Please use read_h5mu or read_sage to read the data."
            )
        else:
            mdata_components = _decompose_data(data=mdata, name=name_, parent_dict=mdata_components)
            for mod in mdata.mod_names:
                adata_components = _decompose_data(
                    data=mdata[mod],
                    name=name_,
                    modality=mod,
                    parent_dict=adata_components,
                )

    # merge adata components
    merged_adatas = _merge_components(components_dict=adata_components)
    # merge mdata components
    merged_mdata = _merge_components(components_dict=mdata_components, adatas=merged_adatas)["mdata"].copy()

    merged_mdata.obs = to_categorical(merged_mdata.obs)
    merged_mdata.push_obs()
    merged_mdata.update_var()

    return merged_mdata


def _decompose_data(
    data: md.MuData | ad.AnnData,
    name: str,
    parent_dict: dict,
    modality: str | None = None,
) -> dict:
    components = [
        "adata",
        "var",
        "varm",
        "varp",
        "obs",
        "obsm",
        "obsp",
        "uns",
    ]

    if isinstance(data, md.MuData):
        if modality is not None:
            raise ValueError("If data is a MuData object, mod should be None. ")
        else:
            mod: str = "mdata"
            components = [
                component for component in components if component not in ["adata", "varm", "varp", "obsm", "obsp"]
            ]

    elif isinstance(data, ad.AnnData):
        if modality is None:
            raise ValueError("If data is an AnnData object, mod should be specified.")
        else:
            mod: str = modality

    else:
        raise TypeError(
            f"Expected MuData or AnnData object, got {type(data)} for {name}. "
            "Please use read_h5mu or read_sage to read the data."
        )

    components_dict = parent_dict.copy()
    if mod not in components_dict.keys():
        components_dict[mod] = {}
    for component in components:
        if component not in components_dict[mod].keys():
            components_dict[mod][component] = {}
        if component == "adata":
            components_dict[mod][component][name] = data.copy()
        else:
            tmp = getattr(data, component, None)
            if tmp is not None:
                if component == "var":
                    if "level" in data.uns:
                        if data.uns["level"] in ["precursor", "psm"]:
                            tmp["dataset"] = name
                    components_dict[mod][component][name] = tmp
                elif component == "obs":
                    tmp["dataset"] = name
                    components_dict[mod][component][name] = tmp
                elif component in ["varm", "varp", "obsm", "obsp", "uns"]:
                    for sub_comp in tmp.keys():
                        if sub_comp not in components_dict[mod][component].keys():
                            components_dict[mod][component][sub_comp] = {}
                        components_dict[mod][component][sub_comp][name] = tmp[sub_comp]

    return components_dict


def _merge_components(components_dict: dict, adatas: dict | None = None) -> dict:

    merged_data = dict()
    if adatas is not None:
        mods = ["mdata"]
        type_ = "mdata"

    else:
        mods = components_dict.keys()
        type_ = "adata"

    for mod in mods:
        if type_ == "mdata":
            merged_data[mod] = md.MuData(adatas)
        else:
            merged_data[mod] = ad.concat(components_dict[mod]["adata"].values(), join="outer")

        for component in components_dict[mod].keys():
            if component != "adata":
                if component in ["var"]:
                    setattr(
                        merged_data[mod],
                        component,
                        reduce(
                            lambda left, right: left.combine_first(right),
                            components_dict[mod][component].values(),
                        ),
                    )
                elif component == "obs":
                    merged_data[mod].obs = pd.concat(components_dict[mod][component].values(), axis=0)
                elif component in ["varm", "varp", "obsm", "obsp"]:
                    setattr(
                        merged_data[mod],
                        component,
                        {
                            k: reduce(
                                lambda left, right: left.combine_first(right),
                                v.values(),
                            )
                            for k, v in components_dict[mod][component].items()
                        },
                    )
                elif component == "uns":
                    for sub_comp in components_dict[mod][component].keys():
                        uns_type = set([type(v).__name__ for k, v in components_dict[mod][component][sub_comp].items()])
                        if len(uns_type) == 1:
                            uns_type = uns_type.pop()
                        else:
                            raise ValueError(f"Uns type for {sub_comp} in {mod} is not consistent: {uns_type}")
                        if "DataFrame" in uns_type:
                            dfs = components_dict[mod][component][sub_comp].values()
                            merged_data[mod].uns[sub_comp] = pd.concat(dfs, axis=0, ignore_index=True).drop_duplicates()
                        elif "dict" in uns_type:
                            merged_data[mod].uns[sub_comp] = {
                                k: v for k, v in components_dict[mod][component][sub_comp].items()
                            }
                        elif "list" in uns_type:
                            merged_data[mod].uns[sub_comp] = reduce(
                                lambda left, right: left + right,
                                components_dict[mod][component][sub_comp].values(),
                            )
                        elif "str" in uns_type:
                            str_set = set(components_dict[mod][component][sub_comp].values())
                            if len(str_set) == 1:
                                merged_data[mod].uns[sub_comp] = str_set.pop()
                            else:
                                merged_data[mod].uns[sub_comp] = {
                                    k: v for k, v in components_dict[mod][component][sub_comp].items()
                                }

    return merged_data


def add_modality(mdata: md.MuData, adata: ad.AnnData, mod_name: str, parent_mods: list[str]) -> md.MuData:
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
    
    zero_indices = (merged_obsmap == 0)
    merged_obsmap = np.arange(1, len(merged_obsmap) + 1, dtype=int).reshape(-1, 1)
    merged_obsmap[zero_indices] = 0

    mdata.obsmap[mod_name] = merged_obsmap
    mdata.push_obs()
    mdata.update_var()

    return mdata
