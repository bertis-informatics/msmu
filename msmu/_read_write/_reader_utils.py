from functools import reduce

import pandas as pd
import numpy as np
import mudata as md
import anndata as ad


# Utility functions for Readers
def merge_mudata(mdatas: dict[str, md.MuData]) -> md.MuData:
    """
    Merges multiple MuData objects into a single MuData object.

    Parameters:
        mdatas: Dictionary of MuData objects to merge.

    Returns:
        Merged MuData object.
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
                        elif "int" in uns_type or "float" in uns_type:
                            num_set = set(components_dict[mod][component][sub_comp].values())
                            if len(num_set) == 1:
                                merged_data[mod].uns[sub_comp] = num_set.pop()
                            else:
                                merged_data[mod].uns[sub_comp] = {
                                    k: v for k, v in components_dict[mod][component][sub_comp].items()
                                }
                        elif "NoneType" in uns_type:
                            none_set = set(components_dict[mod][component][sub_comp].values())
                            if len(none_set) == 1:
                                merged_data[mod].uns[sub_comp] = none_set.pop()
                            else:
                                merged_data[mod].uns[sub_comp] = {
                                    k: v for k, v in components_dict[mod][component][sub_comp].items()
                                }

    return merged_data


def add_modality(mdata: md.MuData, adata: ad.AnnData, mod_name: str, parent_mods: list[str]) -> md.MuData:
    """
    Adds a new modality to a MuData object.

    Args:
        mdata: Input MuData object.
        adata: AnnData object to add as a modality.
        mod_name: Name of the new modality.
        parent_mods: List of parent modalities.

    Returns:
        Updated MuData object with the new modality.
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


def to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts object-type columns in a DataFrame to categorical.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with object columns converted to categorical.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.Categorical(df[col], categories=df[col].unique())

    return df
