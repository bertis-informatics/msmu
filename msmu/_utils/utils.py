from typing import Iterable
from collections.abc import Mapping, Sequence
import pandas as pd
import functools
import datetime
import logging
import numpy as np

import anndata as ad
import mudata as md

from .._read_write._reader_utils import add_modality


logger = logging.getLogger(__name__)


def serialize(obj):
    if isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, Mapping):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, Sequence) and not isinstance(obj, str):
        return [serialize(v) for v in obj]
    else:
        return obj


def uns_logger(func):
    @functools.wraps(func)
    def wrapper(mdata, *args, **kwargs):
        # Run the function
        result = func(mdata, *args, **kwargs)

        if not isinstance(mdata, md.MuData) or not isinstance(result, md.MuData):
            return result

        # Filter kwargs (e.g., None values)
        filtered_kwargs = {k: str(v) for k, v in kwargs.items() if v is not None}

        # Create log entry
        log_entry = {
            "function": func.__name__,
            "timestamp": datetime.datetime.now().isoformat(),
            "args": serialize(args),
            "kwargs": serialize(filtered_kwargs),
        }

        # Initialize log structure if not present
        if "_cmd" not in result.uns_keys():
            result.uns["_cmd"] = {}

        result.uns["_cmd"][str(len(result.uns["_cmd"]))] = log_entry

        return result

    return wrapper


def get_modality_dict(
    mdata: md.MuData,
    level: str | None = None,
    modality: str | None = None,
) -> dict[str, ad.AnnData]:
    """Get modality data from MuData object"""

    if (level == None) & (modality == None):
        raise ValueError("Either level or modality must be provided")

    if (level != None) & (modality != None):
        print("Both level and modality are provided. Using level prior to modality.")

    mod_dict: dict = dict()
    if level != None:
        for mod_name in mdata.mod_names:
            if mdata[mod_name].uns["level"] == level:
                mod_dict[mod_name] = mdata[mod_name].copy()

    elif modality != None:
        mod_dict[modality] = mdata[modality].copy()

    return mod_dict


def get_label(mdata: md.MuData) -> str:
    psm_mdatas: Iterable[ad.AnnData] = get_modality_dict(mdata=mdata, modality="psm").values()
    label_list: list[str] = [x.uns["label"] for x in psm_mdatas]

    if len(set(label_list)) == 1:
        label: str = label_list[0]
    else:
        raise ValueError("Multiple Label in Adatas! Please check label argument for reading search outputs!")

    return label


def add_quant(
    mdata: md.MuData,
    quant_data: str | pd.DataFrame,
    quant_tool: str,
    index_name: str | None = None,
) -> md.MuData:
    """
    Add quantification data to the MuData object as a new modality.

    Parameters:
        mdata: The MuData object to which the quantification data will be added.
        quant_data: The quantification data, either as a file path or a DataFrame.
        quant_tool: The tool used for quantification (e.g., "flashlfq").
        index_name: Optional; the name of the index column in the quantification data (when changed obs.index with reindex_obs function). default is None.

    Returns:
        The modified MuData object with the added quantification modality.
    """

    if isinstance(quant_data, str):
        quant = pd.read_csv(quant_data, sep="\t")
    elif isinstance(quant_data, pd.DataFrame):
        quant = quant_data.copy()
    else:
        raise ValueError("quant_data must be file for dataframe")

    if quant_tool == "flashlfq":
        quant = quant.set_index("Sequence", drop=True)
        quant = quant.rename_axis(index=None, columns=None)
        intensity_cols = [x for x in quant.columns if x.startswith("Intensity_")]
        input_arr = quant[intensity_cols]
        input_arr.columns = [x.split("Intensity_")[1] for x in intensity_cols]
        input_arr = input_arr.replace(0, np.nan)

        obs_df = mdata.obs.copy()
        if index_name is not None:
            filename = [x.split(".mzML")[0] for x in obs_df[index_name]]
        else:
            filename = [x.split(".mzML")[0] for x in obs_df.index]

        rename_dict = {k: v for k, v in zip(filename, obs_df.index)}
        input_arr = input_arr.rename(columns=rename_dict)
        col_order = list(rename_dict.values())
        input_arr = input_arr[col_order]
        input_arr = input_arr.dropna(how="all")

        peptide_adata = ad.AnnData(X=input_arr.T)
        peptide_adata.uns["level"] = "peptide"

        mdata = add_modality(mdata=mdata, adata=peptide_adata, mod_name="peptide", parent_mods=["psm"])

        logger.info(f"Added quantification modality 'peptide' using {quant_tool} data.")
        logger.info(f"Quantification data shape: {input_arr.shape}\n")

    mdata.update_obs()

    return mdata


def reindex_obs(
    mdata: md.MuData,
    column: str,
) -> md.MuData:
    """
    Reindex the observation (obs) of the MuData object to ensure consistency across modalities.

    Parameters:
        mdata (md.MuData): The MuData object containing the observations to reindex.
        column (str): The column name in mdata.obs to use for reindexing.

    Returns:
        md.MuData: The modified MuData object with reindexed observations.
    """
    mdata = mdata.copy()
    if column not in mdata.obs.columns:
        logger.error(f"Column '{column}' not found in mdata.obs.")
        raise

    new_index = mdata.obs[column].astype(str)
    mdata.obs.reset_index(drop=False, inplace=True)
    mdata.obs.set_index(new_index, inplace=True, drop=False)
    for mod in mdata.mod_names:
        mdata[mod].obs.reset_index(drop=False, inplace=True)
        mdata[mod].obs.set_index(new_index, inplace=True, drop=False)

    return mdata
