from typing import Iterable

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from .._core._mudata import add_modality
from ..logging_utils import get_logger

logger = get_logger(__name__)


def get_modality_dict(
    mdata: md.MuData,
    level: str | None = None,
    modality: str | None = None,
) -> dict[str, ad.AnnData]:
    """Get modality data from MuData object."""
    if level is None and modality is None:
        raise ValueError("Either level or modality must be provided")

    if level is not None and modality is not None:
        logger.warning("Both level and modality are provided. Using level prior to modality.")

    mod_dict: dict = dict()
    if level is not None:
        for mod_name in mdata.mod.keys():
            if mdata.mod[mod_name].uns["level"] == level:
                mod_dict[mod_name] = mdata.mod[mod_name].copy()
    elif modality is not None:
        mod_dict[modality] = mdata.mod[modality].copy()

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
    """
    if isinstance(quant_data, str):
        quant = pd.read_csv(quant_data, sep="\t")
    elif isinstance(quant_data, pd.DataFrame):
        quant = quant_data.copy()
    else:
        raise ValueError("quant_data must be file for dataframe")

    if quant_tool == "flashlfq":
        logger.debug("Normalizing flashlfq quantification input with shape %s.", quant.shape)
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

        logger.info("Added quantification modality 'peptide' using %s data.", quant_tool)
        logger.debug("Added peptide quantification matrix with shape %s.", input_arr.shape)

    mdata.update_obs()

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
        mdata.mod[mod].obs.reset_index(drop=False, inplace=True)
        mdata.mod[mod].obs.set_index(new_index, inplace=True, drop=False)

    return mdata
