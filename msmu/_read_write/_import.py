from os import PathLike

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from .._utils._mudata import add_modality
from ..logging_utils import get_logger

logger = get_logger(__name__)


def _read_quant_data(quant_data: str | PathLike[str] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(quant_data, (str, PathLike)):
        return pd.read_csv(quant_data, sep="\t")
    if isinstance(quant_data, pd.DataFrame):
        return quant_data.copy()

    raise TypeError("quant_data must be a path to a tab-separated file or a pandas DataFrame.")


def _sample_names_from_obs(mdata: md.MuData, index_name: str | None) -> dict[str, str]:
    obs_df = mdata.obs.copy()
    if index_name is not None:
        if index_name not in obs_df.columns:
            raise KeyError(f"Column '{index_name}' not found in mdata.obs.")
        filenames = obs_df[index_name]
    else:
        filenames = obs_df.index

    return {str(filename).split(".mzML")[0]: str(obs_name) for filename, obs_name in zip(filenames, obs_df.index)}


def add_quant(
    mdata: md.MuData,
    quant_data: str | PathLike[str] | pd.DataFrame,
    quant_tool: str,
    index_name: str | None = None,
) -> md.MuData:
    """
    Add quantification data to the MuData object as a new modality.
    """
    if quant_tool != "flashlfq":
        raise ValueError(f"Unsupported quant_tool '{quant_tool}'. Supported tools: flashlfq.")

    quant = _read_quant_data(quant_data)
    logger.debug("Normalizing flashlfq quantification input with shape %s.", quant.shape)

    if "Sequence" not in quant.columns:
        raise ValueError("FlashLFQ quantification data must contain a 'Sequence' column.")

    quant = quant.set_index("Sequence", drop=True)
    quant = quant.rename_axis(index=None, columns=None)
    intensity_cols = [str(column) for column in quant.columns if str(column).startswith("Intensity_")]
    if not intensity_cols:
        raise ValueError("FlashLFQ quantification data must contain at least one 'Intensity_' column.")

    input_arr: pd.DataFrame = quant.loc[:, intensity_cols].copy()
    input_arr.columns = [column.split("Intensity_", maxsplit=1)[1] for column in intensity_cols]
    input_arr = input_arr.replace(0, np.nan)

    rename_dict = _sample_names_from_obs(mdata, index_name)
    input_arr.columns = [rename_dict.get(str(column), str(column)) for column in input_arr.columns]
    col_order = list(rename_dict.values())
    missing_columns = [column for column in col_order if column not in input_arr.columns]
    if missing_columns:
        raise ValueError(f"FlashLFQ quantification data is missing intensity columns for samples: {missing_columns}")

    input_arr = input_arr.loc[:, col_order]
    input_arr = input_arr.dropna(how="all")

    peptide_adata = ad.AnnData(X=input_arr.T)
    peptide_adata.uns["level"] = "peptide"

    mdata = add_modality(mdata=mdata, adata=peptide_adata, mod_name="peptide")

    logger.info("Added quantification modality 'peptide' using %s data.", quant_tool)
    logger.debug("Added peptide quantification matrix with shape %s.", input_arr.shape)

    mdata.update_obs()

    return mdata


__all__ = ["add_quant"]
