from pathlib import Path
import re

import pandas as pd
from anndata import AnnData
from mudata import MuData

from .._utils import get_modality_dict, subset
from .._utils import subset, get_modality_dict, uns_logger

# def add_q_value_filter(
#    mdata: MuData,
#    modality: str,
#    threshold: float | list[float] | dict[str, float],
# ) -> MuData:
#    mdata = mdata.copy()
#    adata = mdata[modality]
#
#    # Format threshold
#    if isinstance(threshold, (float, int)):
#        threshold = {"spectrum": threshold, "peptide": threshold, "protein": threshold}
#    elif isinstance(threshold, list):
#        threshold = {
#            "spectrum": threshold[0],
#            "peptide": threshold[1],
#            "protein": threshold[2],
#        }
#
#    # Get q values
#    spectrum_q = _get_column(adata, "spectrum_q", "search_result")
#    peptide_q = _get_column(adata, "peptide_q", "search_result")
#    protein_q = _get_column(adata, "protein_q", "search_result")
#
#    # Get filter result
#    filter_result_spectrum = spectrum_q < threshold["spectrum"]
#    filter_result_peptide = peptide_q < threshold["peptide"]
#    filter_result_protein = protein_q < threshold["protein"]
#
#    # Save filter
#    _save_filter(
#        adata, "q_value_spectrum", filter_result_spectrum, threshold["spectrum"]
#    )
#    _save_filter(adata, "q_value_peptide", filter_result_peptide, threshold["peptide"])
#    _save_filter(adata, "q_value_protein", filter_result_protein, threshold["protein"])
#
#    return mdata


@uns_logger
def add_q_value_filter(
    mdata: MuData,
    threshold: float | list[float] | dict[str, float],
) -> MuData:
    mdata = mdata.copy()
    adata = mdata["feature"]

    q_val_categories = ["spectrum_q", "peptide_q", "protein_q"]
    var_cols = adata.var.columns
    q_val_cols = [x for x in var_cols if x in q_val_categories]

    # Format threshold
    if isinstance(threshold, (float, int)):
        threshold_dict = {q_col: threshold for q_col in q_val_cols}
    elif isinstance(threshold, list):
        threshold_dict = {q_col: threshold[i] for i, q_col in enumerate(q_val_cols)}
    elif isinstance(threshold, dict):
        threshold_dict = {q_col: threshold[q_col] for q_col in q_val_cols}

    q_values = adata.var[q_val_cols]

    # Get filter result
    filter_result = dict()
    for q_col in q_val_cols:
        filter_result[q_col] = q_values[q_col] < threshold_dict[q_col]
        _save_filter(adata, q_col, filter_result[q_col], threshold_dict[q_col])

    return mdata


@uns_logger
def add_prefix_filter(
    mdata: MuData,
    prefix: str | tuple = ("rev_", "contam_"),
) -> MuData:
    mdata = mdata.copy()
    adata = mdata["feature"]

    # Get protein columns
    proteins = _get_column(adata, "proteins", "search_result")

    # Remove prefix matched from proteins column
    pat = r"(^|;)\s*(?:" + "|".join(map(re.escape, prefix if isinstance(prefix, (tuple, list)) else (prefix,))) + r")"
    # adata.var["proteins"] = proteins.apply(lambda x: _remove_prefix_matched(x, prefix))

    # Get filter result
    filter_result = ~proteins.str.contains(pat, regex=True, na=False)

    # Save filter
    _save_filter(adata, "prefix", filter_result, list(prefix))

    return mdata


@uns_logger
def add_precursor_purity_filter(
    mdata: MuData,
    threshold: float,
) -> MuData:
    mdata = mdata.copy()
    adata = mdata["feature"]

    # Get filter result
    filter_result = mdata["feature"].var["purity"] > threshold

    # Save filter
    _save_filter(adata, "purity", filter_result, threshold)

    return mdata


@uns_logger
def add_decoy_filter(
    mdata: MuData,
) -> MuData:
    mdata = mdata.copy()
    adata = mdata["feature"]

    decoy = _get_column(adata, "decoy", "search_result")
    filter_result = decoy == 0

    # Save filter
    _save_filter(adata, "decoy", filter_result, 0)

    return mdata


@uns_logger
def add_contaminant_filter(
    mdata: MuData,
) -> MuData:
    mdata = mdata.copy()
    adata = mdata["feature"]

    contaminant = _get_column(adata, "contaminant", "search_result")
    filter_result = contaminant == 0

    # Save filter
    _save_filter(adata, "contaminant", filter_result, 0)

    return mdata


@uns_logger
def add_all_nan_filter(
    mdata: MuData,
    modality: str,
) -> MuData:
    mdata = mdata.copy()
    adata = mdata[modality]

    # Get filter result
    filter_result = ~adata.to_df().isna().all(axis=0)

    # Save filter
    _save_filter(adata, "nan", filter_result, True)

    return mdata


@uns_logger
def apply_filter(
    mdata: MuData,
    modality: str | list,
) -> MuData:
    if isinstance(modality, str):
        mods = [modality]
    elif isinstance(modality, list):
        mods = modality
    else:
        raise ValueError("modality should be a string or a list of strings")

    # Apply filter
    for mod in mods:
        mdata = _apply_filter(mdata, mod)

    return mdata.copy()


def _apply_filter(
    mdata: MuData,
    modality: str,
) -> MuData:

    mdata = mdata.copy()

    # Get filter result
    if "filter" not in mdata[modality].varm_keys():
        raise ValueError("Filter result is not found in the data")
    filter_df = mdata[modality].varm["filter"]

    mdata = subset(
        mdata=mdata,
        modality=modality,
        cond_var=filter_df.all(axis=1),
    )

    return mdata


def _get_column(
    adata: AnnData,
    colname: str,
    key: str,
) -> pd.Series:
    # Check if the column is in the var or varm
    if colname in adata.var.columns:
        return adata.var[colname]
    elif colname in adata.varm[key].columns:
        return adata.varm[key][colname]
    else:
        raise ValueError(f"{colname} not found in the data")


def _remove_prefix_matched(row: pd.Series, prefix: str | tuple) -> pd.Series:
    matched_list = []
    for x in row.split(";"):
        if str(x).startswith(prefix):
            continue
        else:
            matched_list.append(x)

    matched_str = ";".join(matched_list)
    if row == matched_str:
        return row
    else:
        return ""


def _save_filter(
    adata: AnnData,
    key: str,
    filter_result: pd.Series,
    content: str | list | tuple | dict,
) -> None:
    key = f"filter_{key}"

    _save_filter_result(adata, key, filter_result)
    _save_filter_content(adata, key, content)
    return None


def _save_filter_result(
    adata: AnnData,
    key: str,
    filter_result: pd.Series,
) -> None:
    if "filter" not in adata.varm_keys():
        adata.varm["filter"] = filter_result.to_frame(name=key)
    else:
        adata.varm["filter"][key] = filter_result
    return None


# TODO: make keeping ordered dict format
def _save_filter_content(
    adata: AnnData,
    key: str,
    content: dict,
) -> None:
    if "filter" not in adata.uns_keys():
        adata.uns["filter"] = {key: content}
    else:
        adata.uns["filter"][key] = content
    return None