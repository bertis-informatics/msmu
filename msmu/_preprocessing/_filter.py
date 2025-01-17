from mudata import MuData
from anndata import AnnData
from pathlib import Path
import pandas as pd

from ._calculate_precursor_purity import calculate_precursor_purity

### THIS BLOCK SHOULD GO TO UTIL ###
from typing import get_type_hints


def check_type(func):
    def wrapper(*args, **kwargs):
        # Ensure no positional arguments are passed
        if args:
            raise TypeError("This function only accepts keyword arguments")

        # Get type hints, excluding the return type
        type_hints = get_type_hints(func)
        type_hints.pop("return", None)

        # Check for the correct number of arguments
        if len(kwargs) != len(type_hints):
            raise TypeError(f"Expected {len(type_hints)} arguments, got {len(kwargs)}")

        for key, value in kwargs.items():
            if key not in type_hints:
                raise TypeError(f"Unexpected argument '{key}'")

        return func(**kwargs)

    return wrapper


def _get_column(adata: AnnData, colname: str, key: str) -> str:
    # Check if the column is in the var or varm
    if colname in adata.var.columns:
        return adata.var[colname]
    elif colname in adata.varm[key].columns:
        return adata.varm[key][colname]
    else:
        raise ValueError(f"{colname} not found in the data")


@check_type
def add_q_value_filter(
    mdata: MuData,
    level: str,
    threshold: float | list[float] | dict[str, float],
) -> MuData:
    mdata = mdata.copy()
    adata = mdata[level]

    # Format threshold
    if isinstance(threshold, (float, int)):
        threshold = {"spectrum": threshold, "peptide": threshold, "protein": threshold}
    elif isinstance(threshold, list):
        threshold = {"spectrum": threshold[0], "peptide": threshold[1], "protein": threshold[2]}

    # Get q values
    spectrum_q = _get_column(adata, "spectrum_q", "search_result")
    peptide_q = _get_column(adata, "peptide_q", "search_result")
    protein_q = _get_column(adata, "protein_q", "search_result")

    # Create filter result
    filter_df = pd.DataFrame(columns=["spectrum_q", "peptide_q", "protein_q"])
    filter_df["spectrum_q"] = spectrum_q < threshold["spectrum"]
    filter_df["peptide_q"] = peptide_q < threshold["peptide"]
    filter_df["protein_q"] = protein_q < threshold["protein"]

    # Store filter result
    if "filter" not in adata.varm_keys():
        adata.varm["filter"] = filter_df
    else:
        adata.varm["filter"]["spectrum_q"] = filter_df["spectrum_q"]
        adata.varm["filter"]["peptide_q"] = filter_df["peptide_q"]
        adata.varm["filter"]["protein_q"] = filter_df["protein_q"]

    # Store filter threshold
    if "filter" not in adata.uns_keys():
        adata.uns["filter"] = {"q_value": threshold}
    else:
        adata.uns["filter"]["q_value"] = threshold

    return mdata


@check_type
def add_decoy_filter(
    mdata: MuData,
    level: str,
) -> MuData:
    mdata = mdata.copy()
    adata = mdata[level]

    return mdata


@check_type
def add_precursor_purity_filter(
    mdata: MuData,
    level: str,
    threshold: float | list[float] | dict[str, float | int],
    mzml_files: list[str | Path] = None,
) -> MuData:
    mdata = mdata.copy()
    adata = mdata[level]

    # Check if the argument is provided
    if mzml_files is None:
        if "mzml_files" in adata.uns:
            mzml_files: list[str | Path] = adata.uns["mzml_files"]
        else:
            raise ValueError("mzml_files should be provided or stored in mdata.uns['mzml_files']")

    # Format threshold
    if isinstance(threshold, (float, int)):
        threshold = {"min": threshold, "max": 1.0}
    elif isinstance(threshold, list):
        threshold = {"min": threshold[0], "max": threshold[1]}

    # Get precursor purity
    adata.var["purity"] = calculate_precursor_purity(
        adata=adata,
        mzml_files=mzml_files,
    )

    # Create filter result
    filter_df = pd.DataFrame(columns=["purity"])
    filter_df["purity"] = adata.var["purity"].between(threshold["min"], threshold["max"])

    # Store filter result
    if "filter" not in adata.varm_keys():
        adata.varm["filter"] = filter_df
    else:
        adata.varm["filter"]["purity"] = filter_df["purity"]

    # Store filter threshold
    if "filter" not in adata.uns_keys():
        adata.uns["filter"] = {"purity": threshold}
    else:
        adata.uns["filter"]["purity"] = threshold

    return mdata


@check_type
def apply_filter(mdata: MuData) -> MuData:
    mdata = mdata.copy()

    return mdata
