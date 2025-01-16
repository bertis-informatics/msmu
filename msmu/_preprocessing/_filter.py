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
def add_q_value_filter(mdata: MuData, q_value: float | list[float] | dict[str, float]) -> MuData:
    mdata = mdata.copy()

    if isinstance(q_value, (float, int)):
        q_value = {"spectrum": q_value, "peptide": q_value, "protein": q_value}
    elif isinstance(q_value, list):
        q_value = {"spectrum": q_value[0], "peptide": q_value[1], "protein": q_value[2]}

    print(f"Filtering by q-value: {q_value}")

    spectrum_q = _get_column(mdata["psm"], "spectrum_q", "search_result")
    peptide_q = _get_column(mdata["psm"], "peptide_q", "search_result")
    protein_q = _get_column(mdata["psm"], "protein_q", "search_result")

    filter_df = pd.DataFrame(columns=["spectrum_q", "peptide_q", "protein_q"])
    filter_df["spectrum_q"] = spectrum_q < q_value["spectrum"]
    filter_df["peptide_q"] = peptide_q < q_value["peptide"]
    filter_df["protein_q"] = protein_q < q_value["protein"]

    if "filter" not in mdata["psm"].varm_keys():
        mdata["psm"].varm["filter"] = filter_df
    else:
        mdata["psm"].varm["filter"] = pd.concat([mdata["psm"].varm["filter"], filter_df], axis=1)

    if "filter" not in mdata["psm"].uns_keys():
        mdata["psm"].uns["filter"] = {"q_value": q_value}
    else:
        mdata["psm"].uns["filter"]["q_value"] = q_value

    return mdata


@check_type
def add_decoy_filter(mdata: MuData) -> MuData:
    mdata = mdata.copy()

    return mdata


@check_type
def add_precursor_purity_filter(mdata: MuData, purity_cutoff: float | list[float] | dict[str, float | int]) -> MuData:
    mdata = mdata.copy()

    if isinstance(purity_cutoff, (float, int)):
        purity_cutoff = {"min": purity_cutoff, "max": 1.0}
    elif isinstance(purity_cutoff, list):
        purity_cutoff = {"min": purity_cutoff[0], "max": purity_cutoff[1]}

    mdata["psm"].var["purity"] = calculate_precursor_purity(mdata, "psm")["purity"]

    filter_df = pd.DataFrame(columns=["purity"])
    filter_df["purity"] = calculate_precursor_purity(mdata, "psm")["purity"]

    if "filter" not in mdata["psm"].varm_keys():
        mdata["psm"].varm["filter"] = filter_df
    else:
        mdata["psm"].varm["filter"] = pd.concat([mdata["psm"].varm["filter"], filter_df], axis=1)

    if "filter" not in mdata["psm"].uns_keys():
        mdata["psm"].uns["filter"] = {"purity_cutoff": purity_cutoff}
    else:
        mdata["psm"].uns["filter"]["purity_cutoff"] = purity_cutoff

    return mdata


# @check_type
# def calculate_precursor_purity(mdata: MuData, mzml_folder: str | Path) -> MuData:
#     mdata = mdata.copy()

#     return mdata


@check_type
def apply_filter(mdata: MuData) -> MuData:
    mdata = mdata.copy()

    return mdata
