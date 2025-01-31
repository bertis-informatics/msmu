from mudata import MuData
from anndata import AnnData
from pathlib import Path
import pandas as pd

from ._calculate_precursor_purity import calculate_precursor_purity


def add_q_value_filter(
    mdata: MuData,
    modality: str,
    threshold: float | list[float] | dict[str, float],
) -> MuData:
    mdata = mdata.copy()
    adata = mdata[modality]

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
    filter_df = pd.DataFrame(columns=["low_spectrum_q", "low_peptide_q", "low_protein_q"])
    filter_df["low_spectrum_q"] = spectrum_q < threshold["spectrum"]
    filter_df["low_peptide_q"] = peptide_q < threshold["peptide"]
    filter_df["low_protein_q"] = protein_q < threshold["protein"]

    # Store filter result
    if "filter" not in adata.varm_keys():
        adata.varm["filter"] = filter_df
    else:
        adata.varm["filter"]["low_spectrum_q"] = filter_df["low_spectrum_q"]
        adata.varm["filter"]["low_peptide_q"] = filter_df["low_peptide_q"]
        adata.varm["filter"]["low_protein_q"] = filter_df["low_protein_q"]

    # Store filter threshold
    if "filter" not in adata.uns_keys():
        adata.uns["filter"] = {"q_value": threshold}
    else:
        adata.uns["filter"]["q_value"] = threshold

    return mdata


def add_prefix_filter(
    mdata: MuData,
    modality: str,
    prefix: str | tuple = ("rev_", "contam_"),
) -> MuData:
    mdata = mdata.copy()
    adata = mdata[modality]

    # Get protein columns
    proteins = _get_column(adata, "proteins", "search_result")

    # Remove prefix matched from proteins column
    adata.var["proteins_wo_prefix"] = proteins.apply(lambda x: _remove_prefix_matched(x, prefix))

    # Create filter result
    filter_df = pd.DataFrame(columns=["prefix"])
    filter_df["prefix"] = adata.var["proteins_wo_prefix"] != ""

    # Store filter result
    if "filter" not in adata.varm_keys():
        adata.varm["filter"] = filter_df
    else:
        adata.varm["filter"]["prefix"] = filter_df["prefix"]

    # Store filter prefix
    if "filter" not in adata.uns_keys():
        adata.uns["filter"] = {"prefix": prefix}
    else:
        adata.uns["filter"]["prefix"] = prefix
    return mdata


def add_precursor_purity_filter(
    mdata: MuData,
    modality: str,
    threshold: float | list[float] | dict[str, float | int],
    mzml_files: list[str | Path] = None,
    n_cores: int = 1,
) -> MuData:
    mdata = mdata.copy()
    adata = mdata[modality]

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
    if "purity" not in adata.var_keys():
        purity_df = calculate_precursor_purity(adata=adata, mzml_files=mzml_files, n_cores=n_cores)
        adata.var = adata.var.join(purity_df)
    else:
        raise ValueError("Precursor purity is already calculated. Please remove it before recalculating.")

    # Create filter result
    filter_df = pd.DataFrame(columns=["high_purity"])
    filter_df["high_purity"] = adata.var["purity"].between(
        left=threshold["min"],
        right=threshold["max"],
        inclusive="right",
    )

    # Store filter result
    if "filter" not in adata.varm_keys():
        adata.varm["filter"] = filter_df
    else:
        adata.varm["filter"]["high_purity"] = filter_df["high_purity"]

    # Store filter threshold
    if "filter" not in adata.uns_keys():
        adata.uns["filter"] = {"purity": threshold}
    else:
        adata.uns["filter"]["purity"] = threshold

    return mdata


def apply_filter(
    mdata: MuData,
    modality: str,
) -> MuData:
    mdata = mdata.copy()
    adata = mdata[modality]

    # Get filter result
    if "filter" not in adata.varm_keys():
        raise ValueError("Filter result is not found in the data")
    filter_df = adata.varm["filter"]

    # Apply filter
    filtered_adata = adata[:, filter_df.all(axis=1)].copy()

    # Store filter result
    mdata.mod[modality] = filtered_adata
    mdata.update_var()

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
    return ";".join([str(x) for x in row.split(";") if not str(x).startswith(prefix)])
