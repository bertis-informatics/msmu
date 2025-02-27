import pandas as pd
from pathlib import Path
from anndata import AnnData
from mudata import MuData

from ._calculate_precursor_purity import calculate_precursor_purity
from .._utils import subset


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

    # Get filter result
    filter_result_spectrum = spectrum_q < threshold["spectrum"]
    filter_result_peptide = peptide_q < threshold["peptide"]
    filter_result_protein = protein_q < threshold["protein"]

    # Save filter
    _save_filter(adata, "q_value_spectrum", filter_result_spectrum, threshold["spectrum"])
    _save_filter(adata, "q_value_peptide", filter_result_peptide, threshold["peptide"])
    _save_filter(adata, "q_value_protein", filter_result_protein, threshold["protein"])

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
    adata.var["proteins"] = proteins.apply(lambda x: _remove_prefix_matched(x, prefix))

    # Get filter result
    filter_result = adata.var["proteins"] != ""

    # Save filter
    _save_filter(adata, "prefix", filter_result, list(prefix))

    return mdata


def add_precursor_purity_filter(
    mdata: MuData,
    modality: str,
    threshold: float,
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

    # Get precursor purity
    if "purity" not in adata.var_keys():
        purity_df = calculate_precursor_purity(adata=adata, mzml_files=mzml_files, n_cores=n_cores)
        adata.var = adata.var.join(purity_df)
    else:
        raise ValueError("Precursor purity is already calculated. Please remove it before recalculating.")

    # Get filter result
    filter_result = adata.var["purity"] > threshold

    # Save filter
    _save_filter(adata, "purity", filter_result, threshold)

    return mdata


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


def apply_filter(
    mdata: MuData,
    modality: str | list,
) -> MuData:
    # Check modality
    if isinstance(modality, str):
        modality = [modality]

    # Apply filter
    for mod in modality:
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
    return ";".join([str(x) for x in row.split(";") if not str(x).startswith(prefix)])


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
        adata.varm["filter"] = pd.DataFrame(columns=[key], index=adata.var.index, data=filter_result)
    else:
        adata.varm["filter"][key] = filter_result
    return None


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
