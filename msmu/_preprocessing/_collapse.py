import warnings
from typing import Literal

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from .._utils._mudata import get_anndata_mod
from .._core._provenance import uns_logger
from ..logging_utils import get_logger

logger = get_logger(__name__)


def _nansum_preserve_nan(arr: np.ndarray, axis: int) -> np.ndarray:
    """`np.nansum` returns 0 for all-NaN slices; override to return NaN for proteomics LFQ semantics."""
    aggregated = np.nansum(arr, axis=axis)
    if arr.size == 0:
        return aggregated
    all_nan_mask = np.all(np.isnan(arr), axis=axis)
    aggregated = np.where(all_nan_mask, np.nan, aggregated)
    return aggregated


def _nansum_log2_space(arr: np.ndarray, axis: int) -> np.ndarray:
    """Sum equivalent for log2-transformed data: 2^x → sum in linear space → log2.

    Equivalent to log2(sum(2^x)) — the proper LFQ rollup when intensities are in log2.
    Direct sum of logs would yield log of the product, which is semantically wrong.
    """
    linear_arr = np.exp2(arr)
    summed_linear = _nansum_preserve_nan(linear_arr, axis=axis)
    with np.errstate(divide="ignore"):
        return np.log2(summed_linear)


_AGG_FUNCTIONS = {
    "sum": _nansum_preserve_nan,
    "max": np.nanmax,
    "median": np.nanmedian,
    "mean": np.nanmean,
}


@uns_logger
def collapse_obs(
    mdata: md.MuData,
    sample_key: str,
    agg_method: Literal["sum", "max", "median", "mean"] = "sum",
    layer: str | None = None,
    log_transformed: bool = False,
) -> md.MuData:
    """
    Collapse obs rows by ``sample_key``, aggregating duplicates into one row per group.

    Common use cases (with recommended ``agg_method``):

    - **LFQ fractions** (sample × LC fraction → sample): ``agg_method="sum"``.
      Standard LFQ rollup since each fraction holds a different portion of the
      sample's peptides.
    - **Technical replicates** (sample × injection/process repeats → sample):
      ``agg_method="median"`` or ``"mean"``. Replicates measure the same analyte,
      so the central tendency is the right estimate (``"sum"`` would inflate).
    - **Other obs-axis duplicates** (plate replicates, etc.): pick ``agg_method``
      that matches the meaning of the duplication.

    The effect on a given modality depends on its ``var`` structure:

    - When ``var`` rows are file-specific (e.g. DIA-NN psm where the var index is
      ``filename.Precursor.Id``), each ``(sample, var)`` cell still holds at most
      one non-NaN value, so this function only reduces the obs dimension; the actual
      cross-fraction precursor aggregation happens later in ``to_peptide``.
    - When ``var`` rows are identity-based (e.g. DDA-LFQ peptide modality where the
      var index is the peptide identity), a single ``(sample, var)`` cell may collect
      values from multiple obs rows, and ``agg_method`` performs the real rollup.

    Workflow contexts:
        - DIA-NN fractionated: read → psm/precursor → ``collapse_obs`` →
          ``to_peptide`` → ``to_protein``.
        - DDA-LFQ fractionated: read → psm + peptide → ``collapse_obs``
          (peptide gets real sum) → ``to_protein``.
        - Any workflow with technical replicates: ``collapse_obs(agg_method="median")``
          early in the pipeline (typically before normalisation).

    Applied uniformly across all modalities so the obs axis stays consistent.

    Parameters:
        mdata: MuData whose obs rows contain duplicates to be collapsed.
        sample_key: Column in ``obs`` identifying the group each row belongs to.
        agg_method: Aggregation across rows of the same group for each
            ``(group, feature)`` pair. One of ``"sum"``, ``"max"``, ``"median"``,
            ``"mean"``. NaN values are skipped; an all-NaN group yields NaN
            (not 0) for ``"sum"`` as well.
        layer: Layer to aggregate. If None, ``.X`` is used.
        log_transformed: Whether the input quantification is in log2-space.
            Defaults to ``False`` because ``collapse_obs`` is typically called before
            ``log2_transform`` (read → collapse → log2 → normalise → ...). When
            ``log_transformed=True`` is combined with ``agg_method="sum"``, the function
            internally converts back to linear space, sums, and re-applies log2
            (i.e. ``log2(sum(2^x))``) — the correct LFQ rollup for log2 input. Other
            ``agg_method`` choices (``max``, ``median``, ``mean``) operate directly in
            the given space; ``mean`` on log2 data corresponds to the geometric mean of
            linear intensities.

    Returns:
        New MuData with collapsed obs. Obs columns uniform within a group retain
        their scalar value; non-uniform columns become lists preserving the original
        row order, so per-row metadata (e.g. filenames) is not lost.

    Notes:
        - ``obsm``/``obsp`` are not propagated through the collapse.
        - ``var`` and modality ``uns`` are preserved.
    """
    if agg_method not in _AGG_FUNCTIONS:
        raise ValueError(f"agg_method '{agg_method}' not recognised. Choose from {sorted(_AGG_FUNCTIONS)}.")

    if log_transformed and agg_method == "sum":
        effective_agg_function = _nansum_log2_space
    else:
        effective_agg_function = _AGG_FUNCTIONS[agg_method]

    aggregated_mods: dict[str, ad.AnnData] = {}
    for modality_name in mdata.mod.keys():
        adata = get_anndata_mod(mdata, modality_name)
        if sample_key not in adata.obs.columns:
            raise KeyError(f"sample_key '{sample_key}' not found in obs of modality '{modality_name}'.")
        aggregated_mods[modality_name] = _collapse_anndata_obs(
            adata=adata,
            sample_key=sample_key,
            agg_function=effective_agg_function,
            layer=layer,
        )

    collapsed_mdata = md.MuData(aggregated_mods)
    for uns_key, uns_value in mdata.uns.items():
        collapsed_mdata.uns[uns_key] = uns_value

    return collapsed_mdata


def _collapse_anndata_obs(
    adata: ad.AnnData,
    sample_key: str,
    agg_function,
    layer: str | None,
) -> ad.AnnData:
    raw_arr = adata.X if layer is None else adata.layers[layer]
    if hasattr(raw_arr, "toarray"):
        raw_arr = raw_arr.toarray()
    raw_arr = np.asarray(raw_arr, dtype=float)

    group_values = adata.obs[sample_key].to_numpy()
    unique_samples, group_row_indices = _ordered_groupby_indices(group_values)

    n_samples = len(unique_samples)
    n_features = raw_arr.shape[1]
    aggregated_x = np.full((n_samples, n_features), np.nan, dtype=float)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        for sample_position, row_indices in enumerate(group_row_indices):
            block = raw_arr[row_indices, :]
            aggregated_x[sample_position, :] = agg_function(block, axis=0)

    aggregated_obs = _aggregate_obs_metadata(
        obs=adata.obs,
        sample_key=sample_key,
        unique_samples=unique_samples,
        group_row_indices=group_row_indices,
    )

    collapsed_adata = ad.AnnData(
        X=aggregated_x,
        obs=aggregated_obs,
        var=adata.var.copy(),
    )
    for uns_key, uns_value in adata.uns.items():
        collapsed_adata.uns[uns_key] = uns_value
    return collapsed_adata


def _ordered_groupby_indices(values: np.ndarray) -> tuple[list, list[np.ndarray]]:
    """Return unique values in first-occurrence order and the row indices per group."""
    seen_groups: dict = {}
    for row_index, value in enumerate(values):
        seen_groups.setdefault(value, []).append(row_index)
    unique_values = list(seen_groups.keys())
    group_row_indices = [np.array(seen_groups[value], dtype=int) for value in unique_values]
    return unique_values, group_row_indices


def _aggregate_obs_metadata(
    obs: pd.DataFrame,
    sample_key: str,
    unique_samples: list,
    group_row_indices: list[np.ndarray],
) -> pd.DataFrame:
    """For each sample group, keep scalar if column is uniform; else collapse into a list."""
    aggregated_records: list[dict] = []
    for sample_value, row_indices in zip(unique_samples, group_row_indices):
        record: dict = {sample_key: sample_value}
        sub_obs = obs.iloc[row_indices]
        for column in obs.columns:
            if column == sample_key:
                continue
            values_in_group = sub_obs[column].tolist()
            unique_in_group = pd.unique(pd.Series(values_in_group))
            if len(unique_in_group) == 1:
                record[column] = unique_in_group[0]
            else:
                record[column] = values_in_group
        aggregated_records.append(record)

    aggregated_obs = pd.DataFrame.from_records(aggregated_records, columns=obs.columns)
    aggregated_obs.index = pd.Index([str(sample) for sample in unique_samples], name=obs.index.name)
    return aggregated_obs
