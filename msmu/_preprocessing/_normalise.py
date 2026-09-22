from os import PathLike

import anndata as ad
import mudata as md
import numpy as np

from .._utils._mudata import get_anndata_mod
from .._core._provenance import log_provenance
from .._core._blockdiag import dense_block, is_sparse, sparse_apply_elementwise, to_observed_sparse
import pandas as pd
from ._normalisation import UnsharedSamplePairsError
from ..logging_utils import get_logger
from ._normalisation import Normalisation, NormalisationMethod, PTMAdjustmentMethod, PTMProteinAdjuster

logger = get_logger(__name__)


@log_provenance
def log2_transform(
    mdata: md.MuData,
    modality: str,
    layer: str | None = None,
) -> md.MuData:
    """
    Apply log2 transformation to the specified modality in MuData object.

    Parameters:
        mdata: MuData object to transform.
        modality: Modality to log2 transform.
        layer: Layer to transform. If None, the default layer (.X) will be used.

    Returns:
        Transformed MuData object.
    """
    mdata = mdata.copy()
    adata = get_anndata_mod(mdata, modality)

    if layer is None:
        raw_arr = adata.X
    else:
        raw_arr = adata.layers[layer]

    # log2 is elementwise, so on a sparse block-diagonal it transforms only the stored
    # (observed) values and keeps the matrix sparse -- absent cells stay absent.
    log2_arr = sparse_apply_elementwise(raw_arr, np.log2)

    if layer is None:
        adata.X = log2_arr
    else:
        adata.layers[layer] = log2_arr

    return mdata


@log_provenance
def scale_data(
    mdata: md.MuData,
    modality: str,
    layer: str | None = None,
) -> md.MuData:
    """
    Scale data in MuData object to have zero mean and unit variance.

    Parameters:
        mdata: MuData object to scale.
        modality: Modality to scale.
        layer: Layer to scale. If None, the default layer (.X) will be used.

    Returns:
        Scaled MuData object.
    """
    mdata = mdata.copy()
    adata = get_anndata_mod(mdata, modality)

    if layer is None:
        raw_arr: np.ndarray = adata.X
    else:
        raw_arr: np.ndarray = adata.layers[layer]

    # Scaling needs per-feature mean/std across all samples, so it densifies (NaN for absent).
    input_was_sparse = is_sparse(raw_arr)
    if input_was_sparse:
        input_dtype = raw_arr.dtype
        raw_arr = dense_block(raw_arr).astype(input_dtype)

    mean_arr: np.ndarray = np.nanmean(raw_arr, axis=0)
    std_arr: np.ndarray = np.nanstd(raw_arr, axis=0)
    scaled_arr = (raw_arr - mean_arr) / std_arr

    # Standardising leaves the observed pattern unchanged, so re-sparsify a sparse input back to a
    # sparse output -- recovering the memory the densify spent on the absent cells.
    if input_was_sparse:
        scaled_arr = to_observed_sparse(scaled_arr, dtype=input_dtype)

    if layer is None:
        adata.X = scaled_arr
    else:
        adata.layers[layer] = scaled_arr

    return mdata


@log_provenance
def normalise(
    mdata: md.MuData,
    method: NormalisationMethod,
    modality: str,
    layer: str | None = None,
    group_obs: str | None = None,
    group_var: str | None = None,
    batch_key: str | None = None,
    fraction_key: str | None = None,
    fraction: bool = False,
) -> md.MuData:
    """
    Normalise data in MuData object.

    Parameters:
        mdata: MuData object to normalise.
        method: Normalisation method to use. Options are 'median', 'median_center', 'quantile',
            'total_sum', 'pairwise_median'.
        modality: Modality to normalise.
        layer: Layer to normalise. If None, the default layer (.X) will be used.
        group_obs: Column name in ``adata.obs`` defining sample groups. If provided, normalisation is
            performed independently within each group. Missing group values are rejected.
            If None, no obs grouping is applied.
        group_var: Column name in ``adata.var`` defining feature groups (e.g. ``"filename"`` for
            fractionated TMT or label-free workflows). If provided, normalisation is performed
            independently within each group. If None, no var grouping is applied.
        batch_key: Deprecated alias for ``group_obs``.
        fraction_key: Deprecated alias for ``group_var``.
        fraction: Deprecated. If True, equivalent to ``group_var="filename"``.

    Returns:
        Normalised MuData object.

    Notes:
        When both ``group_obs`` and ``group_var`` are provided, normalisation is performed
        independently within each (obs-group × var-group) block.

        ``method="pairwise_median"`` aligns samples on the features they share: for every
        pair of samples it takes the median log2 difference over their co-observed features, and each
        sample's shift is the mean of its pairwise medians over all samples (the equal-weight
        least-squares solution, shifts summing to zero). It assumes that most of the features shared
        between two samples are unchanged. Per-sample medians of observed values are no longer aligned
        afterwards; this is intended, as those differences reflect which features each sample
        observed rather than loading. A true global change between conditions is removed, as with
        ``median``. Two samples of a block that observe no feature in common raise a ``ValueError``
        naming them (for example PSMs or precursors across runs or TMT plexes): normalise within
        ``group_obs``, or after summarising to a level where the samples share features.

        Every call records what was done to each sample of each block in
        ``adata.uns["normalisation"]``: a ``summary`` table (sample, obs_group, var_group, method,
        layer, n_observed_features, shift_log2, location_before, location_after) and, per block, the
        per-sample vectors plus, for ``pairwise_median``, the ``pair_median_log2`` and
        ``pair_shared_count`` matrices the shifts were derived from. Records for the same layer are
        replaced on a repeated call.
    """
    if batch_key is not None:
        logger.warning("`batch_key` is deprecated; use `group_obs` instead.")
        if group_obs is None:
            group_obs = batch_key
    if fraction_key is not None:
        logger.warning("`fraction_key` is deprecated; use `group_var` instead.")
        if group_var is None:
            group_var = fraction_key
    if fraction:
        logger.warning(
            "`fraction=True` is deprecated; use `group_var='filename'` instead.",
        )
        if group_var is None:
            group_var = "filename"

    axis: str = "obs"

    mdata = mdata.copy()
    adata: ad.AnnData = get_anndata_mod(mdata, modality)
    norm_cls: Normalisation = Normalisation(method=method, axis=axis)

    if layer is None:
        raw_arr: np.ndarray = adata.X
    else:
        raw_arr: np.ndarray = adata.layers[layer]

    if group_obs is not None and group_obs not in adata.obs.columns:
        raise KeyError(f"group_obs '{group_obs}' not found in adata.obs of modality '{modality}'.")
    if group_var is not None and group_var not in adata.var.columns:
        raise KeyError(f"group_var '{group_var}' not found in adata.var of modality '{modality}'.")
    if group_obs is not None and adata.obs[group_obs].isna().any():
        raise ValueError(
            f"group_obs '{group_obs}' contains missing values in modality '{modality}'. "
            "Fill in the missing sample groups before normalisation."
        )

    obs_groups = adata.obs[group_obs].to_numpy() if group_obs is not None else None
    var_groups = adata.var[group_var].to_numpy() if group_var is not None else None

    # Sparse-native methods (those the ``Normalisation`` object defines a ``_{method}_sparse`` rescaler
    # for) are computed directly on the sparse block-diagonal -- each obs row is rescaled from its own
    # stored values -- so the layer stays sparse instead of materialising the dense matrix, for the
    # common ungrouped case as well as grouped normalisation (each block normalised independently).
    # Quantile is not sparse-native (its per-sample rank mapping couples all samples), so it densifies.
    block_records: list[dict] = []
    try:
        if is_sparse(raw_arr) and norm_cls.is_sparse_native:
            normalised_arr = _normalise_per_group_sparse(raw_arr, obs_groups, var_groups, norm_cls, block_records)
        else:
            input_was_sparse = is_sparse(raw_arr)
            if input_was_sparse:
                input_dtype = raw_arr.dtype
                raw_arr = dense_block(raw_arr).astype(input_dtype)

            normalised_arr = _normalise_by_groups(
                raw_arr=raw_arr,
                norm_cls=norm_cls,
                obs_groups=obs_groups,
                var_groups=var_groups,
                block_records=block_records,
            )
            # quantile densifies to compute (its per-sample rank mapping couples all samples); re-sparsify so
            # a sparse input yields a sparse output, recovering the memory freed by dropping absent cells.
            if input_was_sparse:
                normalised_arr = to_observed_sparse(normalised_arr, dtype=input_dtype)
    except UnsharedSamplePairsError as error:
        raise ValueError(_describe_unshared_sample_pairs(error, adata.obs_names.to_list(), method, modality)) from None

    if layer is None:
        adata.X = normalised_arr
    else:
        adata.layers[layer] = normalised_arr

    _record_normalisation(adata, block_records, method=method, layer=layer, modality=modality)

    return mdata


def _describe_unshared_sample_pairs(
    error: UnsharedSamplePairsError, sample_names: list[str], method: str, modality: str, max_listed_pairs: int = 5
) -> str:
    pair_labels = [f"'{sample_names[first]}' x '{sample_names[second]}'" for first, second in error.sample_position_pairs]
    listed = ", ".join(pair_labels[:max_listed_pairs])
    hidden_count = len(pair_labels) - max_listed_pairs
    if hidden_count > 0:
        listed += f" (+{hidden_count} more)"
    return (
        f"Samples share no observed feature: method '{method}' cannot align samples in modality "
        f"'{modality}' because, within one normalisation block, these sample pairs observe no feature "
        f"in common: {listed}. Normalise the sets separately by passing a group_obs column that "
        "separates them, or normalise after summarising to a higher level (e.g. peptide or protein) "
        "where the samples share features."
    )


def _summarise_block(
    row_positions: np.ndarray,
    obs_group: str,
    var_group: str,
    raw_rows: list[np.ndarray],
    normalised_rows: list[np.ndarray],
    block_diagnostics: dict,
) -> dict:
    """Per-sample depth, applied shift (median change; exact for shift methods, a summary for quantile)
    and observed-value median before/after, over the block's features."""
    sample_count = len(raw_rows)
    n_observed_features = np.zeros(sample_count, dtype=np.int64)
    shift_log2 = np.full(sample_count, np.nan)
    location_before = np.full(sample_count, np.nan)
    location_after = np.full(sample_count, np.nan)
    for index, (raw_values, normalised_values) in enumerate(zip(raw_rows, normalised_rows)):
        is_observed = ~np.isnan(raw_values)
        n_observed_features[index] = int(is_observed.sum())
        if not is_observed.any():
            continue
        location_before[index] = np.median(raw_values[is_observed])
        location_after[index] = np.median(normalised_values[is_observed])
        shift_log2[index] = np.median(raw_values[is_observed] - normalised_values[is_observed])
    record = {
        "obs_group": obs_group,
        "var_group": var_group,
        "sample_positions": np.asarray(row_positions),
        "n_observed_features": n_observed_features,
        "shift_log2": shift_log2,
        "location_before": location_before,
        "location_after": location_after,
    }
    if block_diagnostics:
        record["pair_sample_positions"] = np.asarray(row_positions)[block_diagnostics["sample_positions"]]
        record["pair_median_log2"] = block_diagnostics["pair_median_log2"]
        record["pair_shared_count"] = block_diagnostics["pair_shared_count"]
    return record


def _record_normalisation(adata: ad.AnnData, block_records: list[dict], method: str, layer: str | None, modality: str) -> None:
    """Write the block records to ``adata.uns["normalisation"]`` (replacing earlier records of the same
    layer) and log one line per block."""
    layer_name = layer if layer is not None else "X"
    sample_names = np.asarray(adata.obs_names, dtype=object)
    blocks: dict = {}
    summary_frames: list[pd.DataFrame] = []
    for record in block_records:
        block_samples = sample_names[record["sample_positions"]]
        block_key = f"{layer_name}|{record['obs_group']}|{record['var_group']}"
        block = {
            "method": method,
            "layer": layer_name,
            "obs_group": record["obs_group"],
            "var_group": record["var_group"],
            "samples": list(block_samples),
            "n_observed_features": record["n_observed_features"],
            "shift_log2": record["shift_log2"],
            "location_before": record["location_before"],
            "location_after": record["location_after"],
        }
        if "pair_median_log2" in record:
            pair_samples = list(sample_names[record["pair_sample_positions"]])
            block["pair_median_log2"] = pd.DataFrame(record["pair_median_log2"], index=pair_samples, columns=pair_samples)
            block["pair_shared_count"] = pd.DataFrame(record["pair_shared_count"], index=pair_samples, columns=pair_samples)
            _warn_on_weakly_shared_pairs(block["pair_shared_count"], method, modality, block_key)
        blocks[block_key] = block
        summary_frames.append(
            pd.DataFrame(
                {
                    "sample": block_samples,
                    "obs_group": record["obs_group"],
                    "var_group": record["var_group"],
                    "method": method,
                    "layer": layer_name,
                    "n_observed_features": record["n_observed_features"],
                    "shift_log2": record["shift_log2"],
                    "location_before": record["location_before"],
                    "location_after": record["location_after"],
                }
            )
        )
        logger.info(
            "normalise[%s] %s layer %s block %s: %d samples; observed features per sample %d..%d (median %d); "
            "shift %+.3f..%+.3f log2",
            method,
            modality,
            layer_name,
            f"{record['obs_group']}|{record['var_group']}",
            len(block_samples),
            int(record["n_observed_features"].min()),
            int(record["n_observed_features"].max()),
            int(np.median(record["n_observed_features"])),
            float(np.nanmin(record["shift_log2"])),
            float(np.nanmax(record["shift_log2"])),
        )

    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame(
        columns=["sample", "obs_group", "var_group", "method", "layer", "n_observed_features", "shift_log2",
                 "location_before", "location_after"]
    )
    summary["sample"] = summary["sample"].astype(str)

    existing = adata.uns.get("normalisation")
    if isinstance(existing, dict) and isinstance(existing.get("summary"), pd.DataFrame) and isinstance(existing.get("blocks"), dict):
        kept_summary = existing["summary"][existing["summary"]["layer"] != layer_name]
        summary = pd.concat([kept_summary, summary], ignore_index=True)
        kept_blocks = {key: value for key, value in existing["blocks"].items() if not key.startswith(f"{layer_name}|")}
        blocks = {**kept_blocks, **blocks}
    adata.uns["normalisation"] = {"summary": summary, "blocks": blocks}


def _warn_on_weakly_shared_pairs(
    pair_shared_count: pd.DataFrame, method: str, modality: str, block_key: str, minimum_shared_ratio: float = 0.5
) -> None:
    """Warn when the least-sharing pair shares fewer than ``minimum_shared_ratio`` of the median pair's
    features; on real data this marked a failed or near-empty run."""
    counts = pair_shared_count.to_numpy()
    if counts.shape[0] < 2:
        return
    off_diagonal = counts[~np.eye(counts.shape[0], dtype=bool)]
    median_shared_count = float(np.median(off_diagonal))
    if median_shared_count <= 0:
        return
    first, second = np.unravel_index(np.argmin(np.where(np.eye(counts.shape[0], dtype=bool), np.iinfo(np.int64).max, counts)), counts.shape)
    minimum_shared_count = int(counts[first, second])
    if minimum_shared_count < minimum_shared_ratio * median_shared_count:
        logger.warning(
            "normalise[%s] %s block %s: samples '%s' and '%s' share only %d features (median pair shares %d); "
            "their pairwise median rests on few features. Check whether one of these runs failed.",
            method,
            modality,
            block_key,
            pair_shared_count.index[first],
            pair_shared_count.index[second],
            minimum_shared_count,
            int(median_shared_count),
        )


def normalize(
    mdata: md.MuData,
    method: NormalisationMethod,
    modality: str,
    layer: str | None = None,
    group_obs: str | None = None,
    group_var: str | None = None,
    batch_key: str | None = None,
    fraction_key: str | None = None,
    fraction: bool = False,
) -> md.MuData:
    """
    Alias for normalise function to support American English spelling.
    """
    return normalise(
        mdata=mdata,
        method=method,
        modality=modality,
        layer=layer,
        group_obs=group_obs,
        group_var=group_var,
        batch_key=batch_key,
        fraction_key=fraction_key,
        fraction=fraction,
    )


def _partition_indices(groups: np.ndarray | None, length: int) -> list[np.ndarray]:
    """Return positional index arrays — one per unique group, or a single full-range array if groups is None."""
    if groups is None:
        return [np.arange(length)]
    unique_groups = np.unique(groups)
    return [np.where(groups == group)[0] for group in unique_groups]


def _group_label(groups: np.ndarray | None, indices: np.ndarray) -> str:
    return "all" if groups is None else str(groups[indices[0]])


def _normalise_per_group_sparse(matrix, obs_groups, var_groups, norm_cls, block_records: list[dict] | None = None):
    """Per-sample normalisation of a sparse block-diagonal within each (obs_group x var_group) block,
    without densifying.

    Generalises the ungrouped per-sample path: with ``obs_groups`` and ``var_groups`` both None the
    whole matrix is a single block (the common PSM-level TMT/DIA case, taken via a whole-row fast
    path). ``obs_groups`` (obs grouping) partitions rows and ``var_groups`` (var grouping) partitions
    columns; each block is normalised independently, matching the dense ``_normalise_by_groups`` path.
    ``norm_cls`` supplies the per-block rescaler (``rescale_sparse_block``). Only ``.data`` is rewritten
    so structurally-absent cells stay absent and the layer stays sparse; the stored dtype is preserved.
    ``pairwise_median`` densifies the shared features of a block after checking the stored pattern.
    Each block's per-sample record is appended to ``block_records`` when given.
    """
    csr = matrix.tocsr(copy=True)
    row_partitions = _partition_indices(obs_groups, csr.shape[0])
    # A single ``None`` column-partition keeps the whole-row fast path (no per-row column split) when
    # no var grouping is requested -- covers the hot ungrouped case and obs-only grouping.
    col_partitions = _partition_indices(var_groups, csr.shape[1]) if var_groups is not None else [None]
    for row_indices in row_partitions:
        for col_indices in col_partitions:
            record = _normalise_sparse_block(csr, row_indices, col_indices, norm_cls)
            if block_records is not None and record is not None:
                record["obs_group"] = _group_label(obs_groups, row_indices)
                record["var_group"] = "all" if col_indices is None else _group_label(var_groups, col_indices)
                block_records.append(record)
    return csr


def _normalise_sparse_block(csr, row_indices, col_indices, norm_cls) -> dict | None:
    """Collect the stored-cell indices of one (row_indices x col_indices) block, then hand them to
    ``norm_cls.rescale_sparse_block`` to rewrite in place. ``col_indices=None`` means all columns
    (whole-row slices -- the hot path, no per-row column split). Rows with no stored cell in the block
    are skipped and excluded from the block scalar, matching the dense path's all-NaN-row filter. Only
    stored cells are touched, so absent cells stay absent and the stored dtype is preserved. Returns
    the block's per-sample record, or None for an empty block.
    """
    indptr, indices = csr.indptr, csr.indices
    column_mask = None
    if col_indices is not None:
        column_mask = np.zeros(csr.shape[1], dtype=bool)
        column_mask[col_indices] = True

    # Per-row index into ``csr.data`` for this block's cells: a contiguous slice for the whole-row path,
    # or an explicit position array when a var-group column mask restricts the row.
    block_cell_indices: list = []
    block_row_indices: list[int] = []
    for row in row_indices:
        start, end = indptr[row], indptr[row + 1]
        if end <= start:
            continue
        if column_mask is None:
            block_cell_indices.append(slice(start, end))
            block_row_indices.append(row)
        else:
            selected = np.nonzero(column_mask[indices[start:end]])[0]
            if selected.size:
                block_cell_indices.append(start + selected)
                block_row_indices.append(row)
    if not block_cell_indices:
        return None
    raw_rows = [csr.data[idx].copy() for idx in block_cell_indices]
    try:
        norm_cls.rescale_sparse_block(csr, block_cell_indices)
    except UnsharedSamplePairsError as error:
        raise error.map_sample_positions(np.asarray(block_row_indices)) from None
    normalised_rows = [csr.data[idx] for idx in block_cell_indices]
    return _summarise_block(
        row_positions=np.asarray(block_row_indices),
        obs_group="all",
        var_group="all",
        raw_rows=raw_rows,
        normalised_rows=normalised_rows,
        block_diagnostics=norm_cls.block_diagnostics,
    )


def _normalise_by_groups(
    raw_arr: np.ndarray,
    norm_cls: Normalisation,
    obs_groups: np.ndarray | None,
    var_groups: np.ndarray | None,
    block_records: list[dict] | None = None,
) -> np.ndarray:
    """Normalise raw_arr within each (obs_group × var_group) block; un-grouped axes use a single block.
    Each block's per-sample record is appended to ``block_records`` when given."""
    obs_partitions = _partition_indices(obs_groups, raw_arr.shape[0])
    var_partitions = _partition_indices(var_groups, raw_arr.shape[1])

    normalised_arr = np.full_like(raw_arr, np.nan, dtype=float)

    for obs_idx in obs_partitions:
        for var_idx in var_partitions:
            sub_block = raw_arr[np.ix_(obs_idx, var_idx)]
            not_all_nan_rows = ~np.all(np.isnan(sub_block), axis=1)
            valid_rows = np.where(not_all_nan_rows)[0]
            if valid_rows.size == 0:
                continue
            try:
                block_normalised = norm_cls.normalise(arr=sub_block[valid_rows, :])
            except UnsharedSamplePairsError as error:
                raise error.map_sample_positions(obs_idx[valid_rows]) from None
            for local_row_idx, original_local_row in enumerate(valid_rows):
                target_row = obs_idx[original_local_row]
                normalised_arr[target_row, var_idx] = block_normalised[local_row_idx]
            if block_records is not None:
                block_records.append(
                    _summarise_block(
                        row_positions=obs_idx[valid_rows],
                        obs_group=_group_label(obs_groups, obs_idx),
                        var_group=_group_label(var_groups, var_idx),
                        raw_rows=list(sub_block[valid_rows, :]),
                        normalised_rows=list(block_normalised),
                        block_diagnostics=norm_cls.block_diagnostics,
                    )
                )

    return normalised_arr


def _read_global_mdata(global_mdata: md.MuData | str | PathLike[str]) -> md.MuData:
    """Accept the matched global dataset as an object, or as the path of an ``.h5mu`` holding it.

    A path keeps the PTM container's history one chain: the file is recorded as an input with its
    content hash instead of merging the global dataset's own history into the result. That is what
    lets ``mm.pv.replay`` and ``mm.pv.to_script`` reproduce a PTM workflow through the adjustment,
    which they cannot do for a history with two parents.
    """
    if isinstance(global_mdata, md.MuData):
        return global_mdata

    # Lazy import: _reader_registry imports this package, so importing read_h5mu at module top
    # creates a circular import that fails depending on which subpackage loads first.
    from .._read_write._reader_registry import read_h5mu

    return read_h5mu(global_mdata)


@log_provenance
def adjust_ptm_by_protein(
    mdata: md.MuData,
    global_mdata: md.MuData | str | PathLike[str],
    modality: str = "phospho_site",
    layer: str | None = None,
    method: PTMAdjustmentMethod = "ratio",
    rescale: bool = True,
    ridge_alpha: float | None = None,
) -> md.MuData:
    """
    Adjust PTM site intensities by parent protein abundance from a matched global dataset.

    This computes *differential PTM usage* (DPU): the site's log intensity minus the log intensity
    of its parent protein in the same sample, so that a site's change is read relative to whatever
    its protein did. It is not occupancy/stoichiometry -- that additionally requires the unmodified
    counterpart peptide, and is a different estimand with far lower coverage.

    A site's denominator is found through the accessions it was localised on, translated into a
    global protein group via the global dataset's ``uns['protein_map']``. Nothing is looked up by
    peptide, so a PTM peptide the global run never observed is still adjustable whenever its protein
    was quantified there. Sites whose accessions span two or more quantified global groups have no
    valid denominator and are left unadjusted; ``var['adjustment_status']`` records why for every
    site.

    The adjusted values replace the quantification that was read -- ``.X``, or ``layers[layer]`` when
    given -- the same contract as ``log2_transform``, ``normalise`` and ``correct_batch_effect``. A
    site that could not be adjusted is set to NaN rather than left holding its raw abundance, so
    residuals and raw abundances never share a matrix. To keep the unadjusted values, copy them into
    a layer first::

        mdata[modality].layers["unadjusted"] = mdata[modality].X.copy()

    Parameters:
        mdata: MuData object holding the PTM data.
        global_mdata: The matched global dataset: a MuData holding global protein expression in its
            'protein' modality and the protein mapping in uns['protein_map'], or the path of an
            ``.h5mu`` file holding one. A path keeps this container's provenance a single chain --
            the file is recorded as an input with its content hash instead of merging the global
            dataset's own history -- so ``mm.pv.replay`` and ``mm.pv.to_script`` can reproduce the
            workflow through this step. Pass a ``Path`` rather than a ``str`` for that content hash.
        modality: PTM modality to adjust (e.g. phospho_site, {ptm}_site).
        layer: Layer to adjust. If None, the default layer (.X) will be used.
        method: Estimator to use. 'ratio' subtracts the protein level, assuming the slope-one
            relationship mass action predicts; 'ridge' instead fits a slope per site. Default is
            'ratio', which needs no fitting and so stays usable at proteomics sample counts.
        rescale: If True, shift the adjusted values by the PTM data's overall median so they read on
            a comparable scale. A single constant, so it cancels in any contrast. Default is True.
        ridge_alpha: Ridge penalty, used only when method='ridge'. A single-predictor ridge keeps
            ``Sxx / (Sxx + alpha)`` of the least-squares slope, so a large alpha silently turns the
            residual into a plain mean-centring. Defaults to DEFAULT_RIDGE_ALPHA.

    Returns:
        MuData object with the adjusted quantification and per-site adjustment annotations.
    """
    mdata = mdata.copy()

    ptm_adjuster: PTMProteinAdjuster = PTMProteinAdjuster(
        ptm_mdata=mdata,
        global_mdata=_read_global_mdata(global_mdata),
        ptm_mod=modality,
        global_mod="protein",
        layer=layer,
    )
    adj_ptm_mdata: md.MuData = ptm_adjuster.adjust(
        method=method,
        rescale=rescale,
        alpha=ridge_alpha,
    )

    return adj_ptm_mdata

    # class FractionNormalisation(Normalisation):
    #    def __init__(self, method: str) -> None:
    #        super().__init__(method=method)
    #
    #    def reshape(self, arr):
    #        # Implement the reshape method specific to FractionNormalisation
    #        pass
    #
    #    def inverse_shape(self, normalised_arr) -> np.ndarray:
    #        return super().inverse_shape(normalised_arr=normalised_arr)
    #
    #    def normalise_intra_fraction(self, arr, fraction_arr):
    #        original_arr = arr.copy()
    #        normalised_arr = np.full_like(original_arr, np.nan, dtype=float).T
    #
    #        for fraction in np.unique(fraction_arr):
    #            fraction_idx = np.where(fraction_arr == fraction)[0]
    #            fraction_data = original_arr[:, fraction_idx].T
    #
    #            fraction_data = self._method_call(fraction_data).T
    #
    #            normalised_arr[fraction_idx] = fraction_data.T
    #
    #        return normalised_arr.T
    #
    #    def normalise_inter_fraction(self, arr, fraction_arr):
    #        # Normalize across fractions
    #        flattened_channel = [
    #            arr[:, np.where(fraction_arr == fraction)[0]].flatten()
    #            for fraction in np.unique(fraction_arr)
    #        ]
    #        flatten_array = np.array(pd.DataFrame(flattened_channel)).T
    #        normed_flattened_channel = self._method_call(flatten_array)
    #
    #        return self.reconstruct_data(
    #            arr=arr,
    #            fraction_arr=fraction_arr,
    #            normed_flattened_channel=normed_flattened_channel,
    #        )
    #
    #    def reconstruct_data(self, arr, fraction_arr, normed_flattened_channel):
    #        normalised_arr = np.full_like(arr, np.nan, dtype=float)
    #
    #        for index, fraction in enumerate(sorted(set(fraction_arr))):
    #            fraction_index = fraction_arr == fraction
    #            original_shape = arr[:, fraction_index].shape
    #            original_length = original_shape[0] * original_shape[1]
    #
    #            normed_flattened_fraction_data = normed_flattened_channel.T[index][
    #                :original_length
    #            ]
    #            reconstructed_fraction_data = np.reshape(
    #                normed_flattened_fraction_data, original_shape
    #            )
    #
    #            normalised_arr[:, fraction_index] = reconstructed_fraction_data
    #
    #        return normalised_arr.T
    #
    #    def normalise(self, arr, var):
    #        self._fraction_arr = var["filename"].values
    #        intra_normalised_arr = self.normalise_intra_fraction(
    #            arr=arr, fraction_arr=self._fraction_arr
    #        )
    #        inter_normalised_arr = self.normalise_inter_fraction(
    #            arr=intra_normalised_arr, fraction_arr=self._fraction_arr
    #        )
    #        fraction_normalised_arr = self._method_call(inter_normalised_arr)
    #        fraction_normalised_arr = super().inverse_shape(fraction_normalised_arr)
    #


#        return fraction_normalised_arr
