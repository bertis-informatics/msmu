import warnings

import anndata as ad
import mudata as md
import numpy as np
from typing import Literal

from .._utils._mudata import get_anndata_mod
from .._core._provenance import uns_logger
from .._core._blockdiag import dense_block, is_sparse, sparse_apply_elementwise, to_observed_sparse
from ..logging_utils import get_logger
from ._normalisation import Normalisation, NormalisationMethod, PTMProteinAdjuster

logger = get_logger(__name__)


@uns_logger
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


@uns_logger
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


@uns_logger
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
        method: Normalisation method to use. Options are 'quantile', 'median', 'total_sum'.
        modality: Modality to normalise.
        layer: Layer to normalise. If None, the default layer (.X) will be used.
        group_obs: Column name in ``adata.obs`` defining sample groups. If provided, normalisation is
            performed independently within each group. If None, no obs grouping is applied.
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
    """
    if batch_key is not None:
        warnings.warn("`batch_key` is deprecated; use `group_obs` instead.", DeprecationWarning, stacklevel=2)
        if group_obs is None:
            group_obs = batch_key
    if fraction_key is not None:
        warnings.warn("`fraction_key` is deprecated; use `group_var` instead.", DeprecationWarning, stacklevel=2)
        if group_var is None:
            group_var = fraction_key
    if fraction:
        warnings.warn(
            "`fraction=True` is deprecated; use `group_var='filename'` instead.",
            DeprecationWarning,
            stacklevel=2,
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

    obs_groups = adata.obs[group_obs].to_numpy() if group_obs is not None else None
    var_groups = adata.var[group_var].to_numpy() if group_var is not None else None

    # Sparse-native methods (those the ``Normalisation`` object defines a ``_{method}_sparse`` rescaler
    # for) are computed directly on the sparse block-diagonal -- each obs row is rescaled from its own
    # stored values -- so the layer stays sparse instead of materialising the dense matrix, for the
    # common ungrouped case as well as grouped normalisation (each block normalised independently).
    # Quantile is not sparse-native (its per-sample rank mapping couples all samples), so it densifies.
    if is_sparse(raw_arr) and norm_cls.is_sparse_native:
        normalised_arr = _normalise_per_group_sparse(raw_arr, obs_groups, var_groups, norm_cls)
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
        )
        # quantile densifies to compute (its per-sample rank mapping couples all samples); re-sparsify so
        # a sparse input yields a sparse output, recovering the memory freed by dropping absent cells.
        if input_was_sparse:
            normalised_arr = to_observed_sparse(normalised_arr, dtype=input_dtype)

    if layer is None:
        adata.X = normalised_arr
    else:
        adata.layers[layer] = normalised_arr

    return mdata


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


def _normalise_per_group_sparse(matrix, obs_groups, var_groups, norm_cls):
    """Per-sample normalisation of a sparse block-diagonal within each (obs_group x var_group) block,
    without densifying.

    Generalises the ungrouped per-sample path: with ``obs_groups`` and ``var_groups`` both None the
    whole matrix is a single block (the common PSM-level TMT/DIA case, taken via a whole-row fast
    path). ``obs_groups`` (obs grouping) partitions rows and ``var_groups`` (var grouping) partitions
    columns; each block is normalised independently, matching the dense ``_normalise_by_groups`` path.
    ``norm_cls`` supplies the per-block rescaler (``rescale_sparse_block``). Only ``.data`` is rewritten
    so structurally-absent cells stay absent and the layer stays sparse; the stored dtype is preserved.
    """
    csr = matrix.tocsr(copy=True)
    row_partitions = _partition_indices(obs_groups, csr.shape[0])
    # A single ``None`` column-partition keeps the whole-row fast path (no per-row column split) when
    # no var grouping is requested -- covers the hot ungrouped case and obs-only grouping.
    col_partitions = _partition_indices(var_groups, csr.shape[1]) if var_groups is not None else [None]
    for row_indices in row_partitions:
        for col_indices in col_partitions:
            _normalise_sparse_block(csr, row_indices, col_indices, norm_cls)
    return csr


def _normalise_sparse_block(csr, row_indices, col_indices, norm_cls):
    """Collect the stored-cell indices of one (row_indices x col_indices) block, then hand them to
    ``norm_cls.rescale_sparse_block`` to rewrite in place. ``col_indices=None`` means all columns
    (whole-row slices -- the hot path, no per-row column split). Rows with no stored cell in the block
    are skipped and excluded from the block scalar, matching the dense path's all-NaN-row filter. Only
    stored cells are touched, so absent cells stay absent and the stored dtype is preserved.
    """
    indptr, indices = csr.indptr, csr.indices
    column_mask = None
    if col_indices is not None:
        column_mask = np.zeros(csr.shape[1], dtype=bool)
        column_mask[col_indices] = True

    # Per-row index into ``csr.data`` for this block's cells: a contiguous slice for the whole-row path,
    # or an explicit position array when a var-group column mask restricts the row.
    block_cell_indices: list = []
    for row in row_indices:
        start, end = indptr[row], indptr[row + 1]
        if end <= start:
            continue
        if column_mask is None:
            block_cell_indices.append(slice(start, end))
        else:
            selected = np.nonzero(column_mask[indices[start:end]])[0]
            if selected.size:
                block_cell_indices.append(start + selected)
    if block_cell_indices:
        norm_cls.rescale_sparse_block(csr, block_cell_indices)


def _normalise_by_groups(
    raw_arr: np.ndarray,
    norm_cls: Normalisation,
    obs_groups: np.ndarray | None,
    var_groups: np.ndarray | None,
) -> np.ndarray:
    """Normalise raw_arr within each (obs_group × var_group) block; un-grouped axes use a single block."""
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
            block_normalised = norm_cls.normalise(arr=sub_block[valid_rows, :])
            for local_row_idx, original_local_row in enumerate(valid_rows):
                target_row = obs_idx[original_local_row]
                normalised_arr[target_row, var_idx] = block_normalised[local_row_idx]

    return normalised_arr


@uns_logger
def adjust_ptm_by_protein(
    mdata: md.MuData,
    global_mdata: md.MuData,
    modality: str = "phospho_site",
    layer: str | None = None,
    method: Literal["ridge", "ratio"] = "ridge",
    rescale: bool = True,
) -> md.MuData:
    """
    Estimation of PTM stoichiometry by using Global Protein Data.

    Parameters:
        mdata: MuData object to normalise.
        global_mdata: MuData object which contains global protein expression, read from its
            'protein' modality.
        modality: PTM modality to normalise (e.g. phospho_site, {ptm}_site).
        layer: Layer to normalise. If None, the default layer (.X) will be used.
        method: A method for normalisation. Options: ridge, ratio. Default is 'ridge'.
        rescale: If True, rescale the data after normalisation with median value across dataset. Default is True.

    Returns:
        Normalised MuData object.
    """
    mdata = mdata.copy()
    adata = get_anndata_mod(mdata, modality)

    if layer is not None:
        adata.X = adata.layers[layer]

    ptm_adjuster: PTMProteinAdjuster = PTMProteinAdjuster(
        ptm_mdata=mdata,
        global_mdata=global_mdata,
        ptm_mod=modality,
        global_mod="protein",
    )
    adj_ptm_mdata: md.MuData = ptm_adjuster.adjust(method=method, rescale=rescale)

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
