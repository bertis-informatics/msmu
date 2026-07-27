import warnings

import anndata as ad
import mudata as md
import numpy as np
from typing import Literal

from .._utils._mudata import get_anndata_mod
from .._core._provenance import uns_logger
from .._core._blockdiag import dense_block, is_sparse, sparse_apply_elementwise
from ..logging_utils import get_logger
from ._normalisation import Normalisation, PTMProteinAdjuster

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
    if is_sparse(raw_arr):
        raw_arr = dense_block(raw_arr).astype(raw_arr.dtype)

    mean_arr: np.ndarray = np.nanmean(raw_arr, axis=0)
    std_arr: np.ndarray = np.nanstd(raw_arr, axis=0)
    scaled_arr: np.ndarray = (raw_arr - mean_arr) / std_arr

    if layer is None:
        adata.X = scaled_arr
    else:
        adata.layers[layer] = scaled_arr

    return mdata


@uns_logger
def normalise(
    mdata: md.MuData,
    method: str,
    modality: str,
    layer: str | None = None,
    batch_key: str | None = None,
    fraction_key: str | None = None,
    fraction: bool = False,
) -> md.MuData:
    """
    Normalise data in MuData object.

    Parameters:
        mdata: MuData object to normalise.
        method: Normalisation method to use. Options are 'quantile', 'median', 'total_sum (not implemented)'.
        modality: Modality to normalise.
        layer: Layer to normalise. If None, the default layer (.X) will be used.
        batch_key: Column name in ``adata.obs`` defining batches. If provided, normalisation
            is performed independently within each batch. If None, no batch grouping is applied.
        fraction_key: Column name in ``adata.var`` defining fractions (e.g. ``"filename"`` for
            fractionated TMT or fractionated label-free workflows). If provided, normalisation
            is performed independently within each fraction. If None, no fraction grouping
            is applied.
        fraction: Deprecated. If True, equivalent to ``fraction_key="filename"``. Use
            ``fraction_key`` instead.

    Returns:
        Normalised MuData object.

    Notes:
        When both ``batch_key`` and ``fraction_key`` are provided, normalisation is performed
        independently within each (batch × fraction) block.
    """
    if fraction:
        warnings.warn(
            "`fraction=True` is deprecated; pass `fraction_key='filename'` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if fraction_key is None:
            fraction_key = "filename"

    axis: str = "obs"

    mdata = mdata.copy()
    adata: ad.AnnData = get_anndata_mod(mdata, modality)
    norm_cls: Normalisation = Normalisation(method=method, axis=axis)

    if layer is None:
        raw_arr: np.ndarray = adata.X
    else:
        raw_arr: np.ndarray = adata.layers[layer]

    if batch_key is not None and batch_key not in adata.obs.columns:
        raise KeyError(f"batch_key '{batch_key}' not found in adata.obs of modality '{modality}'.")
    if fraction_key is not None and fraction_key not in adata.var.columns:
        raise KeyError(f"fraction_key '{fraction_key}' not found in adata.var of modality '{modality}'.")

    obs_groups = adata.obs[batch_key].to_numpy() if batch_key is not None else None
    var_groups = adata.var[fraction_key].to_numpy() if fraction_key is not None else None

    # Normalisation (median/quantile) needs per-sample distributions across features, so it
    # densifies a sparse block-diagonal (NaN for absent, same dtype as the dense path). Memory-
    # efficient sparse normalisation is a follow-up; correctness here matches the dense path.
    if is_sparse(raw_arr):
        raw_arr = dense_block(raw_arr).astype(raw_arr.dtype)

    normalised_arr = _normalise_by_groups(
        raw_arr=raw_arr,
        norm_cls=norm_cls,
        obs_groups=obs_groups,
        var_groups=var_groups,
    )

    if layer is None:
        adata.X = normalised_arr
    else:
        adata.layers[layer] = normalised_arr

    return mdata


def normalize(
    mdata: md.MuData,
    method: str,
    modality: str,
    layer: str | None = None,
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
        global_mdata: MuData object which contains global protein expression.
        modality: PTM modality to normalise (e.g. phospho_site, {ptm}_site).
        layer: Layer to normalise. If None, the default layer (.X) will be used.
        global_mod: Modality in global_mdata to normalise PTM site. Default is 'protein'.
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
