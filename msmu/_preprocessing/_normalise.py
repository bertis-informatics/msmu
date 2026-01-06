import anndata as ad
import mudata as md
import numpy as np
from typing import Literal

from ._normalisation import Normalisation, PTMProteinAdjuster
from .._utils import uns_logger


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

    if layer is None:
        raw_arr = mdata.mod[modality].X
    else:
        raw_arr = mdata.mod[modality].layers[layer]

    log2_arr = np.log2(raw_arr)

    if layer is None:
        mdata.mod[modality].X = log2_arr
    else:
        mdata.mod[modality].layers[layer] = log2_arr

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

    if layer is None:
        raw_arr: np.ndarray = mdata.mod[modality].X
    else:
        raw_arr: np.ndarray = mdata.mod[modality].layers[layer]

    mean_arr: np.ndarray = np.nanmean(raw_arr, axis=0)
    std_arr: np.ndarray = np.nanstd(raw_arr, axis=0)
    scaled_arr: np.ndarray = (raw_arr - mean_arr) / std_arr

    if layer is None:
        mdata.mod[modality].X = scaled_arr
    else:
        mdata.mod[modality].layers[layer] = scaled_arr

    return mdata


@uns_logger
def normalise(
    mdata: md.MuData,
    method: str,
    modality: str,
    layer: str | None = None,
    fraction: bool = False,
) -> md.MuData:
    """
    Normalise data in MuData object.

    Parameters:
        mdata: MuData object to normalise.
        method: Normalisation method to use. Options are 'quantile', 'median', 'total_sum (not implemented)'.
        modality: Modality to normalise. If None, all modalities at the specified level will be normalised.
        layer: Layer to normalise. If None, the default layer (.X) will be used.
        fraction: If True, normalise within fractions. If False, normalise across all data. "fraction" yet supports fractionated TMT.

    Returns:
        Normalised MuData object.
    """
    axis: str = "obs"

    mdata = mdata.copy()
    adata: ad.AnnData = mdata.mod[modality]
    norm_cls: Normalisation = Normalisation(method=method, axis=axis)

    if layer is None:
        raw_arr: np.ndarray = adata.X
    else:
        raw_arr: np.ndarray = adata.layers[layer]

    rescale_arr: np.array[float] = np.array([])
    rescale_arr = np.append(rescale_arr, raw_arr.flatten())

    # TODO: refactor and package intra-fraction normalisation
    if fraction:
        normalised_arr = np.full_like(raw_arr, np.nan, dtype=float)
        for frac in np.unique(adata.var["filename"]):
            fraction_idx = adata.var["filename"] == frac

            arr = raw_arr[:, fraction_idx]
            not_all_nan_rows = ~np.all(np.isnan(arr), axis=1)
            indices = np.where(not_all_nan_rows)[0]

            arr = arr[indices, :]
            fraction_normalised_data = norm_cls.normalise(arr=arr)

            for i, r in enumerate(indices):
                normalised_arr[r, fraction_idx] = fraction_normalised_data[i]
            # normalised_arr[indices, fraction_idx] = fraction_normalised_data

    else:
        arr = raw_arr
        normalised_arr = norm_cls.normalise(arr=arr)

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
        fraction=fraction,
    )


@uns_logger
def correct_batch_effect(
    mdata: md.MuData,
    modality: str,
    method: Literal["gis", "median_center"],
    layer: str | None = None,
    gis_prefix: str | None = None,
    gis_col: list[str] | None = None,
    rescale: bool = True,
) -> md.MuData:
    """
    correct_batch_effect in MuData object.

    Parameters:
        mdata: MuData object to normalise.
        method: Normalisation method to use. Options are 'gis', 'median_center'.
        modality: Modality to normalise.
        layer: Layer to normalise. If None, the default layer (.X) will be used.
        gis_prefix: Prefix for GIS samples. If None, all samples with 'gis' in the name will be used.
        gis_col: Column name for GIS samples. If None, all samples with 'gis' in the name will be used.
        rescale: If True, rescale the data after normalisation with median value across dataset. This is only applicable for median normalisation.

    Returns:
        Normalised MuData object.
    """
    mdata = mdata.copy()
    adata: ad.AnnData = mdata.mod[modality]
    median_rescale_arr: np.array[float] = np.array([])

    if layer is not None:
        raw_arr = adata.layers[layer]
    else:
        raw_arr = adata.X

    if method == "gis":
        if (gis_prefix is None) & (gis_col is None):
            raise ValueError("Please provide either a GIS prefix or GIS column name")

        if gis_col is not None:
            gis_idx: np.array[bool] = adata.obs[gis_col] == True
        else:
            gis_idx: np.array[bool] = adata.obs_names.str.startswith(gis_prefix) == True

        if gis_idx.sum() == 0:
            raise ValueError(f"No GIS samples found in {modality}")

        gis_normalised_data: np.array[float] = _normalise_gis(arr=raw_arr, gis_idx=gis_idx)

        gis_drop_mod = adata[~gis_idx].copy()
        gis_drop_mod.X = gis_normalised_data
        mdata.mod[modality] = gis_drop_mod

        median_rescale_arr = np.append(median_rescale_arr, adata[gis_idx].X.flatten())

    elif method == "median_center":
        median_centered_data = Normalisation(method="median", axis="var").normalise(
            arr=raw_arr,
        )
        mdata[modality].X = median_centered_data

        median_rescale_arr = np.append(median_rescale_arr, raw_arr.flatten())
    else:
        raise ValueError(f"Method {method} not recognised. Please choose from 'gis' or 'median_center'")

    if rescale:
        all_gis_median = np.nanmedian(median_rescale_arr.flatten())
        mdata[modality].X = mdata[modality].X + all_gis_median

    mdata.update_obs()

    return mdata


def _normalise_gis(arr: np.ndarray, gis_idx: np.array) -> np.ndarray:
    gis_data = arr[gis_idx]
    sample_data = arr[~gis_idx]
    na_idx = np.isnan(sample_data)

    gis_median = np.nanmedian(gis_data, axis=0)
    gis_normalised_data = sample_data - gis_median
    gis_normalised_data[na_idx] = np.nan

    return gis_normalised_data


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

    if layer is not None:
        mdata.mod[modality].X = mdata.mod[modality].layers[layer]

    ptm_adjuster: PTMProteinAdjuster = PTMProteinAdjuster(
        ptm_mdata=mdata, global_mdata=global_mdata, ptm_mod=modality, global_mod="protein"
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
