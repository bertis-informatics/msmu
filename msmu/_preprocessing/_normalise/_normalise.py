from typing import Callable

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from .normalisation_methods import (
    normalise_median_center,
    normalise_quantile,
    normalise_total_sum,
)
from ..._utils import uns_logger


@uns_logger
def log2_transform(
    mdata: md.MuData,
    modality: str | None,
):
    adata = mdata[modality].copy()

    log2_arr = np.log2(adata.X)

    mdata[modality].X = log2_arr

    return mdata


def normalise(
    mdata: md.MuData,
    method: str,
    modality: str,
    fraction: bool = False,
    rescale: bool = True,
):
    """
    Normalise data in MuData object.
    Parameters
    ----------
    mdata: MuData
        MuData object to normalise.
    method: str
        Normalisation method to use. Options are 'quantile', 'median', 'total_sum (not implemented)'.
    modality: str
        Modality to normalise. If None, all modalities at the specified level will be normalised.
    fraction: bool
        If True, normalise within fractions. If False, normalise across all data.
    rescale: bool
        If True, rescale the data after normalisation with median value across dataset. This is only applicable for median normalisation.

    Returns
    -------
    mdata: MuData
        Normalised MuData object.
    """
    axis: str = "obs"

    adata: ad.AnnData = mdata.mod[modality].copy()
    norm_cls: Normalisation = Normalisation(method=method, axis=axis)

    rescale_arr: np.array[float] = np.array([])
    rescale_arr = np.append(rescale_arr, adata.X.flatten())
    if fraction:
        normalised_arr = np.full_like(adata.X, np.nan, dtype=float)
        for frac in np.unique(adata.var["filename"]):
            fraction_idx = adata.var["filename"] == frac

            arr = adata.X[:, fraction_idx].copy()  

            not_all_nan_rows = ~np.all(np.isnan(arr), axis=1)
            indices = np.where(not_all_nan_rows)[0]
            
            arr = arr[indices, :].copy()
            fraction_normalised_data = norm_cls.normalise(arr=arr)
            
            for i, r in enumerate(indices):
                normalised_arr[r, fraction_idx] = fraction_normalised_data[i]
            # normalised_arr[indices, fraction_idx] = fraction_normalised_data

    else:
        arr = adata.X.copy()
        normalised_arr = norm_cls.normalise(arr=arr)

    mdata.mod[modality].X = normalised_arr

    # rescale function for median normalisation
    if (method == "median") & rescale:
        all_median = np.nanmedian(rescale_arr.flatten())
        mdata.mod[modality].X = mdata[modality].X + all_median

    return mdata


def feature_scale(
    mdata: md.MuData,
    method: str,
    modality: str,
    gis_prefix: str | None = None,
    gis_col: list[str] | None = None,
    rescale: bool = True,
) -> md.MuData:
    """
    Feature scale data in MuData object.
    Parameters
    ----------
    mdata: MuData
        MuData object to normalise.
    method: str
        Normalisation method to use. Options are 'gis', 'median_center'.
    modality: str
        Modality to normalise.
    gis_prefix: str
        Prefix for GIS samples. If None, all samples with 'gis' in the name will be used.
    gis_col: str
        Column name for GIS samples. If None, all samples with 'gis' in the name will be used.
    rescale: bool
        If True, rescale the data after normalisation with median value across dataset. This is only applicable for median normalisation.
    Returns
    -------
    mdata: MuData
        Normalised MuData object.
    """
    adata: ad.AnnData = mdata.mod[modality].copy()
    median_rescale_arr: np.array[float] = np.array([])
    if method == "gis":
        if (gis_prefix is None) & (gis_col is None):
            raise ValueError("Please provide either a GIS prefix or GIS column name")

        if gis_col is not None:
            gis_idx: np.array[bool] = adata.obs[gis_col] == True
        else:
            gis_idx: np.array[bool] = adata.obs_names.str.startswith(gis_prefix) == True

        if gis_idx.sum() == 0:
            raise ValueError(f"No GIS samples found in {modality}")

        gis_normalised_data: np.array[float] = normalise_gis(
            arr=adata.X, gis_idx=gis_idx
        )

        gis_drop_mod = adata[~gis_idx]
        gis_drop_mod.X = gis_normalised_data
        mdata.mod[modality] = gis_drop_mod

        median_rescale_arr = np.append(median_rescale_arr, adata[gis_idx].X.flatten())

    elif method == "median_center":
        median_centered_data = Normalisation(method="median", axis="var").normalise(
            arr=adata.X,
        )
        mdata[modality].X = median_centered_data

        median_rescale_arr = np.append(median_rescale_arr, adata.X.flatten())

    else:
        raise ValueError(
            f"Method {method} not recognised. Please choose from 'gis' or 'median_center'"
        )

    if rescale:
        all_gis_median = np.nanmedian(median_rescale_arr.flatten())
        mdata[modality].X = mdata[modality].X + all_gis_median

    mdata.update_obs()

    return mdata


def normalise_gis(arr: np.ndarray, gis_idx: np.array) -> np.ndarray:
    gis_data = arr[gis_idx]
    sample_data = arr[~gis_idx]
    na_idx = np.isnan(sample_data)

    gis_median = np.nanmedian(gis_data, axis=0)
    gis_normalised_data = sample_data - gis_median
    gis_normalised_data[na_idx] = np.nan

    return gis_normalised_data


def get_modality_dict(
    mdata: md.MuData, level: str | None = None, modality: str | None = None
) -> dict:
    """Get modality data from MuData object"""

    if (level == None) & (modality == None):
        level = "psm"

    mod_dict: dict = dict()
    if level != None:
        for mod_name in mdata.mod_names:
            if mdata[mod_name].uns["level"] == level:
                mod_dict[mod_name] = mdata[mod_name].copy()

    elif modality != None:
        mod_dict[modality] = mdata[modality].copy()

    return mod_dict


class Normalisation:
    def __init__(self, method: str, axis: str) -> None:
        self._method_call: Callable = getattr(self, f"_{method}")
        self._axis = axis

    def _quantile(self, arr) -> np.ndarray:
        return normalise_quantile(arr=arr)

    def _median(self, arr) -> np.ndarray:
        return normalise_median_center(arr=arr)

    def _total_sum(self, arr) -> np.ndarray:
        return normalise_total_sum()

    def normalise(self, arr) -> np.ndarray:
        na_idx = np.isnan(arr)
        if self._axis == "obs":
            transposed_arr = arr.T
            normalised_arr = self._method_call(arr=transposed_arr)
            normalised_arr = normalised_arr.T

        elif self._axis == "var":
            normalised_arr = self._method_call(arr=arr)

        else:
            raise ValueError(f"Axis {self._axis} not recognised. Please choose from 'obs' or 'var'")

        normalised_arr[na_idx] = np.nan

        return normalised_arr

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
