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


def log2_transform(
    mdata: md.MuData, modality: str | None = None, level: str | None = None
):
    mod_dict = get_modality_dict(mdata=mdata, level=level, modality=modality)
    for mod_name, mod in mod_dict.items():
        log2_arr = np.log2(mod.X)

        mdata[mod_name].X = log2_arr

    return mdata


def normalise(
    mdata: md.MuData,
    method: str,
    modality: str,
    level: str,
    axis: str = "obs",
    fraction: bool = False,
    ignore_blank: bool = True,
    ignore_gis: bool = True,
):

    mod_dict = get_modality_dict(mdata=mdata, level=level, modality=modality)
    norm_cls = Normalisation(method=method, axis=axis)

    for mod_name, mod in mod_dict.items():
        if fraction:
            for fraction in np.unique(mod.var["filename"]):
                fraction_idx = mod.var["filename"] == fraction
                trimmed_arr, valid_idx = _trim_blank_gis(
                    adata=mod[:, fraction_idx],
                    ignore_blank=ignore_blank,
                    ignore_gis=ignore_gis,
                )

                normalised_data = norm_cls.normalise(arr=trimmed_arr)
                mdata.mod[mod_name].X[np.ix_(valid_idx, fraction_idx)] = normalised_data

        else:
            trimmed_arr, valid_idx = _trim_blank_gis(
                adata=mod, ignore_blank=ignore_blank, ignore_gis=ignore_gis
            )
            normalised_data = norm_cls.normalise(arr=trimmed_arr)

            mdata[mod_name].X[valid_idx] = normalised_data

    return mdata


def correct_batch_effect(
    mdata: md.MuData, batch: str, method: str, modality: str, level: str
) -> md.MuData:
    pass


def scale_data(
    mdata: md.MuData, method: str, modality: str | None = None, level: str | None = None
) -> md.MuData:
    mod_dict = get_modality_dict(mdata=mdata, level=level, modality=modality)
    for mod_name, mod in mod_dict.items():
        if method == "gis":
            gis_idx: np.array[bool] = mod.obs["is_gis"] == True

            gis_normalised_data: np.array[float] = normalise_gis(
                arr=mod.X, gis_idx=gis_idx
            )

            gis_drop_mod = mod[~gis_idx]
            gis_drop_mod.X = gis_normalised_data

            mdata.mod[mod_name] = gis_drop_mod

        elif method == "median_center":
            median_centered_data = Normalisation(method="median", axis="var").normalise(
                arr=mod.X
            )
            mdata[mod_name].X = median_centered_data

        else:
            raise ValueError(f"Method {method} not recognised. Please choose from 'gis' or 'median_center'")

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


class BatchCorrection:
    def __init__(self, method: str) -> None:
        self._method_call: Callable = getattr(self, f"_{method}")

    def _remove_batch_effect(self, arr) -> np.ndarray:
        pass

    def correct(self, arr) -> np.ndarray:
        return self._method_call(arr=arr)


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


def _trim_blank_gis(adata, ignore_blank, ignore_gis):
    invalid_idx = np.array([])
    if ignore_blank:
        invalid_idx = np.append(invalid_idx, np.where(adata.obs["is_blank"]))
    if ignore_gis:
        invalid_idx = np.append(invalid_idx, np.where(adata.obs["is_gis"]))

    valid_idx = np.setdiff1d(np.arange(adata.shape[0]), invalid_idx)
    trimmed_arr = adata[valid_idx].X

    return trimmed_arr, valid_idx


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


class BatchNormalisation(Normalisation):
    """Normalise across experimental Batches"""

    def __init__(self, method: str) -> None:
        super().__init__(method=method)

    def reshape(self, arr): ...

    def inverse_shape(self, normalised_arr) -> np.ndarray:
        return super().inverse_shape(normalised_arr=normalised_arr)

    def normalise(self, arr):
        return super().normalise(arr)
