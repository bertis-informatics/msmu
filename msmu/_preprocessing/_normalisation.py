import numpy as np
import pandas as pd
import mudata as md
from scipy.interpolate import interp1d
from scipy.stats import rankdata
from sklearn.linear_model import Ridge

from typing import Callable


class Normalisation:
    def __init__(self, method: str, axis: str) -> None:
        self._method_call: Callable = getattr(self, f"_{method}")
        self._axis = axis

    def _quantile(self, arr) -> np.ndarray:
        return normalise_quantile(arr=arr)

    def _median(self, arr) -> np.ndarray:
        all_median = np.nanmedian(arr.flatten())
        arr = normalise_median_center(arr=arr)  # median center
        arr = arr + all_median  # add scalar to preserve overall scale

        return arr

    def _median_center(self, arr) -> np.ndarray:
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


def normalise_quantile(arr: np.ndarray) -> np.ndarray:
    # set defaults
    values = np.array(arr)
    tiedFlag = True

    # allocate some space for the normalized values
    normalizedVals = values
    valSize = values.shape
    rankedVals = np.zeros(valSize) * np.nan

    # find nans
    nanvals = np.isnan(values)
    numNans = np.sum(nanvals, axis=0)
    ndx = np.ones(valSize, dtype=np.int64)
    N = valSize[0]

    # create space for output
    if tiedFlag == True:
        rr = np.empty([valSize[1]], dtype=object)

    # for each column we want to ordered values and the ranks with ties
    for col in range(valSize[1]):
        sortedVals = np.sort(values[:, col])
        ndx[:, col] = np.argsort(values[:, col])
        if tiedFlag:
            rr[col] = np.sort(rankdata(values[~nanvals[:, col], col]))
        M = N - numNans[col]
        x = np.arange(0, N, (N - 1) / (M - 1))
        y = sortedVals[0:M]
        try:
            f = interp1d(x, y, bounds_error=False)
        except:
            print(f"Error occured at {col}: {y.shape}")
            print(values)
            exit
        xnew = np.arange(0, N)
        ynew = f(xnew)
        rankedVals[:, col] = ynew

    # take the mean of the ranked values
    mean_vals = np.nanmean(rankedVals, axis=1)

    # Extract the values from the normalized distribution
    for col in range(valSize[1]):
        M = N - numNans[col]
        if tiedFlag:
            x = np.arange(0, N)
            y = mean_vals
            f = interp1d(x, y, bounds_error=False)
            xnew = (N - 1) * (rr[col] - 1) / (M - 1)
            ynew = f(xnew)
            normalizedVals[ndx[0:M, col], col] = ynew
        else:
            x = np.arange(0, N)
            y = mean_vals
            f = interp1d(x, y, bounds_error=False)
            xnew = np.arange(0, N, (N - 1) / (M - 1))
            ynew = f(xnew)
            normalizedVals[ndx[0:M, col], col] = ynew

    normalizedVals[nanvals] = np.nan

    return normalizedVals


def normalise_median_center(arr: np.ndarray) -> np.ndarray:
    """Median centering of data"""
    raw_arr = arr.copy()
    median_data = np.nanmedian(raw_arr, axis=0)

    median_centered_data = raw_arr - median_data

    return median_centered_data


def normalise_total_sum():
    """Total sum normalisation of data"""
    raise NotImplementedError("Total sum normalisation is not implemented yet.")


class PTMProteinAdjuster:
    def __init__(self, ptm_mdata: md.MuData, global_mdata: md.MuData, ptm_mod: str, global_mod: str):
        self.ptm_mdata = ptm_mdata
        self.ptm_mod = ptm_mod
        self.global_mdata = global_mdata
        self.global_mod = global_mod
        self.sample_cols: list[str] = list(ptm_mdata.obs.index)

        self.ptm_data, self.global_data = self._extract_data()

    def _extract_data(self):
        ptm_data: pd.DataFrame = self.ptm_mdata[self.ptm_mod].to_df().T.copy()
        ptm_data["ptm_site"] = ptm_data.index
        ptm_data["protein_group"] = self.ptm_mdata[self.ptm_mod].var["protein_group"]

        global_data: pd.DataFrame = self.global_mdata[self.global_mod].to_df().T.copy()
        global_data = global_data[self.sample_cols]  # sort sample order
        global_data["protein_group"] = global_data.index

        common_protein_group: set = set(ptm_data["protein_group"]).intersection(set(global_data["protein_group"]))

        ptm_data = ptm_data.loc[ptm_data["protein_group"].isin(common_protein_group)]
        global_data = global_data.loc[global_data["protein_group"].isin(common_protein_group)]

        return ptm_data, global_data

    def _ratio(self):
        ptm_values = self.ptm_data[self.sample_cols]
        global_values = self.global_data.loc[self.ptm_data["protein_group"], self.sample_cols].reset_index(drop=True)

        result = ptm_values.values - global_values.values

        result_df = self.ptm_data.copy()
        result_df[self.sample_cols] = result

        return result_df

    def _ridge(self, alpha=100) -> pd.DataFrame:
        records: list = list()

        for pid, grp in self.ptm_data.groupby("protein_group", sort=False, observed=True):
            x_full = self.global_data.loc[pid, self.sample_cols].to_numpy(float)
            for _, row in grp.iterrows():
                y_full: np.ndarray = row[self.sample_cols].to_numpy(float)

                valid_mask: np.ndarray = ~np.isnan(x_full) & ~np.isnan(y_full)
                if valid_mask.sum() <= 2:
                    continue

                x_valid: np.ndarray = x_full[valid_mask].reshape(-1, 1)
                y_valid: np.ndarray = y_full[valid_mask]

                model = Ridge(alpha=alpha, fit_intercept=True).fit(x_valid, y_valid)

                y_hat: np.ndarray = np.full_like(y_full, np.nan, dtype=float)
                y_hat[valid_mask] = model.predict(x_valid)

                residual: np.ndarray = y_full - y_hat

                records.append({"ptm_site": row["ptm_site"], "protein_group": pid, "residual": residual})

        result_df: pd.DataFrame = pd.DataFrame(records)
        residual_df = result_df.drop(columns="residual").copy()
        residual_values = pd.DataFrame(result_df["residual"].tolist(), columns=self.sample_cols)

        result_df = pd.concat([residual_df, residual_values], axis=1)

        return result_df

    def _adjuted_ptm_to_mdata(self, adjusted_ptm: pd.DataFrame) -> md.MuData:
        adj_ptm_mdata: md.MuData = self.ptm_mdata.copy()
        adj_ptm_adata = adj_ptm_mdata[self.ptm_mod].copy()
        adj_ptm_adata = adj_ptm_adata[:, adjusted_ptm["ptm_site"]].copy()

        adjusted_ptm = adjusted_ptm.set_index("ptm_site", drop=True)
        adjusted_ptm = adjusted_ptm.drop(columns="protein_group")
        adjusted_ptm = adjusted_ptm.rename_axis(index=None)
        adj_ptm_adata.X = adjusted_ptm.T

        adj_ptm_mdata.mod[self.ptm_mod] = adj_ptm_adata.copy()
        adj_ptm_mdata.update()

        return adj_ptm_mdata

    def _rescale(self, adjusted_ptm: pd.DataFrame) -> pd.DataFrame:
        total_median: float = np.nanmedian(self.ptm_data[self.sample_cols].to_numpy().flatten())
        adjusted_ptm[self.sample_cols] = adjusted_ptm[self.sample_cols] + total_median

        return adjusted_ptm

    def adjust(self, method: str, rescale: bool) -> md.MuData:
        adjust_method = getattr(self, f"_{method}")
        adjusted_ptm = adjust_method()
        if rescale:
            adjusted_ptm = self._rescale(adjusted_ptm)

        adj_ptm_mdata = self._adjuted_ptm_to_mdata(adjusted_ptm)

        return adj_ptm_mdata
