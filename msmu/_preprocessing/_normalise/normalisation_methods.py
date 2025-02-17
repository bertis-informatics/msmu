import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import rankdata

# def normalise_quantile(data: np.ndarray):
#    """column wise quantile normalisation"""
#
#    data_shape = data.shape
#    normalised_data = np.full_like(data, np.nan)
#    ranked_data = np.full_like(data, np.nan)


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


def normalise_total_sum(): ...
