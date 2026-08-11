"""Shared label-permutation building blocks for a two-group comparison.

Enumerate/generate label splits, split the data, compute one permutation's log2 fold change, and
turn a fold-change null into a symmetric percentile threshold. Both the hypothesis test
(``_permutation.PermutationTest``) and the engine-independent fold-change guidance line
(``_fc_threshold.compute_fc_thresholds``) build on these, so for the same data they produce
identical permutation nulls (an exact-enumeration design yields identical splits, hence an
identical guidance line across engines).
"""

import math
from itertools import combinations

import numpy as np

from ._statistics import _measure_central_tendency, _calc_log2fc


def count_combinations(n_ctrl: int, n_expr: int) -> int:
    return math.comb(n_ctrl + n_expr, n_ctrl)


def resolve_method(n_ctrl: int, n_expr: int, n_resamples: int, force_resample: bool) -> str:
    """Return "exact" (enumerate every split) or "randomised" for this design and n_resamples."""
    combination_count = count_combinations(n_ctrl, n_expr)
    if n_resamples == -np.inf or n_resamples == combination_count:
        return "exact"
    if n_resamples > combination_count:
        return "randomised" if force_resample else "exact"
    return "randomised"


def make_iterations(n_ctrl: int, n_expr: int, method: str, n_resamples: int) -> list:
    total_sample_num = n_ctrl + n_expr
    if method == "exact":
        return list(combinations(range(total_sample_num), n_ctrl))
    return [np.random.permutation(range(total_sample_num)) for _ in range(n_resamples)]


def split(concated_arr: np.ndarray, combination, method: str, n_ctrl: int) -> tuple[np.ndarray, np.ndarray]:
    if method == "exact":
        total_index = np.arange(concated_arr.shape[0])
        ctrl_idx = list(combination)
        expr_idx = np.delete(total_index, ctrl_idx)
    else:  # randomised
        total_index = combination
        ctrl_idx = total_index[:n_ctrl]
        expr_idx = total_index[n_ctrl:]
    return concated_arr[ctrl_idx, :], concated_arr[expr_idx, :]


def permuted_log2fc(
    concated_arr: np.ndarray,
    combination,
    method: str,
    n_ctrl: int,
    measure: str,
    log_transformed: bool,
) -> np.ndarray:
    """log2 fold change for a single label permutation: split, central tendency each side, difference."""
    perm_ctrl, perm_expr = split(concated_arr, combination, method, n_ctrl)
    return _calc_log2fc(
        _measure_central_tendency(perm_ctrl, measure),
        _measure_central_tendency(perm_expr, measure),
        log_transformed=log_transformed,
    )


def fc_threshold_from_null(null_med_diff: np.ndarray, percentile: int) -> float:
    """Symmetric fold-change magnitude bounding the central (100-2*percentile)% of the null."""
    x = np.asarray(null_med_diff)
    if x.ndim == 2:
        x = x.ravel()
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")
    p = float(percentile)
    low = np.nanpercentile(x, p)  # e.g., 5th
    high = np.nanpercentile(x, 100.0 - p)  # e.g., 95th
    threshold = (abs(low) + abs(high)) / 2.0

    return round(float(threshold), 2)
