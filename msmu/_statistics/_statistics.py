import warnings
from dataclasses import dataclass
from typing import Callable
import logging

from msmu._statistics._de_base import StatTestResult
import numpy as np
from scipy.stats import ranksums, t

from ._multiple_test_correction import PvalueCorrection


logger = logging.getLogger(__name__)


@dataclass
class StatResult:
    """
    Data class to store statistical test results.

    Attributes:
        stat_method: The statistical method used.
        statistic: Array of test statistics.
        p_value: Array of p-values.
    """

    stat_method: str
    statistic: np.ndarray
    p_value: np.ndarray


@dataclass
class NullDistribution:
    """
    Data class to store null distribution from permutation tests.

    Attributes:
        method: The statistical method used.
        null_distribution: 2D array of null test statistics (shape: [n_permutations, n_features]).
    """

    stat_method: str
    null_distribution: np.ndarray

    def add_permutation_result(self, other: StatResult):
        """
        Add (stack) a new permutation result to the null distribution.

        Parameters:
            other: A StatResult object containing the statistic from a new permutation.

        Returns:
            A new NullDistribution object with the updated null distribution.
        """
        row = np.atleast_2d(np.asarray(other.statistic))
        nd = self.null_distribution

        if nd.size == 0:
            nd2d = row
        else:
            nd2d = np.atleast_2d(nd)
            if nd2d.shape[1] != row.shape[1] and nd2d.shape[0] == row.shape[1]:
                nd2d = nd2d.T
            nd2d = np.vstack([nd2d, row])

        return NullDistribution(stat_method=self.stat_method, null_distribution=nd2d)


def simple_test(
    ctrl: np.ndarray,
    expr: np.ndarray,
    stat_method: str,
    fdr: bool | str = "bh",
) -> StatTestResult:
    """
    Perform a simple statistical test between two groups.

    Parameters:
        ctrl: array-like (n_samples_ctrl x n_features)
        expr: array-like (n_samples_expr x n_features)
        stat_method: Statistical test to perform ('welch', 'student', 'wilcoxon', 'med_diff').
        fc_method: Method to calculate fold change ('med_diff' or 'mean_diff').
        fdr: Method for multiple test correction ('bh', 'storey', or False).

    Returns:
        StatResult containing the test statistics and p-values.
    """
    test_res = HypothesisTesting.test(ctrl=ctrl, expr=expr, stat_method=stat_method)

    if fdr and test_res.p_value is not None:
        if fdr == "bh":
            corrected_pvals = PvalueCorrection.bh(test_res.p_value)
        else:
            logger.error(f"'bh' is the only implemented FDR correction method for simple test. Set fdr='bh'.")
            raise NotImplementedError(
                f"'bh' is the only implemented FDR correction method for simple test. Set fdr='bh'."
            )

    stat_res = StatTestResult(
        stat_method=test_res.stat_method,
        p_value=test_res.p_value,
        q_value=corrected_pvals if fdr else None,
    )

    return stat_res


class HypothesisTesting:
    """
    Class for performing statistical tests between two groups of samples.

    Attributes:
        method: The statistical method to use ('welch', 'student', 'wilcoxon', 'med_diff').
    """

    @staticmethod
    def test(
        ctrl,
        expr,
        stat_method: str,
    ) -> StatResult:
        stat_dict: dict[str, Callable] = {
            "welch": HypothesisTesting.welch,
            "student": HypothesisTesting.student,
            "wilcoxon": HypothesisTesting.wilcoxon_rank_sum,
        }

        stat_method: Callable = stat_dict[stat_method]
        stat, pval = stat_method(ctrl, expr)

        return StatResult(stat_method=stat_method, statistic=stat, p_value=pval)

    @staticmethod
    def welch(ctrl: np.ndarray, expr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Welch's t-test with NaN handling (manual implementation).
        Not using scipy because of time complexity.

        Parameters:
            ctrl: array-like (n_samples_ctrl x n_features)
            expr: array-like (n_samples_expr x n_features)

        Returns:
            t_val: T-statistics for each feature.
            pval: Two-tailed p-values.
        """
        ctrl = np.asarray(ctrl)
        expr = np.asarray(expr)

        # Means
        mean_ctrl = np.nanmean(ctrl, axis=0)
        mean_expr = np.nanmean(expr, axis=0)

        # Variances (ddof=1 for sample variance)
        # Ignore NaN warnings for variance calculation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            var_ctrl = np.nanvar(ctrl, axis=0, ddof=1)
            var_expr = np.nanvar(expr, axis=0, ddof=1)

            # Sample sizes (account for NaNs)
            n_ctrl = np.sum(~np.isnan(ctrl), axis=0)
            n_expr = np.sum(~np.isnan(expr), axis=0)

            # T-statistic
            denom = np.sqrt(var_ctrl / n_ctrl + var_expr / n_expr)
            t_val = (mean_expr - mean_ctrl) / denom

            # Degrees of freedom (Welch–Satterthwaite equation)
            df_num = (var_ctrl / n_ctrl + var_expr / n_expr) ** 2
            df_denom = (var_ctrl**2 / ((n_ctrl**2) * (n_ctrl - 1))) + (var_expr**2 / ((n_expr**2) * (n_expr - 1)))
            df = df_num / df_denom

            # Handle divisions by zero or invalid DOF
            invalid = (n_ctrl < 2) | (n_expr < 2) | np.isnan(t_val) | np.isnan(df)
            t_val[invalid] = np.nan
            df[invalid] = np.nan

            # Two-sided p-value
            pval = 2 * t.sf(np.abs(t_val), df)

        return t_val, pval

    @staticmethod
    def student(ctrl: np.ndarray, expr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Student's t-test with NaN handling (equal variance assumed).
        Not using scipy because of time complexity.

        Parameters:
            ctrl: array-like (n_samples_ctrl x n_features)
            expr: array-like (n_samples_expr x n_features)

        Returns:
            t_val: T-statistics for each feature.
            pval: Two-tailed p-values.
        """
        ctrl = np.asarray(ctrl)
        expr = np.asarray(expr)

        # Means
        mean_ctrl = np.nanmean(ctrl, axis=0)
        mean_expr = np.nanmean(expr, axis=0)

        # Variances (ddof=1 for sample variance)
        with warnings.catch_warnings():  # make silent nan warnings
            warnings.simplefilter("ignore", category=RuntimeWarning)
            var_ctrl = np.nanvar(ctrl, axis=0, ddof=1)
            var_expr = np.nanvar(expr, axis=0, ddof=1)

            # Sample sizes
            n_ctrl = np.sum(~np.isnan(ctrl), axis=0)
            n_expr = np.sum(~np.isnan(expr), axis=0)

            # Pooled variance (equal variance assumption)
            pooled_var = ((n_ctrl - 1) * var_ctrl + (n_expr - 1) * var_expr) / (n_ctrl + n_expr - 2)

            # T-statistic
            denom = np.sqrt(pooled_var * (1 / n_ctrl + 1 / n_expr))
            t_val = (mean_expr - mean_ctrl) / denom

            # Degrees of freedom
            df = (n_ctrl + n_expr - 2).astype(float)

            # Handle invalid cases
            invalid = (n_ctrl < 2) | (n_expr < 2) | np.isnan(t_val) | np.isnan(df)
            t_val[invalid] = np.nan
            df[invalid] = np.nan

            # Two-sided p-value
            pval = 2 * t.sf(np.abs(t_val), df)

        return t_val, pval

    @staticmethod
    def wilcoxon_rank_sum(ctrl: np.ndarray, expr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Wilcoxon rank-sum test (Mann-Whitney U test) with NaN handling.
        Uses scipy's ranksums function which handles NaNs internally.

        Parameters:
            ctrl: array-like (n_samples_ctrl x n_features)
            expr: array-like (n_samples_expr x n_features)
        Returns:
            stat: Test statistics for each feature.
            pval: Two-tailed p-values.
        """
        stat, pval = ranksums(ctrl, expr, axis=0)

        return stat, pval


def _measure_central_tendency(arr: np.ndarray, method: str) -> np.ndarray:
    """
    Measure central tendency (median or mean) with NaN handling.

    Parameters:
        arr: array-like (n_samples x n_features)
        method: 'median' or 'mean'

    Returns:
        central_tendency: Central tendency values for each feature.
    """
    if method == "median":
        central_tendency = np.nanmedian(arr, axis=0)
    elif method == "mean":
        central_tendency = np.nanmean(arr, axis=0)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'median' or 'mean'.")

    return central_tendency


def _calc_log2fc(ctrl: np.ndarray, expr: np.ndarray, log_transformed: bool) -> np.ndarray:
    """
    Calculate fold change between two groups.

    Parameters:
        ctrl: repersentative statistics of control group (n_features,)
        expr: repersentative statistics of experimental group (n_features,)
        log_transformed: Whether the data is log-transformed.

    Returns:
        log2fc: Fold change values for each feature.
    """
    if log_transformed:
        log2fc = expr - ctrl
    else:
        log2fc = expr / ctrl
        log2fc = np.log2(log2fc + 1e-9)  # add small constant to avoid log2(0)

    return log2fc


def _get_pct_expression(arr: np.ndarray) -> np.ndarray:
    pct_expr = np.sum(~np.isnan(arr), axis=0) / arr.shape[0] * 100

    return pct_expr


def calc_permutation_pvalue(stat_obs: np.ndarray, null_dist: np.ndarray) -> np.ndarray:
    """
    Permutation-based empirical p-value calculation (two-sided).

    Parameters:
        stat_obs: 1D array of observed test statistics (one per feature).
        null_dist: 2D array of null test statistics (shape: [n_permutations, n_features]).

    Returns:
        Array of empirical p-values (NaN-filled where stat_obs was NaN).
    """
    stat_obs = np.asarray(stat_obs)
    valid_mask = ~np.isnan(stat_obs)
    stat_obs_valid = stat_obs[valid_mask]
    abs_stat_obs_valid = np.abs(stat_obs_valid)

    pooled_null = np.abs(np.asarray(null_dist)).ravel()
    pooled_null = pooled_null[~np.isnan(pooled_null)]
    pooled_null = np.sort(pooled_null)

    pvals = np.full_like(stat_obs, np.nan, dtype=float)
    left_idx = np.searchsorted(pooled_null, abs_stat_obs_valid, side="left")  # left: ">="
    exceeded = pooled_null.size - left_idx
    pvals[valid_mask] = (exceeded + 1) / (pooled_null.size + 1)

    return pvals


class Limma: ...
