from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import ranksums, ttest_ind
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
from scipy.stats import t

@dataclass
class StatResult:
    stat_method: str
    statistic: np.ndarray
    p_value: np.ndarray


class StatTest:
    @staticmethod
    def _stat_tests(ctrl, expr, statistic: str) -> StatResult:
        stat_dict: dict[str, Callable] = {
            "t_test": StatTest.t_test,
            "wilcoxon": StatTest.wilcoxon_rank_sum,
            "med_diff": StatTest.median_diff,
        }

        stat_method: Callable = stat_dict[statistic]
        stat, pval = stat_method(ctrl, expr)

        return StatResult(stat_method=statistic, statistic=stat, p_value=pval)

    @staticmethod
    def t_test(ctrl, expr):
        """
        Welch's t-test with NaN handling (manual implementation).

        Parameters:
        -----------
        ctrl : array-like (n_samples_ctrl x n_features)
        expr : array-like (n_samples_expr x n_features)

        Returns:
        --------
        t_val : np.ndarray
            T-statistics for each feature.
        pval : np.ndarray
            Two-tailed p-values.
        """
        ctrl = np.asarray(ctrl)
        expr = np.asarray(expr)

        # Means
        mean_ctrl = np.nanmean(ctrl, axis=0)
        mean_expr = np.nanmean(expr, axis=0)

        # Variances (ddof=1 for sample variance)
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
    
    # @staticmethod
    # def t_test(ctrl, expr):
    #     mean_ctrl = np.nanmean(ctrl, axis=0)
    #     mean_expr = np.nanmean(expr, axis=0)
    #     var_ctrl = np.nanvar(ctrl, axis=0)
    #     var_expr = np.nanvar(expr, axis=0)

    #     t_val = (mean_expr - mean_ctrl) / np.sqrt(
    #         var_expr / len(ctrl) + var_ctrl / len(expr)
    #     )

    #     t_val[np.isinf(t_val)] = np.nan
    #     pval = None
    #     # t_val, pval = ttest_ind(ctrl, expr, axis=0, equal_var=False, nan_policy="omit")

    #     return t_val, pval

    @staticmethod
    def wilcoxon_rank_sum(ctrl, expr):
        stat, pval = ranksums(ctrl, expr, axis=0)

        return stat, pval

    @staticmethod
    def median_diff(ctrl, expr):
        med_diff = np.nanmedian(expr, axis=0) - np.nanmedian(ctrl, axis=0)

        return med_diff, None

    # @staticmethod
    # def pval2tail(stat_obs, null_dist):
    #     # p-value computation routine
    #     # s0: null statistic, s: observed statistic
    #     # p-value : computed by two-tail test

    #     # stacked_s0 = np.hstack(null_dist)
    #     stacked_s0 = null_dist
    #     ecdf_res = ECDF(stacked_s0)
    #     null_dist = ecdf_res.x[1:]
    #     f0 = ecdf_res.y[1:]

    #     f = interp1d(null_dist, f0, bounds_error=False)
    #     p = f(stat_obs)
    #     p[np.isnan(p)] = 0
    #     p = 2 * np.min(np.c_[p, 1 - p], axis=1)

    #     if np.sum(p != 0) != 0:
    #         p[p == 0] = np.nanmin(p[p != 0]) / 2
    #     else:
    #         p[p == 0] = 1e-10
    #     p[p == 1] = (1 - np.max(p[p != 1])) / 2 + np.max(p[p != 1])

    #     return p


    @staticmethod
    def pval2tail(stat_obs, null_dist, min_pval=1e-10):
        """
        Compute two-sided empirical p-values using ECDF interpolation.

        Parameters:
        -----------
        stat_obs : array-like
            Observed test statistics (can include NaN).
        null_dist : array-like
            Pooled null distribution (1D array).
        min_pval : float
            Minimum p-value to avoid zeros.

        Returns:
        --------
        pvals : np.ndarray
            Array of p-values (NaN where stat_obs is NaN).
        """
        stat_obs = np.asarray(stat_obs)
        pooled_null = np.asarray(null_dist)

        # Handle edge case: all nulls are NaN
        if np.all(np.isnan(pooled_null)):
            return np.full_like(stat_obs, np.nan, dtype=float)

        # Build ECDF from absolute null distribution
        ecdf = ECDF(np.abs(pooled_null[~np.isnan(pooled_null)]))
        ecdf_x = ecdf.x
        ecdf_y = ecdf.y

        # Interpolator for one-sided p
        interp = interp1d(ecdf_x, ecdf_y, bounds_error=False, fill_value=(0.0, 1.0))

        # Prepare output array
        pvals = np.full_like(stat_obs, np.nan, dtype=float)

        # Apply only to non-NaN observed stats
        mask = ~np.isnan(stat_obs)
        abs_obs = np.abs(stat_obs[mask])
        p_one_sided = interp(abs_obs)

        # Two-sided p-value: 2 * min(p, 1 - p)
        pval = 2 * np.minimum(p_one_sided, 1 - p_one_sided)

        # Clip extremely low or high p-values
        pval = np.clip(pval, min_pval, 1 - min_pval)

        # Store in output array
        pvals[mask] = pval

        return pvals

    #@staticmethod
    # def pval_calc_test(stat_obs, null_dist):
    #     pval = np.zeros_like(stat_obs)
    #     for i in range(len(stat_obs)):
    #         pval[i] = (np.sum(np.abs(null_dist) >= np.abs(stat_obs[i])) + 1) / (len(null_dist) + 1)
    #     return pval

    @staticmethod 
    def pval_calc_test(stat_obs, null_dist):
        stat_obs = np.asarray(stat_obs)
        null_dist = np.abs(np.asarray(null_dist))
        
        # Initialize p-values with NaNs
        pvals = np.full_like(stat_obs, np.nan, dtype=float)
        
        # Identify valid (non-NaN) indices
        valid_mask = ~np.isnan(stat_obs)
        
        # Only compute for valid stats
        stat_obs_valid = np.abs(stat_obs[valid_mask])[:, np.newaxis]
        null_dist_expanded = null_dist[np.newaxis, :]
        
        more_extreme = null_dist_expanded >= stat_obs_valid

        pvals[valid_mask] = (np.sum(more_extreme, axis=1) + 1) / (null_dist.size + 1)

        return pvals

@dataclass
class NullDistribution:
    method: str
    null_distribution: np.ndarray

    def add_permutation_result(self, other: StatResult):
        return NullDistribution(
            method=self.method,
            null_distribution=np.concatenate(
                (self.null_distribution, other.statistic), axis=0
            ),
        )
