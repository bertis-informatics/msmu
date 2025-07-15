import warnings
from dataclasses import dataclass
from typing import Callable
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ranksums, t, ttest_ind
from statsmodels.stats.multitest import multipletests


@dataclass
class StatResult:
    stat_method: str
    statistic: np.ndarray
    p_value: np.ndarray


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


class StatTest:
    @staticmethod
    def _stat_tests(ctrl, expr, statistic: str) -> StatResult:
        stat_dict: dict[str, Callable] = {
            "welch": StatTest.welch,
            "student": StatTest.student,
            "wilcoxon": StatTest.wilcoxon_rank_sum,
            "med_diff": StatTest.median_diff,
        }

        stat_method: Callable = stat_dict[statistic]
        stat, pval = stat_method(ctrl, expr)

        return StatResult(stat_method=statistic, statistic=stat, p_value=pval)


    @staticmethod
    def welch(ctrl, expr): # welch
        """
        Welch's t-test with NaN handling (manual implementation).
        Not using scipy because of time complexity.

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
            df_denom = (var_ctrl**2 / ((n_ctrl**2) * (n_ctrl - 1))) + (
                var_expr**2 / ((n_expr**2) * (n_expr - 1))
            )
            df = df_num / df_denom

            # Handle divisions by zero or invalid DOF
            invalid = (n_ctrl < 2) | (n_expr < 2) | np.isnan(t_val) | np.isnan(df)
            t_val[invalid] = np.nan
            df[invalid] = np.nan

            # Two-sided p-value
            pval = 2 * t.sf(np.abs(t_val), df)

        return t_val, pval

    # @staticmethod
    # def welch(ctrl, expr):
    #     res = ttest_ind(ctrl, expr, equal_var=False, nan_policy="omit")
    #     t = res.statistic
    #     p = res.pvalue

    #     return t, p

    @staticmethod
    def student(ctrl, expr):
        """
        Student's t-test with NaN handling (equal variance assumed).
        Not using scipy because of time complexity.

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
        with warnings.catch_warnings():
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
    def wilcoxon_rank_sum(ctrl, expr):
        stat, pval = ranksums(ctrl, expr, axis=0)

        return stat, pval

    @staticmethod
    def median_diff(ctrl, expr):
        med_diff = np.nanmedian(expr, axis=0) - np.nanmedian(ctrl, axis=0)

        return med_diff, None

    @staticmethod
    def calc_permutation_pvalue(stat_obs, null_dist):
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


class PvalueCorrection:
    @staticmethod
    def bh(pvals: np.ndarray):
        pvals = np.asarray(pvals)
        qvals = np.full_like(pvals, np.nan, dtype=float)
        mask = ~np.isnan(pvals)
        if np.any(mask):
            _, qvals_nonan, _, _ = multipletests(pvals[mask], method="fdr_bh")
            qvals[mask] = qvals_nonan
        return qvals

    @staticmethod
    def storey(p_values: np.ndarray, lambda_: float = 0.5, alpha: float = 0.05, return_mask: bool = False):
        """
        Storey (2002) q-value estimation with pi0 estimation.

        Parameters
        ----------
        p_values : array-like
            Array of p-values (can include NaN).
        lambda_ : float
            Threshold for estimating pi0 (0 < lambda < 1). Default = 0.5.
        alpha : float
            FDR threshold for significance mask (only if return_mask=True).
        return_mask : bool
            If True, also returns Boolean significance mask.

        Returns
        -------
        q_values : np.ndarray
            Array of q-values (NaN-filled where p was NaN).
        rejected : Optional[np.ndarray]
            Boolean array indicating which features are significant under FDR < alpha.
        """
        p_values = np.asarray(p_values)
        q_values = np.full_like(p_values, np.nan, dtype=float)
        rejected_mask = np.full_like(p_values, False, dtype=bool)

        # Step 1: Remove NaN
        valid_mask = ~np.isnan(p_values)
        p_valid = p_values[valid_mask]
        m = len(p_valid)

        # Step 2: Estimate π₀
        pi0 = np.minimum(1.0, np.sum(p_valid > lambda_) / ((1.0 - lambda_) * m))

        # Step 3: Sort p-values and compute BH-like q
        sorted_idx = np.argsort(p_valid)
        sorted_p = p_valid[sorted_idx]
        ranks = np.arange(1, m + 1)
        q = pi0 * sorted_p * m / ranks

        # Step 4: Cumulative minimum (monotonic q-values)
        q = np.minimum.accumulate(q[::-1])[::-1]
        q = np.clip(q, 0, 1)

        # Step 5: Map back to original index
        q_valid = np.empty_like(p_valid)
        q_valid[sorted_idx] = q
        q_values[valid_mask] = q_valid

        # Step 6: Optional significance mask
        if return_mask:
            rejected_mask[valid_mask] = q_valid <= alpha
            return q_values, rejected_mask
        else:
            return q_values

    @staticmethod
    def estimate_pi0_storey(p_values, lambdas=np.linspace(0.5, 0.95, 10)):
        """
        Storey's estimator of pi0 (proportion of true nulls) from observed p-values.
        https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2013.00179/full
        Based on Equation (7)
        pi0 = #( pval > lamda ) / ( 1 - lambda ) * m

        Parameters:
        - p_values: array of p-values (one per feature)
        - lambdas: array of lambda thresholds (typically 0.5 to 0.95)

        Returns:
        - pi0: estimated pi0 value
        - pi0_by_lambda: array of intermediate pi0 estimates
        """
        p_values = np.asarray(p_values)
        valid_mask = ~np.isnan(p_values)
        p_values = p_values[valid_mask]
        m = len(p_values)
        
        pi0_by_lambda = []
        for lam in lambdas:
            count = np.sum(p_values > lam)
            pi0_hat = count / ((1 - lam) * m)
            pi0_by_lambda.append(min(pi0_hat, 1.0))

        pi0_by_lambda = np.array(pi0_by_lambda)
        pi0 = np.min(pi0_by_lambda)

        return pi0, pi0_by_lambda

    @staticmethod
    def estimate_pi0_null(stat_valid, null_matrix_valid, percentile=95):
        """
        Estimate pi0 (proportion of true null hypotheses) using permutation-based statistic exceedance method.
        https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2013.00179/full
        Based on Equation (8): compares observed and null test statistic exceedances at a given threshold.
        pi0 = (1 - S/m) / (1 - S_star/m)

        Parameters
        ----------
        stat_valid : np.ndarray
            1D array of observed test statistics (NaN-excluded).
        null_matrix_valid : np.ndarray
            2D array of null test statistics (shape: [n_permutations, m_valid]),
            aligned with stat_valid (i.e., same features, same filtering).
        percentile : float, default=95
            Percentile value used to define the threshold for exceedance comparison.

        Returns
        -------
        pi0 : float
            Estimated proportion of true null hypotheses (clipped to [0, 1]).
        """
        m = stat_valid.size
        threshold = np.percentile(stat_valid, percentile)

        S = np.sum(stat_valid >= threshold)
        S_star = np.mean(np.sum(null_matrix_valid >= threshold, axis=0))
        denominator = 1 - (S_star / m) 
        pi0 = (1 - S / m) / denominator if denominator != 0 else 1.0
        pi0 = min(max(pi0, 0.0), 1.0)

        return pi0

    @staticmethod
    def empirical(
        stat_obs: np.ndarray,
        null_dist: np.ndarray,
        pvals: np.ndarray,
        two_sided: bool = True,
    ) -> np.ndarray:
        """
        Permutation-based empirical FDR estimation using:
        - Storey's method for pi0 (default)
        - or permutation-statistic-based method (equation 8)

        References:
        - https://academic.oup.com/bioinformatics/article/21/23/4280/194680
        - https://www.pnas.org/doi/epdf/10.1073/pnas.1530509100

        E[FDR] = pi0 * E[FP] / E[TP]
        E[FP] = #(FP >= s) / B (# permutation)
        E[TP] = #(TP >= s)
        """

        stat_obs = np.asarray(stat_obs)
        null_dist = np.asarray(null_dist)

        B = null_dist.size // stat_obs.size

        # treat nan
        valid_mask = ~np.isnan(stat_obs)
        stat_valid = stat_obs[valid_mask]
        orig_index = np.where(valid_mask)[0]

        # abs for two-sided
        stat_valid = np.abs(stat_valid) if two_sided else stat_valid
        null_valid = null_dist[~np.isnan(null_dist)]
        null_valid = np.abs(null_valid) if two_sided else null_valid

        null_matrix = null_dist.reshape(B, stat_obs.size)
        null_matrix_valid = null_matrix[:, valid_mask]  # shape (B, m)
        null_matrix_valid = np.abs(null_matrix_valid) if two_sided else null_matrix_valid

        # pi0 estimation (direct pi0 estimation from null distribution)
        pi0 = PvalueCorrection.estimate_pi0_null(stat_valid=stat_valid, null_matrix_valid=null_matrix_valid, percentile=95)

        # # pi0 estimation (storey's)
        # pi0, _ = PvalueCorrection.estimate_pi0_storey(p_values=pvals)

        # q-value calculation (FDR = pi0 * E[FP] / E[TP])
        q_vals = []
        for s in stat_valid:
            tp = np.sum(stat_valid >= s)
            fp = np.sum(null_valid >= s)
            e_fp = (fp + 1) / (B + 1)
            e_tp = tp + 1

            fdr = pi0 * e_fp / e_tp
            q_vals.append(fdr)

        # monotonic correction
        sort_idx = np.argsort(-stat_valid)
        q_sorted = np.array(q_vals)[sort_idx]
        q_sorted_monotonic = np.minimum.accumulate(q_sorted[::-1])[::-1]

        # re-order to original index
        q_value_all = np.full_like(stat_obs, np.nan, dtype=float)
        for i, q in zip(orig_index, q_sorted_monotonic[np.argsort(sort_idx)]):
            q_value_all[i] = q

        return np.clip(q_value_all, 0, 1)