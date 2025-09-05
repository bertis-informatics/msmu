import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import ranksums, t
from statsmodels.stats.multitest import multipletests


@dataclass
class StatResult:
    """
    Data class to store statistical test results.
    Attributes:
        stat_method (str): The statistical method used.
        statistic (np.ndarray): Array of test statistics.
        p_value (np.ndarray): Array of p-values.
    """
    stat_method: str
    statistic: np.ndarray
    p_value: np.ndarray


@dataclass
class NullDistribution:
    """
    Data class to store null distribution from permutation tests.
    Attributes:
        method (str): The statistical method used.
        null_distribution (np.ndarray): 2D array of null test statistics (shape: [n_permutations, n_features]).
    """
    method: str
    null_distribution: np.ndarray

    def add_permutation_result(self, other: StatResult):
        """
        Add (stack) a new permutation result to the null distribution.
        Parameters:
        other : StatResult
            A StatResult object containing the statistic from a new permutation.
        Returns:
        NullDistribution
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
        
        return NullDistribution(method=self.method, null_distribution=nd2d)


class StatTest:
    """
    Class for performing statistical tests between two groups of samples.
    Attributes:
        method (str): The statistical method to use ('welch', 'student', 'wilcoxon', 'med_diff').
    """
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
        with warnings.catch_warnings(): # make silent nan warnings
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
        """
        Wilcoxon rank-sum test (Mann-Whitney U test) with NaN handling.
        Uses scipy's ranksums function which handles NaNs internally.
        Parameters:
        -----------
        ctrl : array-like (n_samples_ctrl x n_features)
        expr : array-like (n_samples_expr x n_features)
        Returns:
        --------
        stat : np.ndarray
            Test statistics for each feature.
        pval : np.ndarray
            Two-tailed p-values.
        """
        stat, pval = ranksums(ctrl, expr, axis=0)

        return stat, pval

    @staticmethod
    def median_diff(ctrl, expr):
        """
        Median difference (expr - ctrl) with NaN handling.
        Parameters:
        -----------
        ctrl : array-like (n_samples_ctrl x n_features)
        expr : array-like (n_samples_expr x n_features)
        Returns:
        --------
        med_diff : np.ndarray
            Median differences for each feature.
        """
        med_diff = np.nanmedian(expr, axis=0) - np.nanmedian(ctrl, axis=0)

        return med_diff, None

    @staticmethod
    def calc_permutation_pvalue(
        stat_obs: np.ndarray, 
        null_dist: np.ndarray
        ) -> np.ndarray:
        """
        Permutation-based empirical p-value calculation (two-sided).
        Parameters
        ----------
        stat_obs : np.ndarray
            1D array of observed test statistics (one per feature).
        null_dist : np.ndarray
            2D array of null test statistics (shape: [n_permutations, n_features]).
        Returns
        -------
        pvals : np.ndarray
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
        left_idx = np.searchsorted(pooled_null, abs_stat_obs_valid, side="left") # left: ">="
        exceeded = pooled_null.size - left_idx
        pvals[valid_mask] = (exceeded + 1) / (pooled_null.size + 1)

        return pvals


class PvalueCorrection:
    """
    Class for multiple testing correction methods.
    Methods:
        bh : Benjamini-Hochberg FDR correction.
        storey : Storey's q-value estimation with pi0 estimation.
        empirical : Permutation-based empirical FDR estimation.
    """
    @staticmethod
    def bh(pvals: np.ndarray):
        """
        Benjamini-Hochberg FDR correction with NaN handling.
        Parameters
        ----------
        pvals : array-like
            Array of p-values (can include NaN).
        Returns
        -------
        qvals : np.ndarray
            Array of q-values (NaN-filled where p was NaN).
        """
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
    def estimate_pi0_storey(
        p_values: np.ndarray,
        lambdas: np.ndarray=np.linspace(0.5, 0.95, 10)
        ) -> tuple[float, np.ndarray]:
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
    def estimate_pi0_null(
        stat_valid: np.ndarray,
        null_matrix_valid: np.ndarray, 
        percentile:int=95
        ) -> float:
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

        s = np.sum(stat_valid >= threshold)
        s_star = np.mean(np.sum(null_matrix_valid >= threshold, axis=0))
        denominator = 1 - (s_star / m) 
        pi0 = (1 - s / m) / denominator if denominator != 0 else 1.0
        pi0 = min(max(pi0, 0.0), 1.0)

        return pi0

    @staticmethod
    def empirical(
        stat_obs: np.ndarray,
        null_dist: np.ndarray,
        # pvals: np.ndarray, # optional, if pi0 estimated by storey
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
        null_dist = np.asarray(null_dist).ravel()

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

@dataclass
class StatTestReusult:
    """
    Data class to store and convert statistical test results to a DataFrame.
    Attributes:
        stat_method (str): The statistical method used.
        statistic (str): The statistic computed.
        ctrl (str | None): Control group label.
        expr (str | None): Experimental group label.
        features (pd.Index | np.ndarray | None): Feature identifiers.
        median_ctrl (np.ndarray | None): Median values for control group.
        median_expr (np.ndarray | None): Median values for experimental group.
        pct_ctrl (np.ndarray | None): Percentage of non-missing values in control group.
        pct_expr (np.ndarray | None): Percentage of non-missing values in experimental group.
        log2_fc (np.ndarray | None): Log2 fold changes between groups.
        p_value (np.ndarray | None): P-values from statistical tests.
        q_value (np.ndarray | None): Adjusted q-values for multiple testing.
    """
    statistic: str
    ctrl: str | None
    expr: str | None = None
    features: pd.Index | np.ndarray | None = None
    median_ctrl: np.ndarray | None = None
    median_expr: np.ndarray | None = None
    pct_ctrl: np.ndarray | None = None
    pct_expr: np.ndarray | None = None
    log2_fc: np.ndarray | None = None
    p_value: np.ndarray | None = None
    q_value: np.ndarray | None = None

    def to_df(self) -> pd.DataFrame:
        """
        Convert the statistical test results to a pandas DataFrame.
        Returns:
            pd.DataFrame: DataFrame containing the statistical test results.
        """
        contents: dict ={
            "features": self.features,
            "median_ctrl": self.median_ctrl,
            "median_expr": self.median_expr,
            "pct_ctrl": self.pct_ctrl,
            "pct_expr": self.pct_expr,
            "log2_fc": self.log2_fc,
            "p_value": self.p_value,
            "q_value": self.q_value,
        }
        df = pd.DataFrame(contents)

        return df