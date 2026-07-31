import numpy as np
from statsmodels.stats.multitest import multipletests

# Percentile of the observed statistics used as the exceedance threshold when estimating pi0
# from the permutation null. Because the threshold is a quantile of the very statistics whose
# exceedances are counted, S/m is pinned at (1 - PI0_NULL_PERCENTILE/100) whatever the data,
# so the estimator's numerator (1 - S/m) is always PI0_NULL_PERCENTILE/100 and pi0 cannot fall
# below it.
PI0_NULL_PERCENTILE = 95
PI0_LOWER_BOUND = PI0_NULL_PERCENTILE / 100.0

# R p.adjust / limma adjust.method names -> statsmodels multipletests methods. msmu exposes R
# limma's adjust.method vocabulary; each maps 1:1 onto a statsmodels routine (verified numerically
# against R's p.adjust, see tests/test_statistics_multiple_test_correction.py). "none" is
# intentionally absent — the uncorrected values are always the p-value column itself, so "no
# correction" is not offered as an adjustment method.
_R_TO_STATSMODELS_METHOD = {
    "bh": "fdr_bh",
    "by": "fdr_by",
    "bonferroni": "bonferroni",
    "holm": "holm",
    "hochberg": "simes-hochberg",
    "hommel": "hommel",
}

# The R p.adjust adjustment methods msmu exposes (canonical lowercase names), for callers that
# validate a requested method against the supported family.
P_ADJUST_METHODS = tuple(_R_TO_STATSMODELS_METHOD)


class PvalueCorrection:
    """
    Class for multiple testing correction methods.

    Methods:
        adjust : R p.adjust family (BH/BY/holm/hochberg/hommel/bonferroni) on a p-value vector.
        storey : Storey's q-value estimation with pi0 estimation.
        empirical : Permutation-based empirical FDR estimation.
    """

    @staticmethod
    def adjust(pvals: np.ndarray, method: str = "bh") -> np.ndarray:
        """
        Multiple-testing correction from R's p.adjust family, with NaN handling.

        Maps an R p.adjust / limma ``adjust.method`` name (case-insensitive) onto the matching
        statsmodels ``multipletests`` routine and applies it to the non-NaN p-values, leaving NaN
        entries untouched. This is the shared correction path for both DE engines: limma's moderated
        p-values and the permutation p-values are adjusted identically here (the permutation engine's
        alternative, empirical FDR, lives in :meth:`empirical` because it needs the null distribution,
        not just the p-values).

        Parameters:
            pvals: Array of p-values (can include NaN).
            method: R-style correction name, one of
                "bh", "by", "holm", "hochberg", "hommel", "bonferroni" (case-insensitive).

        Returns:
            qvals: Array of adjusted p-values (NaN-filled where p was NaN).
        """
        method_key = method.lower()
        if method_key not in _R_TO_STATSMODELS_METHOD:
            raise ValueError(
                f"Unknown p-value adjustment method {method!r}. Choose from "
                f"{list(P_ADJUST_METHODS)} (R p.adjust names, case-insensitive)."
            )
        statsmodels_method = _R_TO_STATSMODELS_METHOD[method_key]

        pvals = np.asarray(pvals)
        qvals = np.full_like(pvals, np.nan, dtype=float)
        mask = ~np.isnan(pvals)
        if np.any(mask):
            _, qvals_nonan, _, _ = multipletests(pvals[mask], method=statsmodels_method)
            qvals[mask] = qvals_nonan
        return qvals

    @staticmethod
    def storey(
        p_values: np.ndarray,
        lambda_: float = 0.5,
    ) -> np.ndarray:
        """
        Storey (2002) q-value estimation with pi0 estimation.

        Parameters:
            p_values: Array of p-values (can include NaN).
            lambda_: Threshold for estimating pi0 (0 < lambda < 1). Default = 0.5.
            alpha: FDR threshold for significance mask (only if return_mask=True).
            return_mask: If True, also returns Boolean significance mask.

        Returns:
            Array of q-values (NaN-filled where p was NaN).
            Array of q-values (NaN-filled where p was NaN).
        """
        p_values = np.asarray(p_values)
        q_values = np.full_like(p_values, np.nan, dtype=float)

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

        return q_values

    # @staticmethod
    # def estimate_pi0_storey(
    #     p_values: np.ndarray, lambdas: np.ndarray = np.linspace(0.5, 0.95, 10)
    # ) -> tuple[float, np.ndarray]:
    #     """
    #     Storey's estimator of pi0 (proportion of true nulls) from observed p-values.
    #     https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2013.00179/full
    #     Based on Equation (7)
    #     pi0 = #( pval > lamda ) / ( 1 - lambda ) * m

    #     Parameters:
    #         p_values: array of p-values (one per feature)
    #         lambdas: array of lambda thresholds (typically 0.5 to 0.95)

    #     Returns:
    #         estimated pi0 value
    #         array of intermediate pi0 estimates
    #     """
    #     p_values = np.asarray(p_values)
    #     valid_mask = ~np.isnan(p_values)
    #     p_values = p_values[valid_mask]
    #     m = len(p_values)

    #     pi0_by_lambda = []
    #     for lam in lambdas:
    #         count = np.sum(p_values > lam)
    #         pi0_hat = count / ((1 - lam) * m)
    #         pi0_by_lambda.append(min(pi0_hat, 1.0))

    #     pi0_by_lambda = np.array(pi0_by_lambda)
    #     pi0 = np.min(pi0_by_lambda)

    #     return pi0, pi0_by_lambda

    @staticmethod
    def estimate_pi0_null(stat_valid: np.ndarray, null_matrix_valid: np.ndarray, percentile: int = 95) -> float:
        """
        Estimate pi0 (proportion of true null hypotheses) using permutation-based statistic exceedance method.
        https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2013.00179/full
        Based on Equation (8): compares observed and null test statistic exceedances at a given threshold.
        pi0 = (1 - S/m) / (1 - S_star/m)

        Both S and S_star are counts of *features* exceeding the threshold: S over the observed
        statistics, S_star averaged over permutations. Summing the null matrix over its
        permutation axis instead would make S_star a count of permutations, i.e. roughly
        m/n_permutations times too small, which drives the denominator to 1 and collapses pi0
        onto the constant (1 - percentile/100) regardless of the data.

        Parameters:
            stat_valid: 1D array of observed test statistics (NaN-excluded).
            null_matrix_valid: 2D array of null test statistics (shape: [n_permutations, m_valid]), aligned with stat_valid (i.e., same features, same filtering).
            percentile: Percentile value used to define the threshold for exceedance comparison.

        Returns:
            pi0, Estimated proportion of true null hypotheses (clipped to [0, 1]).
        """
        m = stat_valid.size
        threshold = np.percentile(stat_valid, percentile)

        s = np.sum(stat_valid >= threshold)
        s_star = np.mean(np.sum(null_matrix_valid >= threshold, axis=1))
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
        pi0 = PvalueCorrection.estimate_pi0_null(
            stat_valid=stat_valid, null_matrix_valid=null_matrix_valid, percentile=PI0_NULL_PERCENTILE
        )

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
