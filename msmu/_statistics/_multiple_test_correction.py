import numpy as np
from statsmodels.stats.multitest import multipletests


class PvalueCorrection:
    """
    Class for multiple testing correction methods.

    Methods:
        bh : Benjamini-Hochberg FDR correction.
        storey : Storey's q-value estimation with pi0 estimation.
        empirical : Permutation-based empirical FDR estimation.
    """

    @staticmethod
    def bh(pvals: np.ndarray) -> np.ndarray:
        """
        Benjamini-Hochberg FDR correction with NaN handling.

        Parameters:
            pvals: Array of p-values (can include NaN).

        Returns:
            qvals: Array of q-values (NaN-filled where p was NaN).
        """
        pvals = np.asarray(pvals)
        qvals = np.full_like(pvals, np.nan, dtype=float)
        mask = ~np.isnan(pvals)
        if np.any(mask):
            _, qvals_nonan, _, _ = multipletests(pvals[mask], method="fdr_bh")
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
        pi0 = PvalueCorrection.estimate_pi0_null(
            stat_valid=stat_valid, null_matrix_valid=null_matrix_valid, percentile=95
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
