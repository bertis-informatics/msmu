import numpy as np
from tqdm import tqdm

from ._statistics import (
    NullDistribution,
    StatResult,
    HypothesisTesting,
    calc_permutation_pvalue,
)
from ._multiple_test_correction import PvalueCorrection, PI0_LOWER_BOUND
from ._de_base import PermTestResult
from ._permutation_core import (
    count_combinations,
    resolve_method,
    make_iterations,
    split,
    permuted_log2fc,
    fc_threshold_from_null,
)
from ..logging_utils import get_logger

logger = get_logger(__name__)

# q threshold that reported hits are conventionally screened at; used to decide whether a
# design's floor makes significance unreachable and therefore worth warning about.
_CONVENTIONAL_Q_CUTOFF = 0.05


def min_achievable_q(n_ctrl: int, n_expr: int, n_permutations: int, fdr: str) -> float:
    """
    Smallest q-value a permutation design can produce, whatever the data says.

    The observed labelling is itself one of the permutations the null is built from, so every
    observed feature at ``|stat| >= s`` contributes its own statistic back into the null pool.
    A balanced design contributes it twice, because the complement of the observed split is
    another split of the same sizes and is therefore also enumerated. That forces
    ``fp >= self_inclusion * tp`` and leaves a floor under q that no effect size can cross.

    For fdr="empirical" the estimator is ``pi0 * (fp+1)/(B+1) / (tp+1)``, which is increasing
    in tp, so the floor is taken at tp=1. For fdr="bh" the pooled p-value is
    ``(fp+1)/(B*m+1)`` and BH multiplies by m/k, leaving ``self_inclusion/B`` in the limit.

    Parameters:
        n_ctrl: Number of control samples.
        n_expr: Number of experimental samples.
        n_permutations: Number of permutations the null is actually built from.
        fdr: Multiple testing correction in use ("empirical" or "bh").

    Returns:
        The minimum q-value attainable for this design.
    """
    self_inclusion = 2 if n_ctrl == n_expr else 1

    if fdr == "bh":
        return self_inclusion / n_permutations

    return PI0_LOWER_BOUND * (self_inclusion + 1) / ((n_permutations + 1) * 2)


class PermutationTest:
    """
    Class to perform permutation tests on two groups of data (control and experimental).

    Parameters:
        ctrl_arr: Array of control group data (n_features x n_samples_ctrl).
        expr_arr: Array of experimental group data (n_features x n_samples_expr).
        n_resamples: Number of resamples for the permutation test.
        _force_resample: If True, forces resampling even if the number of resamples exceeds the number of combinations.

    Attributes
        ctrl_arr: Array of control group data (n_features x n_samples_ctrl).
        expr_arr: Array of experimental group data (n_features x n_samples_expr).
        possible_combination_count: Total number of possible combinations of control and experimental samples.
        permutation_method: Method used for permutation (exact or randomised).
        n_resamples: Number of resamples for the permutation test.
        _force_resample: If True, forces resampling even if the number of resamples exceeds the number of combinations.
    """

    def __init__(
        self,
        ctrl_arr: np.ndarray,
        expr_arr: np.ndarray,
        n_resamples: int,
        _force_resample: bool,
        fdr: bool | str,
    ):
        self._ctrl_arr: np.ndarray = ctrl_arr
        self._expr_arr: np.ndarray = expr_arr

        self._possible_combination_count: int = count_combinations(len(ctrl_arr), len(expr_arr))
        self._n_resamples: int = n_resamples
        self._force_resample: bool = _force_resample
        self._permutation_method: str = resolve_method(len(ctrl_arr), len(expr_arr), n_resamples, _force_resample)
        self.fdr: bool | str = fdr

        self._warn_if_resamples_ignored()
        self._warn_if_significance_unreachable()

    def _warn_if_resamples_ignored(self) -> None:
        if self._permutation_method != "exact" or self._n_resamples <= self._possible_combination_count:
            return

        logger.warning(
            "n_resamples=%s exceeds the %s distinct %s vs %s splits that exist, so every split is "
            "enumerated instead and the null is built from %s permutations. Drawing more would "
            "only resample the same splits.",
            self._n_resamples,
            self._possible_combination_count,
            len(self.ctrl_arr),
            len(self.expr_arr),
            self._possible_combination_count,
        )

    def _warn_if_significance_unreachable(self) -> None:
        if self.fdr not in ("empirical", "bh") or self.n_permutations_used < 1:
            return

        floor = min_achievable_q(
            n_ctrl=len(self.ctrl_arr),
            n_expr=len(self.expr_arr),
            n_permutations=self.n_permutations_used,
            fdr=self.fdr,
        )
        if floor < _CONVENTIONAL_Q_CUTOFF:
            return

        logger.warning(
            "%s vs %s with fdr='%s' cannot produce q < %s: the null is built from only %s "
            "permutations and the observed labelling is one of them, which floors q at %.3f "
            "no matter how strong the effect. Use stat_method='limma' for this design.",
            len(self.ctrl_arr),
            len(self.expr_arr),
            self.fdr,
            _CONVENTIONAL_Q_CUTOFF,
            self.n_permutations_used,
            floor,
        )

    def _perm_test(
        self,
        concated_arr: np.ndarray,
        iterations: list,
        stat_method: str,
        measure: str,
        log_transformed: bool,
    ) -> PermTestResult:

        perm_test_res: PermTestResult = PermTestResult(
            permutation_method=self.permutation_method,
            n_permutations=len(iterations),
            stat_method=stat_method,
            p_value=np.array([]),
            q_value=np.array([]),
            fc_pct_1=None,
            fc_pct_5=None,
        )

        tqdm_iter = tqdm(
            iterations,
            desc="Running Permutations",
            position=0,
            leave=True,
        )

        obs_stats: StatResult = HypothesisTesting.test(
            ctrl=self.ctrl_arr,
            expr=self.expr_arr,
            stat_method=stat_method,
        )

        # Initialize NullDistribution objects for the statistic and log2fc and q values
        stat_null_dist = NullDistribution(stat_method=stat_method, null_distribution=np.array([]))
        log2fc_null_dist = NullDistribution(stat_method=measure, null_distribution=np.array([]))

        # Iterate over the combinations or randomised permutations
        for combn in tqdm_iter:
            # Calculate the statistic for the current permutation
            tmp_stat: StatResult = self._calc_permuted_stats(
                concated_arr=concated_arr,
                combinations=combn,
                stat_method=stat_method,
            )

            # Add the result to the null distribution
            stat_null_dist = stat_null_dist.add_permutation_result(tmp_stat)

            # Calculate the log2 fold change for the current permutation
            tmp_log2fc: StatResult = self._calc_permuted_log2fc(
                concated_arr=concated_arr,
                combinations=combn,
                measure=measure,
                log_transformed=log_transformed,
            )
            # Add the result to the log2fc null distribution
            log2fc_null_dist = log2fc_null_dist.add_permutation_result(tmp_log2fc)

        pval_permutation = calc_permutation_pvalue(
            stat_obs=obs_stats.statistic, null_dist=stat_null_dist.null_distribution
        )

        if self.fdr == "empirical":
            q_vals = PvalueCorrection.empirical(
                stat_obs=obs_stats.statistic,
                null_dist=stat_null_dist.null_distribution,
            )
        elif self.fdr == "bh":
            q_vals = PvalueCorrection.bh(pvals=pval_permutation)

        # put results to PermutationTestResult
        perm_test_res.p_value = pval_permutation
        perm_test_res.q_value = q_vals
        # observed parametric statistic (the value ranked against the empirical null)
        perm_test_res.statistic = obs_stats.statistic

        # Calculate the fold change percentile
        fc_pct_criteria = [1, 5]  # 1% and 5% thresholds
        perm_test_res.fc_pct_1, perm_test_res.fc_pct_5 = [
            fc_threshold_from_null(log2fc_null_dist.null_distribution, x) for x in fc_pct_criteria
        ]

        return perm_test_res

    def _calc_permuted_stats(self, concated_arr: np.ndarray, combinations: np.ndarray, stat_method: str) -> StatResult:
        perm_ctrl, perm_expr = split(concated_arr, combinations, self.permutation_method, len(self.ctrl_arr))
        return HypothesisTesting.test(ctrl=perm_ctrl, expr=perm_expr, stat_method=stat_method)

    def _calc_permuted_log2fc(
        self,
        concated_arr: np.ndarray,
        combinations: np.ndarray,
        measure: str,
        log_transformed: bool,
    ) -> StatResult:
        log2fc: np.ndarray = permuted_log2fc(
            concated_arr, combinations, self.permutation_method, len(self.ctrl_arr), measure, log_transformed
        )
        return StatResult(stat_method=None, statistic=log2fc, p_value=None)

    def run(
        self,
        n_permutations: int,
        stat_method: str,
        measure: str,
        log_transformed: bool,
    ) -> PermTestResult:

        concated_arr: np.ndarray = np.concatenate((self.ctrl_arr, self.expr_arr), axis=0)

        iterations: list = make_iterations(
            len(self.ctrl_arr), len(self.expr_arr), self.permutation_method, n_permutations
        )

        perm_test_res: PermTestResult = self._perm_test(
            concated_arr=concated_arr,
            iterations=iterations,
            stat_method=stat_method,
            measure=measure,
            log_transformed=log_transformed,
        )

        return perm_test_res

    @property
    def ctrl_arr(self):
        return self._ctrl_arr

    @property
    def expr_arr(self):
        return self._expr_arr

    @property
    def possible_combination_count(self):
        return self._possible_combination_count

    @property
    def n_permutations_used(self) -> int:
        """Number of permutations the null is actually built from ('exact' enumerates every split)."""
        if self._permutation_method == "exact":
            return self._possible_combination_count
        return self._n_resamples

    @property
    def permutation_method(self):
        return self._permutation_method

    @permutation_method.setter
    def permutation_method(self, method: str):
        self._permutation_method = method
