import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from numpy import floating
from scipy.stats import percentileofscore
from tqdm import tqdm

from .StatTest import NullDistribution, StatResult, StatTest, PvalueCorrection


@dataclass
class PermutationTestResult:
    """
    Data class to hold the results of a permutation test.
    Attributes:
    - permutation_method: Method used for permutation (exact or randomised).
    - statistic: The statistic used for the test (e.g., t-test, wilcoxon, median_diff).
    - features: Array of feature names (e.g., proteins, ptm sites).
    - median_ctrl: Median values for control group.
    - median_expr: Median values for experimental group.
    - pct_ctrl: Percentage of non-missing values in control group.
    - pct_expr: Percentage of non-missing values in experimental group.
    - log2fc: Log2 fold change values.
    - p_value: P-values for the permutation test.
    - fc_pct_1: log2fc cutoff at 1%.
    - fc_pct_5: log2fc cutoff at 5%.
    """

    permutation_method: str
    statistic: str
    ctrl: str | None
    expr: str | None
    features: np.ndarray
    median_ctrl: np.ndarray
    median_expr: np.ndarray
    pct_ctrl: np.ndarray
    pct_expr: np.ndarray
    log2fc: np.ndarray
    p_value: np.ndarray
    q_value: np.ndarray
    fc_pct_1: float | None
    fc_pct_5: float | None

    def to_df(self) -> pd.DataFrame:
        contents: dict = {
            "features": self.features,
            "median_ctrl": self.median_ctrl,
            "median_expr": self.median_expr,
            "pct_ctrl": self.pct_ctrl,
            "pct_expr": self.pct_expr,
            "log2fc": self.log2fc,
            "p_value": self.p_value,
            "q_value": self.q_value,
        }

        df: pd.DataFrame = pd.DataFrame(contents)

        return df


class PermutationTest:
    """
    Class to perform permutation tests on two groups of data (control and experimental).
    Parameters
    ----------
    ctrl_arr : np.ndarray
        Array of control group data (n_features x n_samples_ctrl).
    expr_arr : np.ndarray
        Array of experimental group data (n_features x n_samples_expr).
    n_resamples : int
        Number of resamples for the permutation test.
    force_resample : bool
        If True, forces resampling even if the number of resamples exceeds the number of combinations.

    Attributes
    ----------
    ctrl_arr : np.ndarray
        Control group data.
    expr_arr : np.ndarray
        Experimental group data.
    possible_combination_count : int
        Total number of possible combinations of control and experimental samples.
    permutation_method : str
        Method used for permutation (exact or randomised).
    n_resamples : int
        Number of resamples for the permutation test.
    force_resample : bool
        If True, forces resampling even if the number of resamples exceeds the number of combinations.
    """

    def __init__(
        self,
        ctrl_arr: np.ndarray,
        expr_arr: np.ndarray,
        n_resamples: int,
        force_resample: bool,
        fdr: bool | str
    ):
        self._ctrl_arr: np.ndarray = ctrl_arr
        self._expr_arr: np.ndarray = expr_arr
        self._possible_combination_count: int = self._get_number_of_combinations()
        self._n_resamples: int = n_resamples
        self._force_resample: bool = force_resample
        self._permutation_method: str = self._get_permutation_method()
        self.fdr: bool | str = fdr

    def _get_permutation_method(self) -> str:
        if self._n_resamples == -np.inf:
            permutation_method = "exact"
        elif self._n_resamples == self._possible_combination_count:
            permutation_method = "exact"
        elif (
            self._n_resamples > self._possible_combination_count
        ) and not self._force_resample:
            permutation_method = "exact"
        elif (
            self._n_resamples > self._possible_combination_count
        ) and self._force_resample:
            permutation_method = "randomised"
        else:
            permutation_method = "randomised"

        return permutation_method

    def _get_combinations(self) -> list:
        total_sample_num = len(self.ctrl_arr) + len(self.expr_arr)

        return list(combinations(range(total_sample_num), len(self.ctrl_arr)))

    def _get_number_of_combinations(self) -> int:
        total_sample_num = len(self.ctrl_arr) + len(self.expr_arr)
        combination_count = math.comb(total_sample_num, len(self.ctrl_arr))

        return combination_count

    def _get_iterations(self, method: str, n_resamples: int) -> list:
        if method == "exact":
            return self._get_combinations()
        elif method == "randomised":
            return [
                np.random.permutation(range(len(self.ctrl_arr) + len(self.expr_arr)))
                for _ in range(n_resamples)
            ]

    def _get_fc_percentile(self, obs_med_diff, null_med_diff) -> np.ndarray:
        return percentileofscore(
            null_med_diff, obs_med_diff, kind="rank", nan_policy="omit"
        )

    def _calc_two_sided_p_value(self, stat_obs, stat_perm):
        return np.mean(np.abs(stat_perm) >= np.abs(stat_obs), axis=0)

    def _get_pct_expression(self, arr: np.ndarray) -> np.ndarray:
        pct_expr = np.sum(~np.isnan(arr), axis=0) / arr.shape[0] * 100

        return pct_expr

    def _perm_test(
        self,
        concated_arr: np.ndarray,
        iterations: list,
        statistic: str,
        n_jobs: int,
    ) -> PermutationTestResult:

        perm_test_res: PermutationTestResult = PermutationTestResult(
            permutation_method=self.permutation_method,
            statistic=statistic,
            ctrl=None,
            expr=None,
            features=np.array([]),
            median_ctrl=np.nanmedian(self.ctrl_arr, axis=0),
            median_expr=np.nanmedian(self.expr_arr, axis=0),
            pct_ctrl=self._get_pct_expression(self.ctrl_arr),
            pct_expr=self._get_pct_expression(self.expr_arr),
            log2fc=np.array([]),
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

        obs_stats: StatResult = StatTest._stat_tests(
            ctrl=self.ctrl_arr,
            expr=self.expr_arr,
            statistic=statistic,
        )
        # print(obs_stats.statistic)
        obs_log2fc: StatResult = StatTest._stat_tests(
            ctrl=self.ctrl_arr,
            expr=self.expr_arr,
            statistic="med_diff",
        )

        # Initialize NullDistribution objects for the statistic and log2fc and q values
        stat_null_dist = NullDistribution(
            method=statistic, null_distribution=np.array([])
        )
        log2fc_null_dist = NullDistribution(
            method="med_diff", null_distribution=np.array([])
        )

        # Iterate over the combinations or randomised permutations
        for combn in tqdm_iter:
            # Calculate the statistic for the current permutation
            tmp_stat: StatResult = self._sub_perm(
                concated_arr=concated_arr,
                combinations=combn,
                statistic=statistic,
            )

            # Add the result to the null distribution
            stat_null_dist = stat_null_dist.add_permutation_result(tmp_stat)

            # Calculate the log2 fold change for the current permutation
            tmp_log2fc: StatResult = self._sub_perm(
                concated_arr=concated_arr,
                combinations=combn,
                statistic="med_diff",
            )
            # Add the result to the log2fc null distribution
            log2fc_null_dist = log2fc_null_dist.add_permutation_result(tmp_log2fc)

        pval_permutation = StatTest.calc_permutation_pvalue(
            stat_obs=obs_stats.statistic,
            null_dist=stat_null_dist.null_distribution
        )

        if self.fdr == 'empirical':
            q_vals = PvalueCorrection.empirical(
                stat_obs=obs_stats.statistic, 
                null_dist=stat_null_dist.null_distribution, 
                pvals=pval_permutation,
                )
        elif self.fdr == 'bh':
            q_vals = PvalueCorrection.bh(
                pvals=pval_permutation
            )

        # put results to PermutationTestResult
        perm_test_res.log2fc = obs_log2fc.statistic
        perm_test_res.p_value = pval_permutation
        perm_test_res.q_value = q_vals

        # Calculate the fold change percentile
        fc_pct_criteria = [1, 5]  # 1% and 5% thresholds
        perm_test_res.fc_pct_1, perm_test_res.fc_pct_5 = [
            self._get_fc_threshold(log2fc_null_dist.null_distribution, x)
            for x in fc_pct_criteria
        ]

        return perm_test_res

    @staticmethod
    def _get_fc_threshold(null_med_diff: np.ndarray, percentile: int) -> float:
        low_quantile: floating = np.nanpercentile(null_med_diff, percentile)
        high_quantile: floating = np.nanpercentile(null_med_diff, 100 - percentile)

        fc_cutoff: float = float(np.mean([abs(low_quantile), abs(high_quantile)]))
        rounded_cutoff: float = round(fc_cutoff, 2)

        return rounded_cutoff

    def _sub_perm(
        self, concated_arr: np.ndarray, combinations: np.ndarray, statistic: str
    ) -> StatResult:
        if self.permutation_method == "exact":
            total_index: np.ndarray = np.arange(len(self.ctrl_arr) + len(self.expr_arr))
            ctrl_idx = list(combinations)
            expr_idx: np.ndarray = np.delete(total_index, ctrl_idx)
        else:  # randomised
            total_index = combinations
            ctrl_idx = total_index[: len(self.ctrl_arr)]
            expr_idx: np.ndarray = total_index[len(self.ctrl_arr) :]

        perm_ctrl: np.ndarray = concated_arr[ctrl_idx, :]
        perm_expr: np.ndarray = concated_arr[expr_idx, :]

        stat_res: StatResult = StatTest._stat_tests(
            ctrl=perm_ctrl, expr=perm_expr, statistic=statistic
        )

        return stat_res

    def run(
        self,
        n_permutations: int,
        statistic: str,
        n_jobs: int,
    ) -> PermutationTestResult:

        concated_arr: np.ndarray = np.concatenate(
            (self.ctrl_arr, self.expr_arr), axis=0
        )

        iterations: list = self._get_iterations(
            method=self.permutation_method,
            n_resamples=n_permutations,
        )

        perm_test_res: PermutationTestResult = self._perm_test(
            concated_arr=concated_arr,
            iterations=iterations,
            statistic=statistic,
            n_jobs=n_jobs,
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
    def permutation_method(self):
        return self._permutation_method

    @permutation_method.setter
    def permutation_method(self, method: str):
        self._permutation_method = method


class Limma: ...
