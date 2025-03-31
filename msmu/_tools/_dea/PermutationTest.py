from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from tqdm import tqdm

from .StatTest import NullDistribution, StatResult, StatTest


@dataclass
class PermutationTestResult:
    method: str
    features: np.ndarray
    ctrl_median: np.ndarray
    expr_median: np.ndarray
    log2fc: np.ndarray
    # fc_pct: np.ndarray

    def to_df(self) -> pd.DataFrame:
        contents: dict = {
            "features": self.features,
            "ctrl_median": self.ctrl_median,
            "expr_median": self.expr_median,
            "log2fc": self.log2fc,
            # "fc_pct": self.fc_pct
        }
        for key in self.method:
            contents[f"p_perm_{key}"] = getattr(self, f"p_perm_{key}")

        df: pd.DataFrame = pd.DataFrame(contents)

        return df

    def fc_threshold(self, threshold: float):
        low_quantile = np.min(self.fc_pct > threshold)
        print(low_quantile)
        high_quantile = np.max(self.fc_pct < (100 - threshold))
        print(high_quantile)

        fc_cutoff = np.mean([abs(low_quantile), abs(high_quantile)])

        return fc_cutoff


class PermutationTest:
    def __init__(self, ctrl: str, expr: str):
        self._ctrl: str = ctrl
        self._expr: str = expr
        self._possible_combinations: list = self._get_combinations()
        self._permutation_method: str | None = None

    def _get_combinations(self) -> list:
        total_sample_num = len(self.ctrl) + len(self.expr)

        return list(combinations(range(total_sample_num), len(self.ctrl)))

    def _get_iterations(self, method: str, n_resamples: int) -> list:
        if method == "exact":
            return self._possible_combinations
        elif method == "randomised":
            return [
                np.random.permutation(range(len(self.ctrl) + len(self.expr)))
                for _ in range(n_resamples)
            ]

    def _get_fc_percentile(self, obs_med_diff, null_med_diff) -> np.array:
        return percentileofscore(
            null_med_diff, obs_med_diff, kind="rank", nan_policy="omit"
        )

    def _calc_two_sided_p_value(self, stat_obs, stat_perm):
        return np.mean(np.abs(stat_perm) >= np.abs(stat_obs), axis=0)

    def _perm_test(
        self, concated_arr: np.ndarray, iterations: list, statistic: list, n_jobs: int
    ) -> PermutationTestResult:

        if "med_diff" in statistic:
            stat_to_run = statistic
        else:
            stat_to_run = statistic + ["med_diff"]

        perm_test_res: PermutationTestResult = PermutationTestResult(
            method=statistic,
            features=np.array([]),
            ctrl_median=np.nanmedian(self.ctrl, axis=0),
            expr_median=np.nanmedian(self.expr, axis=0),
            log2fc=np.array([]),
            # fc_pct=np.array([])
        )

        tqdm_stat = tqdm(stat_to_run, desc="Running Statistics", position=0)
        tqdm_iter = tqdm(
            iterations, desc="Running Permutations", position=1, leave=False
        )
        for stat_method in tqdm_stat:
            obs_stats: StatResult = StatTest._stat_tests(
                ctrl=self.ctrl, expr=self.expr, statistic=stat_method
            )

            null_dist = NullDistribution(
                method=stat_method, null_distribution=np.array([])
            )
            for combinations in tqdm_iter:
                tmp_stat: StatResult = self._sub_perm(
                    concated_arr=concated_arr,
                    combinations=combinations,
                    statistic=[stat_method],
                )
                null_dist = null_dist.add_permutation_result(tmp_stat)

            # pval_permutation = StatTest.pval2tail(stat_obs=obs_stats.statistic, null_dist=null_dist.null_distribution)
            pval_permutation = StatTest.pval_calc_test(
                stat_obs=obs_stats.statistic, null_dist=null_dist.null_distribution
            )
            setattr(perm_test_res, f"p_perm_{stat_method}", pval_permutation)

            if stat_method == "med_diff":
                perm_test_res.log2fc = obs_stats.statistic
                # perm_test_res.fc_pct = self._get_fc_percentile(
                #     obs_med_diff=obs_stats.statistic, null_med_diff=null_dist.null_distribution
                # )

        return perm_test_res

    def _sub_perm(
        self, concated_arr: np.ndarray, combinations: np.array, statistic: list
    ) -> StatResult:
        if self.permutation_method == "exact":
            total_index: np.array = np.arange(len(self.ctrl) + len(self.expr))
            ctrl_idx: np.array = list(combinations)
            expr_idx: np.array = np.delete(total_index, ctrl_idx)
        else:  # randomised
            total_index = combinations
            ctrl_idx: np.array = total_index[: len(self.ctrl)]
            expr_idx: np.array = total_index[len(self.ctrl) :]

        perm_ctrl: np.ndarray = concated_arr[ctrl_idx, :]
        perm_expr: np.ndarray = concated_arr[expr_idx, :]

        for stat_method in statistic:
            stat_res = StatTest._stat_tests(
                ctrl=perm_ctrl, expr=perm_expr, statistic=stat_method
            )

        return stat_res

    def run(self, n_permutations: int, n_jobs: int, statistic: str):
        concated_arr: np.array = np.concatenate((self.ctrl, self.expr), axis=0)

        iterations: list = self._get_iterations(self.permutation_method, n_permutations)
        perm_test_res: PermutationTestResult = self._perm_test(
            concated_arr=concated_arr,
            iterations=iterations,
            statistic=statistic,
            n_jobs=n_jobs,
        )

        return perm_test_res

    @property
    def ctrl(self):
        return self._ctrl

    @property
    def expr(self):
        return self._expr

    @property
    def possible_combinations(self):
        return self._possible_combinations

    @property
    def permutation_method(self):
        return self._permutation_method

    @permutation_method.setter
    def permutation_method(self, method: str):
        self._permutation_method = method


class Limma: ...
