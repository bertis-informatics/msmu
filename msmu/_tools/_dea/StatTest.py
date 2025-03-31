from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import ranksums, ttest_ind
from statsmodels.distributions.empirical_distribution import ECDF


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
        mean_ctrl = np.nanmean(ctrl, axis=0)
        mean_expr = np.nanmean(expr, axis=0)
        var_ctrl = np.nanvar(ctrl, axis=0)
        var_expr = np.nanvar(expr, axis=0)

        t_val = (mean_expr - mean_ctrl) / np.sqrt(
            var_expr / len(ctrl) + var_ctrl / len(expr)
        )
        pval = None
        # t_val, pval = ttest_ind(ctrl, expr, axis=0, equal_var=False, nan_policy="omit")

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
    def pval2tail(stat_obs, null_dist):
        # p-value computation routine
        # s0: null statistic, s: observed statistic
        # p-value : computed by two-tail test

        # stacked_s0 = np.hstack(null_dist)
        stacked_s0 = null_dist
        ecdf_res = ECDF(stacked_s0)
        null_dist = ecdf_res.x[1:]
        f0 = ecdf_res.y[1:]

        f = interp1d(null_dist, f0, bounds_error=False)
        p = f(stat_obs)
        p[np.isnan(p)] = 0
        p = 2 * np.min(np.c_[p, 1 - p], axis=1)

        if np.sum(p != 0) != 0:
            p[p == 0] = np.nanmin(p[p != 0]) / 2
        else:
            p[p == 0] = 1e-10
        p[p == 1] = (1 - np.max(p[p != 1])) / 2 + np.max(p[p != 1])

        return p

    @staticmethod
    def pval_calc_test(stat_obs, null_dist):
        pval = np.zeros_like(stat_obs)
        for i in range(len(stat_obs)):
            pval[i] = np.sum(np.abs(null_dist) >= np.abs(stat_obs[i])) / len(null_dist)
        return pval


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
