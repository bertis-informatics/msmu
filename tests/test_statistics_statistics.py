import numpy as np
from scipy.stats import t

from msmu._statistics._statistics import (
    calc_permutation_pvalue,
    HypothesisTesting,
    _measure_central_tendency,
    _calc_log2fc,
)


# def test_hypothesis_testing_median_diff():
#     ctrl = np.array([[1.0, 2.0], [3.0, 4.0]])
#     expr = np.array([[2.0, 2.0], [4.0, 6.0]])
#     diff, _ = HypothesisTesting.median_diff(ctrl, expr)
#     # Median(expr) - Median(ctrl) = [3-2, 4-3] = [1, 1].
#     assert np.allclose(diff, np.array([1.0, 1.0]))


def test_hypothesis_testing_welch():
    ctrl = np.array([[1.0, 2.0], [3.0, 4.0]])
    expr = np.array([[5.0, 6.0], [7.0, 8.0]])
    stat, pval = HypothesisTesting.welch(ctrl, expr)
    # Means: [2,3] vs [6,7], variances: [2,2] vs [2,2], n=2 each => denom=sqrt(2/2+2/2)=sqrt(2).
    # t = (4)/sqrt(2) = 2*sqrt(2) for each feature, df=2, p=2*sf(|t|).
    expected_stat = np.array([2 * np.sqrt(2), 2 * np.sqrt(2)])
    expected_pval = 2 * t.sf(np.abs(expected_stat), 2)
    assert np.allclose(stat, expected_stat)
    assert np.allclose(pval, expected_pval)


def test_calc_permutation_pvalue():
    stat_obs = np.array([1.0, np.nan, 2.0])
    null = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    pvals = calc_permutation_pvalue(stat_obs, null)
    # pooled_null has 6 zeros; exceedances for |stat| are 0 so p=(0+1)/(6+1)=1/7.
    assert np.isnan(pvals[1])
    expected = np.array([1 / 7, np.nan, 1 / 7])
    assert np.allclose(pvals[~np.isnan(pvals)], expected[~np.isnan(expected)])


def test_measure_central_tendency_mean():
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [8.0, 6.0]])
    mean = _measure_central_tendency(arr, method="mean")
    expected = np.array([4.0, 4.0])
    assert np.allclose(mean, expected)


def test_measure_central_tendency_median():
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [8.0, 6.0]])
    median = _measure_central_tendency(arr, method="median")
    expected = np.array([3.0, 4.0])
    assert np.allclose(median, expected)


def test_calc_log2fc_logtransformed():
    ctrl = np.array([2.0, 4.0, 0.0])
    expr = np.array([8.0, 2.0, 0.0])
    log2fc = _calc_log2fc(ctrl, expr, log_transformed=True)
    expected = np.array([6.0, -2.0, 0.0])
    assert np.allclose(log2fc, expected)


def test_calc_log2fc_not_logtransformed():
    ctrl = np.array([2.0, 4.0, 1.0])
    expr = np.array([8.0, 2.0, 1.0])
    log2fc = _calc_log2fc(ctrl, expr, log_transformed=False)
    expected = np.array([2.0, -1.0, 0.0])
    assert np.allclose(log2fc, expected)
