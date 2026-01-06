import numpy as np

from msmu._statistics._permutation import PermutationTest
from msmu._statistics._statistics import simple_test


def test_permutation_test_empirical_q_values():
    ctrl = np.array([[1.0, 2.0], [2.0, 3.0]])
    expr = np.array([[1.5, 2.5], [2.5, 3.5]])
    perm = PermutationTest(ctrl_arr=ctrl, expr_arr=expr, n_resamples=2, _force_resample=True, fdr="empirical")
    res = perm.run(n_permutations=2, stat_method="welch", n_jobs=1)
    assert res.q_value.shape == res.p_value.shape


def test_permutation_test_bh_q_values():
    ctrl = np.array([[1.0, 2.0], [2.0, 3.0]])
    expr = np.array([[1.5, 2.5], [2.5, 3.5]])
    perm = PermutationTest(ctrl_arr=ctrl, expr_arr=expr, n_resamples=2, _force_resample=True, fdr="bh")
    res = perm.run(n_permutations=2, stat_method="welch", n_jobs=1)
    assert res.q_value.shape == res.p_value.shape


def test_simple_test_storey_unsupported():
    ctrl = np.array([[1.0, 2.0], [2.0, 3.0]])
    expr = np.array([[2.0, 3.0], [3.0, 4.0]])
    try:
        simple_test(ctrl, expr, stat_method="welch", fdr="storey")
    except AttributeError as exc:
        assert "correct_pvalues" in str(exc)
    else:
        raise AssertionError("Expected AttributeError for unsupported storey path")
