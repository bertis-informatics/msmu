import numpy as np

from msmu._statistics._multiple_test_correction import PvalueCorrection


def test_bh():
    pvals = np.array([0.1, 0.4, 0.3, 0.02])
    qvals = PvalueCorrection.bh(pvals)
    # BH steps: sort p=[0.02,0.1,0.3,0.4], compute q_i = p_i * m / i
    # q_raw=[0.08,0.2,0.4,0.4], then apply monotonic correction and map back.
    expected = np.array([0.2, 0.4, 0.4, 0.08])
    assert np.allclose(qvals, expected)


def test_storey():
    pvals = np.array([0.1, 0.2, 0.6, 0.8])
    qvals = PvalueCorrection.storey(pvals)
    # Storey steps (lambda=0.5): pi0 = #(p>0.5)/((1-0.5)*m) = 2/(0.5*4)=1
    # With pi0=1, q-values reduce to BH-like: sorted p=[0.1,0.2,0.6,0.8]
    # q_raw=[0.4,0.4,0.8,0.8], monotonic holds, map back to original order.
    expected = np.array([0.4, 0.4, 0.8, 0.8])
    assert np.allclose(qvals, expected)


def test_estimate_pi0_null_bounds():
    stat = np.array([1.0, 2.0, 3.0, 4.0])
    null = np.tile(stat, (5, 1))
    pi0 = PvalueCorrection.estimate_pi0_null(stat_valid=stat, null_matrix_valid=null)
    assert 0 <= pi0 <= 1


def test_estimate_pi0_null_counts_features_per_permutation():
    # s_star must be the number of *features* exceeding the threshold per permutation, not the
    # number of permutations exceeding it per feature. m=100, B=4, exactly 3 features clear the
    # 95th percentile of the observed in every permutation.
    stat = np.arange(100, dtype=float)  # 95th percentile = 94.05 -> s = 5, so 1 - s/m = 0.95
    null = np.zeros((4, 100))
    null[:, 97:] = 200.0

    pi0 = PvalueCorrection.estimate_pi0_null(stat_valid=stat, null_matrix_valid=null, percentile=95)

    # s_star = 3 features -> pi0 = 0.95 / (1 - 3/100). Summing the other axis would give
    # s_star = 4*3/100 = 0.12 and pi0 = 0.95 / (1 - 0.0012) = 0.9512.
    assert np.isclose(pi0, 0.95 / 0.97)


def test_estimate_pi0_null_is_one_under_complete_null():
    # Observed and null statistics share a distribution, so essentially every hypothesis is null
    # and pi0 must land at ~1. A pi0 that ignores the feature axis collapses onto the constant
    # 1 - percentile/100 = 0.95 instead, whatever the data.
    rng = np.random.default_rng(0)
    stat = np.abs(rng.normal(size=2000))
    null = np.abs(rng.normal(size=(20, 2000)))

    pi0 = PvalueCorrection.estimate_pi0_null(stat_valid=stat, null_matrix_valid=null, percentile=95)

    assert pi0 > 0.98


def test_empirical():
    stat = np.array([3.0, 2.0, 1.0])
    null = np.array([[0.5, 1.5, 2.5], [0.1, 2.0, 1.2]])
    qvals = PvalueCorrection.empirical(stat_obs=stat, null_dist=null)
    # Empirical steps: B=2 permutations, null_valid = [0.5,1.5,2.5,0.1,2.0,1.2].
    # pi0 from null: threshold=95th percentile=2.8, s=1, s_star=0 => pi0=2/3.
    # For s=3: tp=1, fp=0 => e_fp=1/3, e_tp=2 => q=1/9.
    # For s=2: tp=2, fp=2 => e_fp=1, e_tp=3 => q=2/9.
    # For s=1: tp=3, fp=4 => e_fp=5/3, e_tp=4 => q=5/18.
    # Monotonic correction keeps q in descending stat order.
    expected = np.array([1 / 9, 2 / 9, 5 / 18])
    assert np.allclose(qvals, expected)
