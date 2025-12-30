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
