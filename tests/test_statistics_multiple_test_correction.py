import numpy as np
import pytest

from msmu._statistics._multiple_test_correction import P_ADJUST_METHODS, PvalueCorrection


def test_bh():
    pvals = np.array([0.1, 0.4, 0.3, 0.02])
    qvals = PvalueCorrection.adjust(pvals, "bh")
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


# --- BID-114: PvalueCorrection.adjust — R p.adjust family parity ---------------------------------
# The exposed methods must match R's stats::p.adjust exactly (the driver is "if R limma offers it,
# we do too"). R's algorithms are transcribed straight from the R source below and used as the
# reference; every method is checked against it, so the statsmodels mapping is pinned to R, not to
# statsmodels' own naming.


def _r_padjust(p, method):
    """Closed-form transcription of R stats::p.adjust for the non-hommel methods."""
    p = np.asarray(p, dtype=float)
    n = p.size
    if method == "bonferroni":
        return np.minimum(1.0, n * p)
    if method == "holm":
        o = np.argsort(p)  # ascending
        ro = np.argsort(o)
        return np.minimum(1.0, np.maximum.accumulate((n - np.arange(n)) * p[o]))[ro]
    if method == "hochberg":
        o = np.argsort(-p)  # descending
        ro = np.argsort(o)
        return np.minimum(1.0, np.minimum.accumulate(np.arange(1, n + 1) * p[o]))[ro]
    if method in ("bh", "by"):
        o = np.argsort(-p)  # descending
        ro = np.argsort(o)
        i = np.arange(n, 0, -1)
        cm = np.sum(1.0 / np.arange(1, n + 1)) if method == "by" else 1.0
        return np.minimum(1.0, np.minimum.accumulate(cm * n / i * p[o]))[ro]
    raise ValueError(method)


def _r_hommel(p):
    """Faithful transcription of R stats::p.adjust(p, 'hommel')."""
    p = np.asarray(p, dtype=float)
    n = p.size
    if n <= 1:
        return p.copy()
    o = np.argsort(p, kind="stable")  # ascending
    ps = p[o]
    ro = np.argsort(o, kind="stable")
    q = np.full(n, np.min(n * ps / np.arange(1, n + 1)))
    pa = q.copy()
    for m in range(n - 1, 1, -1):
        n_i1 = n - m + 1
        i2 = np.arange(n_i1, n)
        q1 = np.min(m * ps[i2] / np.arange(2, m + 1))
        q[:n_i1] = np.minimum(m * ps[:n_i1], q1)
        q[i2] = q[n_i1 - 1]
        pa = np.maximum(pa, q)
    return np.maximum(pa, ps)[ro]


def _r_reference(p, method):
    return _r_hommel(p) if method == "hommel" else _r_padjust(p, method)


@pytest.mark.parametrize("method", P_ADJUST_METHODS)
@pytest.mark.parametrize(
    "pvals",
    [
        np.array([0.001, 0.008, 0.039, 0.041, 0.2, 0.6, 0.75, 0.9]),
        np.array([0.04, 0.04, 0.04, 0.5]),  # ties
        np.array([0.2]),  # single value
        np.random.default_rng(0).random(50) ** 2,  # skewed toward small p
    ],
)
def test_adjust_matches_r_padjust(method, pvals):
    """adjust() reproduces R's p.adjust for every exposed method."""
    got = PvalueCorrection.adjust(pvals, method)
    want = _r_reference(pvals, method)
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_adjust_is_case_insensitive():
    pvals = np.array([0.01, 0.2, 0.5, 0.9])
    for lower, upper in [("bh", "BH"), ("by", "BY"), ("holm", "Holm"), ("bonferroni", "BONFERRONI")]:
        np.testing.assert_array_equal(
            PvalueCorrection.adjust(pvals, lower), PvalueCorrection.adjust(pvals, upper)
        )


def test_adjust_preserves_nan_positions():
    pvals = np.array([0.01, np.nan, 0.5, np.nan, 0.9])
    q = PvalueCorrection.adjust(pvals, "bonferroni")
    assert np.isnan(q[[1, 3]]).all()
    assert not np.isnan(q[[0, 2, 4]]).any()
    # correction uses only the 3 non-NaN p-values: bonferroni multiplies by 3
    np.testing.assert_allclose(q[[0, 2, 4]], np.minimum(1.0, 3 * pvals[[0, 2, 4]]))


def test_adjust_rejects_unknown_method_and_none():
    # "none" is deliberately not an adjustment method (the uncorrected values are the p-values).
    for bad in ["none", "None", "fdr", "storey", "sidak", ""]:
        with pytest.raises(ValueError, match="Unknown p-value adjustment method"):
            PvalueCorrection.adjust(np.array([0.1, 0.2]), bad)
