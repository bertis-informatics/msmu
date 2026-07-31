"""BID-114: p_adjust — shared multiple-testing correction across both DE engines.

limma exposes R's p.adjust family (bh/by/holm/hochberg/hommel/bonferroni) for its q-value; the
permutation engines keep the empirical FDR as their default but also accept the same family. The
``p_adjust="auto"`` default resolves to each engine's native default (limma -> bh, permutation ->
empirical), so runs that do not set it are unchanged. ``"empirical"`` is permutation-only.

The 3 vs 3 fixture is dense, so all six features are estimable and the permutation design (C(6,3)=20
splits) is enumerated exactly, making both engines deterministic here.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData

from msmu._statistics._multiple_test_correction import PvalueCorrection
from msmu._tools._dea import run_de

# features x samples, 3 vs 3 (A/B); AnnData stores samples x features.
_Y = np.array(
    [
        [10.305, 8.96, 10.75, 12.441, 9.549, 10.198],
        [10.128, 9.684, 9.983, 10.647, 12.379, 12.278],
        [10.066, 11.127, 10.468, 9.141, 10.369, 9.041],
        [10.878, 9.95, 9.815, 9.319, 11.223, 9.845],
        [9.572, 9.648, 10.532, 10.365, 10.413, 10.431],
        [12.142, 9.594, 9.488, 9.186, 10.616, 11.129],
    ]
)


@pytest.fixture
def mdata_3v3() -> MuData:
    obs = pd.DataFrame({"cond": ["A", "A", "A", "B", "B", "B"]}, index=[f"S{j}" for j in range(6)])
    adata = AnnData(X=_Y.T, obs=obs.copy(), var=pd.DataFrame(index=[f"P{i}" for i in range(6)]))
    return MuData({"protein": adata})


def _n_tested(result) -> int:
    return int(np.sum(~np.isnan(result.p_value)))


# --------------------------------------------------------------------------- limma


def test_limma_auto_resolves_to_bh(mdata_3v3):
    """The default (auto) limma q is exactly BH of the moderated p-values — unchanged from before."""
    auto = run_de(mdata_3v3, "protein", "cond", "A", "B")  # p_adjust default "auto"
    bh = run_de(mdata_3v3, "protein", "cond", "A", "B", p_adjust="bh")

    np.testing.assert_array_equal(auto.q_value, bh.q_value)
    np.testing.assert_allclose(auto.q_value, PvalueCorrection.adjust(auto.p_value, "bh"), equal_nan=True)


def test_limma_bonferroni_is_exact_and_stricter_than_bh(mdata_3v3):
    """A chosen method flows through to the q-value: bonferroni = min(1, n*p) and dominates BH."""
    bh = run_de(mdata_3v3, "protein", "cond", "A", "B", p_adjust="bh")
    bonf = run_de(mdata_3v3, "protein", "cond", "A", "B", p_adjust="bonferroni")

    n = _n_tested(bonf)
    np.testing.assert_allclose(bonf.q_value, np.minimum(1.0, n * bonf.p_value), equal_nan=True)
    tested = ~np.isnan(bh.q_value)
    assert np.all(bonf.q_value[tested] >= bh.q_value[tested] - 1e-12)


def test_limma_p_adjust_is_case_insensitive(mdata_3v3):
    lower = run_de(mdata_3v3, "protein", "cond", "A", "B", p_adjust="by")
    upper = run_de(mdata_3v3, "protein", "cond", "A", "B", p_adjust="BY")
    np.testing.assert_array_equal(lower.q_value, upper.q_value)


def test_limma_rejects_empirical(mdata_3v3):
    """empirical FDR needs a permutation null limma does not have — a clear, engine-specific error."""
    with pytest.raises(ValueError, match="has no null to build it from"):
        run_de(mdata_3v3, "protein", "cond", "A", "B", p_adjust="empirical")


def test_limma_rejects_unknown_method(mdata_3v3):
    with pytest.raises(ValueError, match="Unknown p_adjust"):
        run_de(mdata_3v3, "protein", "cond", "A", "B", p_adjust="storey")


# ----------------------------------------------------------------------- permutation


def test_permutation_auto_resolves_to_empirical(mdata_3v3):
    """welch with unspecified p_adjust keeps the empirical FDR (byte-identical to explicit empirical)."""
    auto = run_de(mdata_3v3, "protein", "cond", "A", "B", stat_method="welch")
    empirical = run_de(mdata_3v3, "protein", "cond", "A", "B", stat_method="welch", p_adjust="empirical")
    np.testing.assert_array_equal(auto.q_value, empirical.q_value)


def test_permutation_accepts_p_adjust_family(mdata_3v3):
    """welch permutation p-values can be BH-adjusted through the shared correction path."""
    res = run_de(mdata_3v3, "protein", "cond", "A", "B", stat_method="welch", p_adjust="bh")
    np.testing.assert_allclose(res.q_value, PvalueCorrection.adjust(res.p_value, "bh"), equal_nan=True)


def test_permutation_bonferroni_is_exact(mdata_3v3):
    res = run_de(mdata_3v3, "protein", "cond", "A", "B", stat_method="welch", p_adjust="bonferroni")
    n = _n_tested(res)
    np.testing.assert_allclose(res.q_value, np.minimum(1.0, n * res.p_value), equal_nan=True)


def test_permutation_rejects_unknown_method(mdata_3v3):
    with pytest.raises(ValueError, match="Unknown p_adjust"):
        run_de(mdata_3v3, "protein", "cond", "A", "B", stat_method="welch", p_adjust="storey")


# ----------------------------------------------------------------------------- shared


def test_p_adjust_must_be_a_string(mdata_3v3):
    with pytest.raises(ValueError, match="must be a string"):
        run_de(mdata_3v3, "protein", "cond", "A", "B", p_adjust=5)
