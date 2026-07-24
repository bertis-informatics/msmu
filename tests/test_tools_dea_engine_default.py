"""BID-71: limma is the default DE engine; permutation is opt-in and always shuffles.

Covers the default-engine switch (+ the one-time transition notice), the removal of the
``measure`` / ``fdr`` parameters (the fold-change central tendency and the q-method are now
engine-decided), and the ``n_resamples`` guard that redirects the "turn permutation off" mistake
to ``stat_method="limma"``.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData

import msmu._tools._dea as dea
from msmu._statistics._statistics import _measure_central_tendency
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


def test_default_engine_is_limma(mdata_3v3):
    """With stat_method unspecified the engine is limma (the new default)."""
    res = run_de(mdata_3v3, "protein", category="cond", ctrl="A", expr="B")

    assert res.stat_method == "limma"
    assert res.contrast_label == "B vs A"  # a limma-only field, so the limma engine ran


def test_default_limma_requires_expr(mdata_3v3):
    """The default engine is limma, which needs an explicit expr — omitting it errors clearly."""
    with pytest.raises(ValueError, match="requires an explicit 'expr'"):
        run_de(mdata_3v3, "protein", category="cond", ctrl="A")  # expr omitted


def test_transition_notice_fires_once_per_session(mdata_3v3, monkeypatch, caplog):
    """The welch->limma default change is announced once per process, not on every call."""
    monkeypatch.setattr(dea, "_default_engine_notice_shown", False)
    with caplog.at_level("WARNING", logger="msmu._tools._dea"):
        run_de(mdata_3v3, "protein", category="cond", ctrl="A", expr="B", stat_method="limma")
        run_de(mdata_3v3, "protein", category="cond", ctrl="A", expr="B", stat_method="limma")

    notices = [r for r in caplog.records if "default DE engine is now 'limma'" in r.getMessage()]
    assert len(notices) == 1


def test_permutation_off_mistake_redirects_to_limma(mdata_3v3):
    """A permutation method with n_resamples<1 is not 'off'; the error points to stat_method='limma'."""
    with pytest.raises(ValueError, match="stat_method='limma'"):
        run_de(mdata_3v3, "protein", category="cond", ctrl="A", expr="B", stat_method="welch", n_resamples=0)


def test_limma_ignores_n_resamples(mdata_3v3):
    """A limma user passing n_resamples is harmless and has NO effect — not on the test and not on
    the guidance line (its shuffle count is fixed). Any value gives byte-identical results."""
    default = run_de(mdata_3v3, "protein", category="cond", ctrl="A", expr="B", stat_method="limma")

    for n_resamples in (0, 1, 50, 5000):
        other = run_de(
            mdata_3v3, "protein", category="cond", ctrl="A", expr="B", stat_method="limma", n_resamples=n_resamples
        )
        np.testing.assert_allclose(other.log2fc, default.log2fc, equal_nan=True)
        assert other.fc_pct_1 == default.fc_pct_1
        assert other.fc_pct_5 == default.fc_pct_5


def test_effect_measure_follows_the_test(mdata_3v3):
    """No `measure` knob: welch tests the mean (mean FC), wilcoxon the rank/median (median FC)."""
    welch = run_de(mdata_3v3, "protein", category="cond", ctrl="A", expr="B", stat_method="welch", n_resamples=1000)
    wilcoxon = run_de(
        mdata_3v3, "protein", category="cond", ctrl="A", expr="B", stat_method="wilcoxon", n_resamples=1000
    )
    ctrl_arr, expr_arr = _Y.T[:3], _Y.T[3:]

    # welch -> mean: repr are group means and repr_expr - repr_ctrl == log2fc exactly
    np.testing.assert_allclose(welch.repr_ctrl, _measure_central_tendency(ctrl_arr, "mean"))
    np.testing.assert_allclose(welch.repr_expr - welch.repr_ctrl, welch.log2fc, atol=1e-9)
    # wilcoxon -> median: repr are group medians
    np.testing.assert_allclose(wilcoxon.repr_ctrl, _measure_central_tendency(ctrl_arr, "median"))
    np.testing.assert_allclose(wilcoxon.repr_expr, _measure_central_tendency(expr_arr, "median"))


def test_removed_measure_and_fdr_params_are_rejected(mdata_3v3):
    """`measure` and `fdr` are gone from the API (engine-decided); passing them raises TypeError."""
    with pytest.raises(TypeError):
        run_de(mdata_3v3, "protein", category="cond", ctrl="A", expr="B", stat_method="welch", fdr="bh")
    with pytest.raises(TypeError):
        run_de(mdata_3v3, "protein", category="cond", ctrl="A", expr="B", stat_method="welch", measure="mean")
