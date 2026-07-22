"""BID-70: the fold-change guidance line (fc_pct) is engine-independent.

fc_pct is populated on every ``DeaResult`` regardless of ``stat_method``: the permutation engine
produces it as a byproduct of its own shuffles, while limma / simple-test results derive it from
the same label-permutation log2FC null via the shared ``compute_fc_thresholds``. As a result
``plot_volcano``'s default log2FC line works for limma, and for the same data the guidance line
agrees across engines (identical when the design is small enough to enumerate every split).
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData

import msmu._tools._dea as dea
from msmu._tools._dea import run_de
from msmu._statistics._fc_threshold import compute_fc_thresholds

# features x samples, 3 vs 3 (A/B). 3-vs-3 has only C(6,3)=20 splits, so the permutation null is
# enumerated exactly -> the guidance line is deterministic and must match across engines.
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


def test_limma_result_has_fc_pct(mdata_3v3):
    """limma results carry the fold-change guidance line (previously fc_pct_5 was absent)."""
    res = run_de(mdata_3v3, "protein", category="cond", ctrl="A", expr="B", stat_method="limma")

    assert res.fc_pct_1 is not None
    assert res.fc_pct_5 is not None
    assert res.fc_pct_5 > 0
    # the more extreme (1st/99th) percentile threshold is at least the 5th/95th one
    assert res.fc_pct_1 >= res.fc_pct_5


def test_fc_pct_identical_across_engines_at_matching_effect_measure(mdata_3v3):
    """Same data + same effect measure -> same guidance line regardless of engine.

    limma's fold change is its mean-based model contrast, so its guidance line is built on the mean
    (``effect_measure="mean"``); it therefore matches a permutation engine run with ``measure="mean"``.
    Exact enumeration (3 vs 3 has only C(6,3)=20 splits) makes the null deterministic, so the
    thresholds are identical.
    """
    limma = run_de(mdata_3v3, "protein", category="cond", ctrl="A", expr="B", stat_method="limma")
    welch = run_de(
        mdata_3v3, "protein", category="cond", ctrl="A", expr="B",
        stat_method="welch", measure="mean", n_resamples=1000,
    )

    assert limma.fc_pct_1 == welch.fc_pct_1
    assert limma.fc_pct_5 == welch.fc_pct_5


def test_plot_volcano_default_works_for_limma(mdata_3v3):
    """plot_volcano() with no explicit threshold uses fc_pct_5 and must not raise for limma."""
    res = run_de(mdata_3v3, "protein", category="cond", ctrl="A", expr="B", stat_method="limma")

    figure = res.plot_volcano()  # previously raised (AttributeError / bare raise)

    assert figure is not None


def test_permutation_does_not_recompute_fc_pct(mdata_3v3, monkeypatch):
    """The permutation engine reuses its own byproduct; the standalone guidance-line routine is not called."""
    call_count = {"n": 0}
    original = dea.compute_fc_guidance_line

    def spy(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(dea, "compute_fc_guidance_line", spy)
    run_de(mdata_3v3, "protein", category="cond", ctrl="A", expr="B", stat_method="welch", n_resamples=1000)

    assert call_count["n"] == 0  # no second shuffle for the permutation path


def test_limma_interaction_skips_guidance_line():
    """An interaction contrast has no two-group fold-change line, so fc_pct stays unset."""
    obs = pd.DataFrame(
        {"treat": ["a"] * 6 + ["b"] * 6, "geno": (["c"] * 3 + ["d"] * 3) * 2},
        index=[f"T{j}" for j in range(12)],
    )
    x = np.random.default_rng(0).normal(size=(12, 6))
    mdata = MuData({"protein": AnnData(X=x, obs=obs, var=pd.DataFrame(index=[f"P{i}" for i in range(6)]))})

    res = run_de(
        mdata,
        "protein",
        category="treat",
        ctrl="a",
        expr="b",
        stat_method="limma",
        interaction="geno",
        interaction_levels=["c", "d"],
    )

    assert res.fc_pct_5 is None


def test_limma_covariate_skips_guidance_line():
    """A covariate-adjusted contrast reports an adjusted log2fc, so the raw two-group FC null would
    be scale-mismatched; the guidance line is skipped (fc_pct stays unset) while the test still runs."""
    obs = pd.DataFrame(
        {"treat": ["a"] * 6 + ["b"] * 6, "geno": (["c"] * 3 + ["d"] * 3) * 2},
        index=[f"T{j}" for j in range(12)],
    )
    x = np.random.default_rng(1).normal(size=(12, 6))
    mdata = MuData({"protein": AnnData(X=x, obs=obs, var=pd.DataFrame(index=[f"P{i}" for i in range(6)]))})

    res = run_de(
        mdata, "protein", category="treat", ctrl="a", expr="b", stat_method="limma", covariates=["geno"]
    )

    assert res.fc_pct_5 is None
    assert np.isfinite(res.log2fc).any()  # the covariate-adjusted test itself still runs


def test_compute_fc_thresholds_standalone():
    """The shared routine returns finite, ordered thresholds from a label-permutation null."""
    ctrl_arr = _Y.T[:3]  # 3 samples x 6 features
    expr_arr = _Y.T[3:]

    fc_pct_1, fc_pct_5 = compute_fc_thresholds(
        ctrl_arr, expr_arr, measure="median", log_transformed=True, n_resamples=1000
    )

    assert np.isfinite(fc_pct_1) and np.isfinite(fc_pct_5)
    assert fc_pct_1 >= fc_pct_5 > 0
