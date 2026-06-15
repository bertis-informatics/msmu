import numpy as np

from msmu._statistics._de_base import DeaResult, StatTestResult
from msmu._tools._dea import run_de


def test_run_de_expr_none_uses_all_other_groups(mdata):
    res = run_de(
        mdata,
        modality="protein",
        category="group",
        ctrl="A",
        expr=None,
        stat_method="welch",
        n_resamples=None,
        fdr=False,
    )
    assert res.expr == "all_other_groups"


def test_run_de_permutation_path(mdata):
    res = run_de(
        mdata,
        modality="protein",
        category="group",
        ctrl="A",
        expr="B",
        stat_method="welch",
        n_resamples=2,
        fdr="bh",
        _force_resample=True,
    )
    assert res.p_value.shape[0] == mdata.mod["protein"].var.shape[0]


def test_dea_result_plot_volcano_uses_public_plotting_facade_behavior():
    res = DeaResult(
        StatTestResult(
            stat_method="welch",
            p_value=np.array([0.01, 0.01, 0.5]),
            q_value=np.array([0.01, 0.01, 0.5]),
        )
    )
    res.ctrl = "A"
    res.expr = "B"
    res.features = np.array(["up_feature", "down_feature", "stable_feature"])
    res.repr_ctrl = np.array([1.0, 3.0, 2.0])
    res.repr_expr = np.array([3.0, 1.0, 2.1])
    res.pct_ctrl = np.array([100.0, 100.0, 100.0])
    res.pct_expr = np.array([100.0, 100.0, 100.0])
    res.log2fc = np.array([2.0, -2.0, 0.1])

    fig = res.plot_volcano(log2fc_threshold=1.0, pval_threshold=0.05, label_top=1)

    assert fig.layout.title.text == "A vs. B"
    assert {trace.name for trace in fig.data} == {"DOWN", "UP", "nonDE"}
    assert len(fig.layout.shapes) == 3
    assert "A (1)" in [annotation.text for annotation in fig.layout.annotations]
    assert "B (1)" in [annotation.text for annotation in fig.layout.annotations]
