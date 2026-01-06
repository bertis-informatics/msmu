import numpy as np

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
        n_jobs=1,
        _force_resample=True,
    )
    assert res.p_value.shape[0] == mdata["protein"].var.shape[0]
