import os

import pytest

os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")

from msmu._tools._dea import _get_test_array, run_de
from msmu._tools._pca import pca

try:
    from msmu._tools._umap import umap
except RuntimeError as exc:
    pytest.skip(f"UMAP import failed: {exc}", allow_module_level=True)


def test_pca_writes_outputs(simple_mdata):
    out = pca(simple_mdata, modality="psm", n_components=2, random_state=0)
    assert "X_pca" in out["psm"].obsm
    assert out["psm"].obsm["X_pca"].shape[1] == 2
    assert "X_pca" in out["psm"].uns
    assert out["psm"].uns["X_pca"]["n_components"] == 2
    assert "X_pca" in out["psm"].varm


@pytest.mark.filterwarnings("ignore:n_jobs value .* overridden .* by setting random_state.*:UserWarning")
def test_umap_writes_outputs(simple_mdata):
    out = umap(simple_mdata, modality="psm", n_neighbors=2, random_state=0)
    assert "X_umap" in out["psm"].obsm
    assert out["psm"].obsm["X_umap"].shape[1] == 2
    assert out["psm"].uns["X_umap"]["n_components"] == 2


def test_get_test_array_splits_groups(mdata):
    ctrl_arr, expr_arr = _get_test_array(
        mdata=mdata,
        modality="protein",
        category="group",
        control="A",
        expr="B",
        layer=None,
    )
    assert ctrl_arr.shape[0] == 2
    assert expr_arr.shape[0] == 2


def test_run_de_with_simple_test(mdata):
    res = run_de(
        mdata=mdata,
        modality="protein",
        category="group",
        ctrl="A",
        expr="B",
        stat_method="welch",
        n_resamples=None,
        fdr=False,
    )
    assert res.ctrl == "A"
    assert res.expr == "B"
    assert res.features.size == mdata.mod["protein"].var.shape[0]


def test_run_de_invalid_stat_method_raises(mdata):
    with pytest.raises(ValueError, match="Invalid statistic"):
        run_de(
            mdata,
            modality="protein",
            category="group",
            ctrl="A",
            expr="B",
            stat_method="nope",
            n_resamples=None,
        )


def test_run_de_invalid_fdr_raises(mdata):
    with pytest.raises(ValueError, match="invalied fdr"):
        run_de(
            mdata,
            modality="protein",
            category="group",
            ctrl="A",
            expr="B",
            fdr="nope",
            n_resamples=None,
        )


def test_run_de_simple_test_exposes_statistic(mdata):
    res = run_de(
        mdata,
        modality="protein",
        category="group",
        ctrl="A",
        expr="B",
        stat_method="welch",
        n_resamples=None,
        fdr=False,
    )
    # the Welch t is now surfaced on DeaResult.statistic (was previously dropped)
    assert res.statistic is not None
    assert res.statistic.size == res.features.size
    assert "statistic" in res.to_df().columns


def test_run_de_permutation_exposes_observed_statistic(mdata):
    res = run_de(
        mdata,
        modality="protein",
        category="group",
        ctrl="A",
        expr="B",
        stat_method="welch",
        n_resamples=1000,
        fdr="bh",
    )
    # permutation path surfaces the observed (Welch) statistic; p_value stays empirical
    assert res.statistic is not None
    assert res.statistic.size == res.features.size
    assert "statistic" in res.to_df().columns
