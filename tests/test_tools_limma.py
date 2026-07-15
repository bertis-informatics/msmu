"""Tests for limma moderated-t DE via msmu.tl.run_de(stat_method="limma").

Golden values are from R limma 3.62 (lmFit -> contrasts.fit -> eBayes -> topTable)
on the embedded fixed datasets, so this regression runs without R installed.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData

from msmu._statistics._limma import build_contrast
from msmu._tools._dea import run_de


# --- fixed datasets (features x samples) and R limma golden outputs ----------------

_Y1 = np.array(
    [
        [10.305, 8.96, 10.75, 12.441, 9.549, 10.198],
        [10.128, 9.684, 9.983, 10.647, 12.379, 12.278],
        [10.066, 11.127, 10.468, 9.141, 10.369, 9.041],
        [10.878, 9.95, 9.815, 9.319, 11.223, 9.845],
        [9.572, 9.648, 10.532, 10.365, 10.413, 10.431],
        [12.142, 9.594, 9.488, 9.186, 10.616, 11.129],
    ]
)
_L1_R_LOGFC = np.array([0.724333, 1.836333, -1.036667, -0.085333, 0.485667, -0.097667])
_L1_R_P = np.array([0.376656, 0.014532, 0.134686, 0.902717, 0.444989, 0.904544])

_Y2 = np.array(
    [
        [11.886, 11.16, 11.176, 10.651, 10.743, 10.543, 9.334, 10.232, 10.117, 10.219, 10.871, 10.224],
        [12.679, 12.068, 12.289, 10.631, 8.543, 9.68, 9.53, 9.361, 9.725, 11.495, 9.134, 10.968],
        [8.317, 9.665, 10.163, 10.586, 10.711, 10.793, 9.651, 9.538, 10.858, 9.809, 8.724, 8.867],
        [9.081, 10.497, 10.142, 10.69, 9.573, 10.159, 10.626, 9.691, 10.457, 9.338, 9.637, 9.618],
        [8.804, 10.487, 9.531, 10.012, 10.481, 10.447, 10.665, 9.902, 9.577, 9.92, 8.313, 8.553],
        [8.677, 9.003, 10.4, 9.095, 9.622, 11.299, 9.644, 10.738, 9.066, 9.795, 9.05, 9.661],
    ]
)
_L2_R_LOGFC = np.array([1.305333, 3.721, -2.197333, -0.961, -1.825333, -0.959333])
_L2_R_P = np.array([0.066578, 0.000109, 0.007883, 0.194519, 0.024872, 0.263995])


def _make_mdata(feature_by_sample: np.ndarray, obs: pd.DataFrame, modality: str = "protein") -> MuData:
    sample_by_feature = feature_by_sample.T
    var = pd.DataFrame(index=[f"P{i}" for i in range(feature_by_sample.shape[0])])
    adata = AnnData(X=sample_by_feature, obs=obs.copy(), var=var)
    return MuData({modality: adata})


@pytest.fixture
def level1_mdata() -> MuData:
    obs = pd.DataFrame(
        {"cond": ["A", "A", "A", "B", "B", "B"]},
        index=[f"S{j}" for j in range(6)],
    )
    return _make_mdata(_Y1, obs)


@pytest.fixture
def level2_mdata() -> MuData:
    obs = pd.DataFrame(
        {
            "treat": ["a"] * 6 + ["b"] * 6,
            "geno": (["c"] * 3 + ["d"] * 3) * 2,
        },
        index=[f"T{j}" for j in range(12)],
    )
    return _make_mdata(_Y2, obs)


# --- golden regression vs R limma -------------------------------------------------

def test_main_effect_matches_r_limma(level1_mdata):
    res = run_de(level1_mdata, "protein", category="cond", ctrl="A", expr="B", stat_method="limma")
    np.testing.assert_allclose(res.log2fc, _L1_R_LOGFC, atol=1e-4)
    np.testing.assert_allclose(res.p_value, _L1_R_P, atol=1e-4)
    assert res.contrast_label == "B vs A"


def test_interaction_matches_r_limma(level2_mdata):
    # DiD (c-d)@a - (c-d)@b == R's (ac-ad)-(bc-bd)
    res = run_de(
        level2_mdata,
        "protein",
        category="geno",
        ctrl="d",
        expr="c",
        stat_method="limma",
        interaction="treat",
        interaction_levels=["a", "b"],
    )
    np.testing.assert_allclose(res.log2fc, _L2_R_LOGFC, atol=1e-4)
    np.testing.assert_allclose(res.p_value, _L2_R_P, atol=1e-4)
    assert "interaction across treat" in res.contrast_label


# --- contrast construction (the part msmu owns) -----------------------------------

def test_build_contrast_main_effect_weights():
    obs = pd.DataFrame({"cond": ["A", "A", "B", "B"]}, index=list("wxyz"))
    contrast = build_contrast(obs, category="cond", ctrl="A", expr="B")
    weights = dict(zip(contrast.cell_columns, contrast.weights))
    assert weights == {"A": -1.0, "B": 1.0}


def test_build_contrast_interaction_is_difference_of_differences():
    obs = pd.DataFrame(
        {"geno": ["c", "d", "c", "d"], "treat": ["a", "a", "b", "b"]},
        index=list("wxyz"),
    )
    contrast = build_contrast(
        obs, category="geno", ctrl="d", expr="c", interaction="treat", interaction_levels=["a", "b"]
    )
    weights = dict(zip(contrast.cell_columns, contrast.weights))
    # (c@a - d@a) - (c@b - d@b)
    assert weights == {"c|a": 1.0, "d|a": -1.0, "c|b": -1.0, "d|b": 1.0}


# --- behaviour / guards -----------------------------------------------------------

def test_sign_convention_positive_is_higher_in_expr(level1_mdata):
    res = run_de(level1_mdata, "protein", category="cond", ctrl="A", expr="B", stat_method="limma")
    # P1 is clearly up in B (injected); logFC must be positive
    p1 = np.where(res.features == "P1")[0][0]
    assert res.log2fc[p1] > 0
    # swapping ctrl/expr flips the sign
    swapped = run_de(level1_mdata, "protein", category="cond", ctrl="B", expr="A", stat_method="limma")
    np.testing.assert_allclose(res.log2fc, -swapped.log2fc, atol=1e-9)


def test_unknown_level_raises_with_available_listed(level1_mdata):
    with pytest.raises(ValueError, match=r"not found in obs column 'cond'.*Available"):
        run_de(level1_mdata, "protein", category="cond", ctrl="A", expr="ZZZ", stat_method="limma")


def test_interaction_requires_exactly_two_levels():
    obs = pd.DataFrame(
        {"geno": ["c", "d", "c", "d", "c", "d"], "treat": ["a", "a", "b", "b", "e", "e"]},
        index=[f"S{i}" for i in range(6)],
    )
    mdata = _make_mdata(np.random.default_rng(0).normal(10, 1, (4, 6)), obs)
    with pytest.raises(ValueError, match="exactly 2 levels"):
        run_de(mdata, "protein", category="geno", ctrl="d", expr="c", stat_method="limma", interaction="treat")


def test_interaction_rejected_for_non_limma_method(level2_mdata):
    with pytest.raises(ValueError, match="only supported with stat_method='limma'"):
        run_de(
            level2_mdata, "protein", category="geno", ctrl="d", expr="c",
            stat_method="welch", interaction="treat", n_resamples=None,
        )


def test_limma_requires_explicit_expr(level1_mdata):
    with pytest.raises(ValueError, match="requires an explicit 'expr'"):
        run_de(level1_mdata, "protein", category="cond", ctrl="A", expr=None, stat_method="limma")


def test_min_pct_filters_low_coverage_features():
    obs = pd.DataFrame({"cond": ["A", "A", "A", "B", "B", "B"]}, index=[f"S{j}" for j in range(6)])
    y = _Y1.copy()
    y[0, 1:] = np.nan  # feature P0: only 1 non-missing across all samples
    mdata = _make_mdata(y, obs)
    res = run_de(mdata, "protein", category="cond", ctrl="A", expr="B", stat_method="limma", min_pct=0.5)
    p0 = np.where(res.features == "P0")[0][0]
    assert np.isnan(res.p_value[p0])
    assert not np.isnan(res.p_value[np.where(res.features == "P1")[0][0]])


def test_covariate_adjustment_runs(level2_mdata):
    res = run_de(
        level2_mdata, "protein", category="geno", ctrl="d", expr="c",
        stat_method="limma", covariates=["treat"],
    )
    assert res.features.size == 6
    assert np.isfinite(res.log2fc).any()


def test_result_frame_has_expected_columns(level1_mdata):
    res = run_de(level1_mdata, "protein", category="cond", ctrl="A", expr="B", stat_method="limma")
    frame = res.to_df()
    for column in ["features", "log2fc", "statistic", "p_value", "q_value"]:
        assert column in frame.columns
    assert len(frame) == 6


def test_statistic_field_is_moderated_t(level1_mdata):
    res = run_de(level1_mdata, "protein", category="cond", ctrl="A", expr="B", stat_method="limma")
    # limma surfaces the moderated t on DeaResult.statistic (aligned to features)
    assert res.statistic is not None
    assert res.statistic.size == res.features.size
    # P1 is up in B (injected): positive log2fc AND positive moderated t, and the
    # moderated t agrees in sign with the fold change everywhere it is defined
    p1 = np.where(res.features == "P1")[0][0]
    assert res.statistic[p1] > 0
    assert np.isfinite(res.statistic[p1])
    defined = np.isfinite(res.statistic) & np.isfinite(res.log2fc)
    np.testing.assert_array_equal(np.sign(res.statistic[defined]), np.sign(res.log2fc[defined]))
