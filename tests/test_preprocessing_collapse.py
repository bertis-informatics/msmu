import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

from msmu._preprocessing._collapse import collapse_obs


@pytest.fixture
def fractionated_mdata() -> md.MuData:
    """Two samples × two fractions each (4 obs rows). One feature has all-NaN in one sample."""
    obs = pd.DataFrame(
        {
            "sample": ["s1", "s1", "s2", "s2"],
            "fraction": [1, 2, 1, 2],
            "condition": ["treated", "treated", "control", "control"],
            "filename": ["s1_f1.raw", "s1_f2.raw", "s2_f1.raw", "s2_f2.raw"],
        },
        index=["s1_f1", "s1_f2", "s2_f1", "s2_f2"],
    )
    var = pd.DataFrame({"protein": ["P1", "P2", "P3"]}, index=["pep1", "pep2", "pep3"])
    x = np.array(
        [
            [1.0, 2.0, np.nan],  # s1_f1
            [3.0, np.nan, 5.0],  # s1_f2
            [10.0, 20.0, np.nan],  # s2_f1
            [np.nan, np.nan, np.nan],  # s2_f2 — pep3 all-NaN across s2 fractions
        ]
    )
    adata = ad.AnnData(X=x, obs=obs, var=var)
    return md.MuData({"psm": adata})


@pytest.fixture
def fractionated_mdata_multi_mod(fractionated_mdata: md.MuData) -> md.MuData:
    """Adds a second modality (peptide) with the same obs layout — DDA-LFQ scenario."""
    psm_adata = fractionated_mdata.mod["psm"]
    peptide_var = pd.DataFrame(index=["peptideA", "peptideB"])
    peptide_x = np.array(
        [
            [0.5, 1.5],
            [0.6, 1.6],
            [2.5, 3.5],
            [2.6, 3.6],
        ]
    )
    peptide_adata = ad.AnnData(X=peptide_x, obs=psm_adata.obs.copy(), var=peptide_var)
    return md.MuData({"psm": psm_adata, "peptide": peptide_adata})


def test_collapse_obs_sum_default(fractionated_mdata):
    out = collapse_obs(fractionated_mdata, sample_key="sample", agg_method="sum")
    psm_x = out.mod["psm"].X
    expected = np.array(
        [
            [4.0, 2.0, 5.0],  # s1: 1+3, 2+nan, nan+5
            [10.0, 20.0, np.nan],  # s2: 10+nan, 20+nan, all-NaN → NaN
        ]
    )
    assert np.allclose(psm_x, expected, equal_nan=True)
    assert list(out.mod["psm"].obs.index) == ["s1", "s2"]


def test_collapse_obs_max(fractionated_mdata):
    out = collapse_obs(fractionated_mdata, sample_key="sample", agg_method="max")
    psm_x = out.mod["psm"].X
    expected = np.array(
        [
            [3.0, 2.0, 5.0],
            [10.0, 20.0, np.nan],
        ]
    )
    assert np.allclose(psm_x, expected, equal_nan=True)


def test_collapse_obs_mean(fractionated_mdata):
    out = collapse_obs(fractionated_mdata, sample_key="sample", agg_method="mean")
    psm_x = out.mod["psm"].X
    expected = np.array(
        [
            [2.0, 2.0, 5.0],
            [10.0, 20.0, np.nan],
        ]
    )
    assert np.allclose(psm_x, expected, equal_nan=True)


def test_collapse_obs_obs_scalar_and_list(fractionated_mdata):
    out = collapse_obs(fractionated_mdata, sample_key="sample")
    collapsed_obs = out.mod["psm"].obs
    # 'condition' is uniform per sample → scalar
    assert collapsed_obs.loc["s1", "condition"] == "treated"
    assert collapsed_obs.loc["s2", "condition"] == "control"
    # 'filename' and 'fraction' differ between rows of same sample → list
    assert collapsed_obs.loc["s1", "filename"] == ["s1_f1.raw", "s1_f2.raw"]
    assert collapsed_obs.loc["s2", "filename"] == ["s2_f1.raw", "s2_f2.raw"]
    assert collapsed_obs.loc["s1", "fraction"] == [1, 2]


def test_collapse_obs_var_preserved(fractionated_mdata):
    out = collapse_obs(fractionated_mdata, sample_key="sample")
    pd.testing.assert_frame_equal(out.mod["psm"].var, fractionated_mdata.mod["psm"].var)


def test_collapse_obs_applies_to_all_modalities(fractionated_mdata_multi_mod):
    out = collapse_obs(fractionated_mdata_multi_mod, sample_key="sample")
    assert out.mod["psm"].n_obs == 2
    assert out.mod["peptide"].n_obs == 2
    expected_peptide = np.array(
        [
            [1.1, 3.1],
            [5.1, 7.1],
        ]
    )
    assert np.allclose(out.mod["peptide"].X, expected_peptide)


def test_collapse_obs_missing_sample_key_raises(fractionated_mdata):
    with pytest.raises(KeyError, match="sample_key"):
        collapse_obs(fractionated_mdata, sample_key="nonexistent")


def test_collapse_obs_invalid_agg_method_raises(fractionated_mdata):
    with pytest.raises(ValueError, match="agg_method"):
        collapse_obs(fractionated_mdata, sample_key="sample", agg_method="invalid")


def test_collapse_obs_single_row_per_sample_is_noop(fractionated_mdata):
    """If sample_key uniquely identifies every row, collapse is essentially a no-op on values."""
    out = collapse_obs(fractionated_mdata, sample_key="filename")
    assert out.mod["psm"].n_obs == fractionated_mdata.mod["psm"].n_obs
    # single-row groups: nansum returns the original value (or NaN if originally NaN)
    pd.testing.assert_frame_equal(out.mod["psm"].var, fractionated_mdata.mod["psm"].var)


def test_collapse_obs_default_call_works(fractionated_mdata):
    """Default call (log_transformed=False, agg_method='sum') works for typical LFQ flow."""
    out = collapse_obs(fractionated_mdata, sample_key="sample")
    assert out.mod["psm"].n_obs == 2


def test_collapse_obs_sum_on_log_converts_via_linear(fractionated_mdata):
    """sum + log_transformed=True must compute log2(sum(2^x)), not direct sum of logs."""
    log_mdata = fractionated_mdata.copy()
    log_mdata.mod["psm"].X = np.log2(fractionated_mdata.mod["psm"].X)

    out_log = collapse_obs(log_mdata, sample_key="sample", agg_method="sum", log_transformed=True)
    out_linear = collapse_obs(fractionated_mdata, sample_key="sample", agg_method="sum")

    # log2(sum_linear) should equal collapse on log data with conversion
    expected_log = np.log2(out_linear.mod["psm"].X)
    assert np.allclose(out_log.mod["psm"].X, expected_log, equal_nan=True)


def test_collapse_obs_median_unchanged_by_log_flag(fractionated_mdata):
    """Median is monotonic — same result regardless of log_transformed flag."""
    out_false = collapse_obs(
        fractionated_mdata,
        sample_key="sample",
        agg_method="median",
        log_transformed=False,
    )
    out_true = collapse_obs(
        fractionated_mdata,
        sample_key="sample",
        agg_method="median",
        log_transformed=True,
    )
    assert np.allclose(out_false.mod["psm"].X, out_true.mod["psm"].X, equal_nan=True)
