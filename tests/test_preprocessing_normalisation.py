import numpy as np
import pytest

from msmu._preprocessing._normalisation import (
    Normalisation,
    PTMProteinAdjuster,
    normalise_median_center,
    normalise_quantile,
    normalise_total_sum,
)


def test_normalise_median_center_preserves_shape():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    centered = normalise_median_center(arr)
    assert centered.shape == arr.shape
    assert np.allclose(np.nanmedian(centered, axis=0), [0.0, 0.0])


def test_normalisation_class_preserves_nan():
    arr = np.array([[1.0, np.nan], [3.0, 5.0]])
    norm = Normalisation(method="median_center", axis="var").normalise(arr=arr)
    assert np.isnan(norm[0, 1])


def test_normalise_quantile_shape_and_nans():
    arr = np.array([[1.0, np.nan], [2.0, 4.0], [3.0, 5.0]])
    norm = normalise_quantile(arr)
    assert norm.shape == arr.shape
    assert np.isnan(norm[0, 1])


def test_normalise_total_sum_scales_columns_to_median_total():
    """total_sum rescales each sample (a column, in the features x samples orientation it receives) so
    its linear total equals T = the median of the per-sample totals."""
    arr = np.log2(np.array([[100.0, 50.0, 25.0], [100.0, 50.0, 25.0]]))  # 2 features x 3 samples; totals 200/100/50
    out = normalise_total_sum(arr)
    assert np.allclose(np.nansum(2.0**out, axis=0), 100.0)  # T = median(200, 100, 50) = 100


def test_normalise_total_sum_removes_pure_loading_and_keeps_within_sample_structure():
    """A per-sample scalar shift: a sample loaded 2x (a +1 log2 shift) collapses onto the other, and the
    differences between features within a sample are unchanged."""
    base = np.array([[1.0], [3.0], [2.0], [4.0]])  # 4 features x 1 sample
    arr = np.hstack([base, base + 1.0])  # sample 1 = sample 0 at 2x linear loading (features x samples)
    out = normalise_total_sum(arr)
    assert np.allclose(out[:, 0], out[:, 1])  # pure loading difference removed
    assert np.allclose(np.diff(out[:, 0]), np.diff(base[:, 0]))  # within-sample structure preserved


def test_total_sum_via_normalisation_preserves_nan():
    """The Normalisation wrapper (axis="obs") restores structurally-absent cells to NaN."""
    arr = np.array([[1.0, np.nan, 3.0], [2.0, 4.0, 5.0]])  # samples x features
    out = Normalisation(method="total_sum", axis="obs").normalise(arr=arr)
    assert np.isnan(out[0, 1])
    assert not np.isnan(out[0, 0])


def test_normalisation_rejects_unknown_method():
    """An unrecognised method fails fast with a clear ValueError rather than a cryptic AttributeError."""
    with pytest.raises(ValueError, match="Unknown normalisation method"):
        Normalisation(method="not_a_method", axis="obs")


def test_ptm_protein_adjuster_ratio(ptm_mdata, global_mdata):
    adjuster = PTMProteinAdjuster(ptm_mdata, global_mdata, ptm_mod="phospho_site", global_mod="protein")
    ratio_df = adjuster._ratio()
    global_values = adjuster.global_data.loc[adjuster.ptm_data["protein_group"], adjuster.sample_cols].reset_index(
        drop=True
    )
    expected = adjuster.ptm_data[adjuster.sample_cols].to_numpy() - global_values.to_numpy()
    assert np.allclose(ratio_df[adjuster.sample_cols].to_numpy(), expected)
