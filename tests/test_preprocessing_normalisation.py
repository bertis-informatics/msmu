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


def test_normalise_total_sum_raises():
    with pytest.raises(NotImplementedError):
        normalise_total_sum()


def test_ptm_protein_adjuster_ratio(ptm_mdata, global_mdata):
    adjuster = PTMProteinAdjuster(ptm_mdata, global_mdata, ptm_mod="phospho_site", global_mod="protein")
    ratio_df = adjuster._ratio()
    global_values = adjuster.global_data.loc[adjuster.ptm_data["protein_group"], adjuster.sample_cols].reset_index(
        drop=True
    )
    expected = adjuster.ptm_data[adjuster.sample_cols].to_numpy() - global_values.to_numpy()
    assert np.allclose(ratio_df[adjuster.sample_cols].to_numpy(), expected)
