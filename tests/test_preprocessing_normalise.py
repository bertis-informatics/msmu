import numpy as np
import pytest

from msmu._preprocessing._normalise import correct_batch_effect, log2_transform, normalise


def test_log2_transform(simple_mdata):
    out = log2_transform(simple_mdata, modality="psm")
    assert np.allclose(out["psm"].X, np.log2(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])))


def test_normalise_quantile_runs(simple_mdata):
    out = normalise(simple_mdata, method="quantile", modality="psm")
    assert out["psm"].X.shape == (3, 2)


def test_correct_batch_effect_gis_drops_samples(simple_mdata):
    out = correct_batch_effect(simple_mdata, modality="psm", method="gis", gis_col="gis", rescale=False)
    assert out["psm"].n_obs == 2


def test_correct_batch_effect_gis_missing_raises(simple_mdata):
    simple_mdata["psm"].obs["gis"] = False
    with pytest.raises(ValueError, match="No GIS samples found"):
        correct_batch_effect(simple_mdata, modality="psm", method="gis", gis_col="gis", rescale=False)


def test_correct_batch_effect_invalid_method(simple_mdata):
    with pytest.raises(ValueError, match="not recognised"):
        correct_batch_effect(simple_mdata, modality="psm", method="nope")
