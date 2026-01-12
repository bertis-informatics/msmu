import numpy as np
import pytest

from msmu._preprocessing._normalise import log2_transform, normalise


def test_log2_transform(simple_mdata):
    out = log2_transform(simple_mdata, modality="psm")
    assert np.allclose(
        out["psm"].X, np.log2(np.array([[1.0, 2.0], [3.0, 4.0], [6.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]))
    )


def test_normalise_quantile_runs(simple_mdata):
    out = normalise(simple_mdata, method="quantile", modality="psm")
    assert out["psm"].X.shape == (6, 2)


def test_normalise_median(simple_mdata):
    out = normalise(simple_mdata, method="median", modality="psm")
    assert out["psm"].X.shape == (6, 2)
