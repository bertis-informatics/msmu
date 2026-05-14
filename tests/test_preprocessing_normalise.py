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


def test_normalise_batch_key_groups_within_batch(mdata):
    """median centering within batches should zero each sample's median within its batch block."""
    out = normalise(mdata, method="median_center", modality="psm", batch_key="batch")
    arr = out["psm"].X
    # mdata: batch="x" → obs rows [0, 2], batch="y" → obs rows [1, 3]
    for batch_rows in ([0, 2], [1, 3]):
        block = arr[batch_rows, :]
        row_medians = np.nanmedian(block, axis=1)
        assert np.allclose(row_medians[~np.isnan(row_medians)], 0.0)


def test_normalise_batch_key_invalid_raises(mdata):
    with pytest.raises(KeyError, match="batch_key"):
        normalise(mdata, method="median", modality="psm", batch_key="nonexistent")


def test_normalise_fraction_key_invalid_raises(mdata):
    with pytest.raises(KeyError, match="fraction_key"):
        normalise(mdata, method="median", modality="psm", fraction_key="nonexistent")


def test_normalise_fraction_bool_emits_deprecation_warning(mdata):
    mdata.mod["psm"].var["filename"] = ["f1", "f2", "f1"]
    with pytest.warns(DeprecationWarning, match="fraction"):
        normalise(mdata, method="median_center", modality="psm", fraction=True)


def test_normalise_fraction_key_matches_legacy_fraction(mdata):
    mdata.mod["psm"].var["filename"] = ["f1", "f2", "f1"]
    new_out = normalise(mdata, method="median_center", modality="psm", fraction_key="filename")
    with pytest.warns(DeprecationWarning):
        legacy_out = normalise(mdata, method="median_center", modality="psm", fraction=True)
    assert np.allclose(new_out["psm"].X, legacy_out["psm"].X, equal_nan=True)


def test_normalise_batch_and_fraction_combined(mdata):
    """Both grouping keys together: normalise within each (batch × fraction) block."""
    mdata.mod["psm"].var["filename"] = ["f1", "f2", "f1"]
    out = normalise(
        mdata, method="median_center", modality="psm", batch_key="batch", fraction_key="filename"
    )
    arr = out["psm"].X
    # batches: "x" → obs rows [0, 2]; "y" → obs rows [1, 3]
    # fractions: "f1" → var cols [0, 2]; "f2" → var col [1]
    # axis="obs" normalises per-sample, so each row's median within each block should be 0
    for obs_idx, var_idx in [([0, 2], [0, 2]), ([0, 2], [1]), ([1, 3], [0, 2]), ([1, 3], [1])]:
        block = arr[np.ix_(obs_idx, var_idx)]
        row_medians = np.nanmedian(block, axis=1)
        assert np.allclose(row_medians[~np.isnan(row_medians)], 0.0)
