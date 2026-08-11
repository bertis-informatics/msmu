import numpy as np
import pytest

from msmu._preprocessing._normalise import log2_transform, normalise


def test_log2_transform(simple_mdata):
    out = log2_transform(simple_mdata, modality="psm")
    assert np.allclose(
        out["psm"].X,
        np.log2(
            np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [6.0, 6.0],
                    [7.0, 8.0],
                    [9.0, 10.0],
                    [11.0, 12.0],
                ]
            )
        ),
    )


def test_normalise_quantile_runs(simple_mdata):
    out = normalise(simple_mdata, method="quantile", modality="psm")
    assert out["psm"].X.shape == (6, 2)


def test_normalise_median(simple_mdata):
    out = normalise(simple_mdata, method="median", modality="psm")
    assert out["psm"].X.shape == (6, 2)


def test_normalise_group_obs_groups_within_group(mdata):
    """median centering within obs groups should zero each sample's median within its group block."""
    out = normalise(mdata, method="median_center", modality="psm", group_obs="batch")
    arr = out["psm"].X
    # mdata: batch="x" → obs rows [0, 2], batch="y" → obs rows [1, 3]
    for group_rows in ([0, 2], [1, 3]):
        block = arr[group_rows, :]
        row_medians = np.nanmedian(block, axis=1)
        assert np.allclose(row_medians[~np.isnan(row_medians)], 0.0)


def test_normalise_group_obs_invalid_raises(mdata):
    with pytest.raises(KeyError, match="group_obs"):
        normalise(mdata, method="median", modality="psm", group_obs="nonexistent")


def test_normalise_group_var_invalid_raises(mdata):
    with pytest.raises(KeyError, match="group_var"):
        normalise(mdata, method="median", modality="psm", group_var="nonexistent")


def test_normalise_group_obs_and_group_var_combined(mdata):
    """Both grouping keys together: normalise within each (obs-group × var-group) block."""
    mdata.mod["psm"].var["filename"] = ["f1", "f2", "f1"]
    out = normalise(mdata, method="median_center", modality="psm", group_obs="batch", group_var="filename")
    arr = out["psm"].X
    # obs groups: "x" → rows [0, 2]; "y" → rows [1, 3]. var groups: "f1" → cols [0, 2]; "f2" → col [1].
    # axis="obs" normalises per-sample, so each row's median within each block should be 0.
    for obs_idx, var_idx in [([0, 2], [0, 2]), ([0, 2], [1]), ([1, 3], [0, 2]), ([1, 3], [1])]:
        block = arr[np.ix_(obs_idx, var_idx)]
        row_medians = np.nanmedian(block, axis=1)
        assert np.allclose(row_medians[~np.isnan(row_medians)], 0.0)


# ---- deprecated aliases: still work, but warn and map to the new names ----


def test_normalise_batch_key_is_deprecated_alias_for_group_obs(mdata):
    with pytest.warns(DeprecationWarning, match="batch_key"):
        legacy = normalise(mdata, method="median_center", modality="psm", batch_key="batch")
    current = normalise(mdata, method="median_center", modality="psm", group_obs="batch")
    assert np.allclose(legacy["psm"].X, current["psm"].X, equal_nan=True)


def test_normalise_fraction_key_is_deprecated_alias_for_group_var(mdata):
    mdata.mod["psm"].var["filename"] = ["f1", "f2", "f1"]
    with pytest.warns(DeprecationWarning, match="fraction_key"):
        legacy = normalise(mdata, method="median_center", modality="psm", fraction_key="filename")
    current = normalise(mdata, method="median_center", modality="psm", group_var="filename")
    assert np.allclose(legacy["psm"].X, current["psm"].X, equal_nan=True)


def test_normalise_fraction_bool_deprecated_and_maps_to_group_var_filename(mdata):
    mdata.mod["psm"].var["filename"] = ["f1", "f2", "f1"]
    with pytest.warns(DeprecationWarning, match="fraction"):
        legacy = normalise(mdata, method="median_center", modality="psm", fraction=True)
    current = normalise(mdata, method="median_center", modality="psm", group_var="filename")
    assert np.allclose(legacy["psm"].X, current["psm"].X, equal_nan=True)
