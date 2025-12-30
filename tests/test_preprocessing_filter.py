import pandas as pd
import pytest

from msmu._preprocessing._filter import _mask_boolean_filter, add_filter, apply_filter


def test_mask_boolean_filter_ops():
    series = pd.Series(["a", "b", "aa"])
    assert _mask_boolean_filter(series, "contains", "a").tolist() == [True, False, True]
    assert _mask_boolean_filter(series, "not_contains", "b").tolist() == [True, False, True]
    assert _mask_boolean_filter(pd.Series([1, 2, 3]), "gt", 1).tolist() == [False, True, True]


def test_mask_boolean_filter_invalid():
    with pytest.raises(ValueError, match="Unknown filter operator"):
        _mask_boolean_filter(pd.Series([1, 2, 3]), "nope", 1)


def test_add_filter_and_apply_filter_with_decoy(filter_mdata):
    filtered = add_filter(filter_mdata, modality="psm", column="score", keep="gt", value=15.0)
    assert "filter" in filtered["psm"].varm_keys()
    assert filtered["psm"].varm["filter"].shape[1] == 1

    applied = apply_filter(filtered, modality="psm")
    assert applied["psm"].var_names.tolist() == ["v2", "v3"]
    assert applied["psm"].uns["decoy"].index.tolist() == ["v2", "v3"]
