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


def test_add_filter_on_obs_stores_in_obsm(filter_mdata):
    mdata = filter_mdata.copy()
    mdata["psm"].obs["group"] = ["A", "B"]
    out = add_filter(mdata, modality="psm", column="group", keep="eq", value="A", on="obs")
    assert "filter" in out["psm"].obsm_keys()
    assert out["psm"].obsm["filter"].shape[1] == 1
    assert out["psm"].obsm["filter"].iloc[:, 0].tolist() == [True, False]


def test_add_filter_on_obsm_with_key_stores_in_obsm(filter_mdata):
    mdata = filter_mdata.copy()
    mdata["psm"].obsm["qc"] = pd.DataFrame({"score": [0.1, 0.9]}, index=mdata["psm"].obs_names)
    out = add_filter(mdata, modality="psm", column="score", keep="gt", value=0.5, on="obsm", key="qc")
    assert "filter" in out["psm"].obsm_keys()
    assert out["psm"].obsm["filter"].iloc[:, 0].tolist() == [False, True]


def test_add_filter_requires_key_for_obsm(filter_mdata):
    with pytest.raises(ValueError, match="key must be provided"):
        add_filter(filter_mdata, modality="psm", column="score", keep="gt", value=0.5, on="obsm")


def test_apply_filter_on_obs(filter_mdata):
    mdata = filter_mdata.copy()
    mdata["psm"].obs["group"] = ["A", "B"]
    filtered = add_filter(mdata, modality="psm", column="group", keep="eq", value="A", on="obs")
    applied = apply_filter(filtered, modality="psm", on="obs")
    assert applied["psm"].obs_names.tolist() == ["s1"]
    assert applied["psm"].var_names.tolist() == ["v1", "v2", "v3"]


def test_apply_filter_on_all(filter_mdata):
    mdata = filter_mdata.copy()
    mdata["psm"].obs["group"] = ["A", "B"]
    filtered = add_filter(mdata, modality="psm", column="score", keep="gt", value=15.0, on="var")
    filtered = add_filter(filtered, modality="psm", column="group", keep="eq", value="A", on="obs")
    applied = apply_filter(filtered, modality="psm", on="all")
    assert applied["psm"].obs_names.tolist() == ["s1"]
    assert applied["psm"].var_names.tolist() == ["v2", "v3"]
    assert applied["psm"].uns["decoy"].index.tolist() == ["v2", "v3"]


def test_apply_filter_columns_limits_var_filters(filter_mdata):
    filtered = add_filter(filter_mdata, modality="psm", column="score", keep="gt", value=15.0, on="var")
    filtered = add_filter(filtered, modality="psm", column="score", keep="lt", value=25.0, on="var")

    applied = apply_filter(filtered, modality="psm", on="var", columns=["score_gt_15.0"])
    assert applied["psm"].var_names.tolist() == ["v2", "v3"]


def test_apply_filter_columns_limits_obs_filters(filter_mdata):
    mdata = filter_mdata.copy()
    mdata["psm"].obs["group"] = ["A", "B"]
    mdata["psm"].obs["cohort"] = ["X", "X"]
    filtered = add_filter(mdata, modality="psm", column="group", keep="eq", value="A", on="obs")
    filtered = add_filter(filtered, modality="psm", column="cohort", keep="eq", value="X", on="obs")

    applied = apply_filter(filtered, modality="psm", on="obs", columns=["group_eq_A"])
    assert applied["psm"].obs_names.tolist() == ["s1"]


def test_apply_filter_columns_with_unknown_raises_for_var_mode(filter_mdata):
    filtered = add_filter(filter_mdata, modality="psm", column="score", keep="gt", value=15.0, on="var")
    with pytest.raises(ValueError, match="No matching var filter columns found"):
        apply_filter(filtered, modality="psm", on="var", columns=["not_existing"])


def test_apply_filter_payload_records_columns(filter_mdata):
    filtered = add_filter(filter_mdata, modality="psm", column="score", keep="gt", value=15.0, on="var")
    applied = apply_filter(filtered, modality="psm", on="var", columns=["score_gt_15.0"])
    last_key = max(applied.uns["_cmd"], key=lambda x: int(x))
    entry = applied.uns["_cmd"][last_key]
    assert entry["function"] == "apply_filter"
    assert entry["payload"]["columns"] == ["score_gt_15.0"]


def test_apply_filter_stdout_records_filter_columns(filter_mdata):
    filtered = add_filter(filter_mdata, modality="psm", column="score", keep="gt", value=15.0, on="var")
    applied = apply_filter(filtered, modality="psm", on="var")
    last_key = max(applied.uns["_cmd"], key=lambda x: int(x))
    entry = applied.uns["_cmd"][last_key]
    assert "stdout" in entry
    assert "[Filter] Applying var filters:" in entry["stdout"]
    assert "score_gt_15.0" in entry["stdout"]
