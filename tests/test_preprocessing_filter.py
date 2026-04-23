import io
import logging

import pandas as pd
import pytest

from msmu.logging_utils import get_logger
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
    mdata.mod["psm"].obs["group"] = ["A", "B"]
    out = add_filter(mdata, modality="psm", column="group", keep="eq", value="A", on="obs")
    assert "filter" in out["psm"].obsm_keys()
    assert out["psm"].obsm["filter"].shape[1] == 1
    assert out["psm"].obsm["filter"].iloc[:, 0].tolist() == [True, False]


def test_add_filter_on_obsm_with_key_stores_in_obsm(filter_mdata):
    mdata = filter_mdata.copy()
    mdata.mod["psm"].obsm["qc"] = pd.DataFrame({"score": [0.1, 0.9]}, index=mdata.mod["psm"].obs_names)
    out = add_filter(mdata, modality="psm", column="score", keep="gt", value=0.5, on="obsm", key="qc")
    assert "filter" in out["psm"].obsm_keys()
    assert out["psm"].obsm["filter"].iloc[:, 0].tolist() == [False, True]


def test_add_filter_requires_key_for_obsm(filter_mdata):
    with pytest.raises(ValueError, match="key must be provided"):
        add_filter(filter_mdata, modality="psm", column="score", keep="gt", value=0.5, on="obsm")


def test_add_filter_requires_unique_source_column(filter_mdata):
    mdata = filter_mdata.copy()
    mdata.mod["psm"].var = pd.DataFrame(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
        columns=["score", "score"],
        index=mdata.mod["psm"].var_names,
    )

    with pytest.raises(ValueError, match="must identify a single column"):
        add_filter(mdata, modality="psm", column="score", keep="gt", value=15.0)


def test_apply_filter_on_obs(filter_mdata):
    mdata = filter_mdata.copy()
    mdata.mod["psm"].obs["group"] = ["A", "B"]
    filtered = add_filter(mdata, modality="psm", column="group", keep="eq", value="A", on="obs")
    applied = apply_filter(filtered, modality="psm", on="obs")
    assert applied["psm"].obs_names.tolist() == ["s1"]
    assert applied["psm"].var_names.tolist() == ["v1", "v2", "v3"]


def test_apply_filter_on_all(filter_mdata):
    mdata = filter_mdata.copy()
    mdata.mod["psm"].obs["group"] = ["A", "B"]
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
    mdata.mod["psm"].obs["group"] = ["A", "B"]
    mdata.mod["psm"].obs["cohort"] = ["X", "X"]
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
    assert "Applying var filters for psm:" in entry["stdout"]
    assert "score_gt_15.0" in entry["stdout"]


def test_apply_filter_all_with_var_filters_does_not_warn_for_missing_obs(filter_mdata):
    filtered = add_filter(filter_mdata, modality="psm", column="score", keep="gt", value=15.0)
    filtered = add_filter(filtered, modality="psm", column="score", keep="lt", value=25.0)

    applied = apply_filter(filtered, modality="psm")

    last_key = max(applied.uns["_cmd"], key=lambda x: int(x))
    entry = applied.uns["_cmd"][last_key]
    assert applied["psm"].var_names.tolist() == ["v2"]
    assert "Applying var filters for psm:" in entry["stdout"]
    assert "obsm['filter']" not in entry["stdout"]


def test_apply_filter_all_with_obs_filter_and_decoy_does_not_require_var_decoy_filter(filter_mdata):
    mdata = filter_mdata.copy()
    mdata.mod["psm"].obs["group"] = ["A", "B"]
    filtered = add_filter(mdata, modality="psm", column="group", keep="eq", value="A", on="obs")

    applied = apply_filter(filtered, modality="psm")

    assert applied["psm"].obs_names.tolist() == ["s1"]
    assert applied["psm"].var_names.tolist() == ["v1", "v2", "v3"]
    assert applied["psm"].uns["decoy"].index.tolist() == ["v1", "v2", "v3"]


def test_apply_filter_all_columns_with_unknown_raises(filter_mdata):
    filtered = add_filter(filter_mdata, modality="psm", column="score", keep="gt", value=15.0, on="var")

    with pytest.raises(ValueError, match="No matching filter columns found"):
        apply_filter(filtered, modality="psm", columns=["not_existing"])


def test_apply_filter_prunes_closed_msmu_stream_handler(filter_mdata, capsys):
    filtered = add_filter(filter_mdata, modality="psm", column="score", keep="gt", value=15.0, on="var")
    logger = get_logger()
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    try:
        stream = io.StringIO()
        stale_handler = logging.StreamHandler(stream)
        stale_handler._msmu_handler = True  # type: ignore[attr-defined]
        logger.handlers = [stale_handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        stream.close()
        capsys.readouterr()

        applied = apply_filter(filtered, modality="psm", on="var")

        captured = capsys.readouterr()
        last_key = max(applied.uns["_cmd"], key=lambda x: int(x))
        assert stale_handler not in logger.handlers
        assert "--- Logging error ---" not in captured.err
        assert "Applying var filters for psm:" in applied.uns["_cmd"][last_key]["stdout"]
    finally:
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_apply_filter_prunes_closed_package_stream_handler(filter_mdata, capsys):
    filtered = add_filter(filter_mdata, modality="psm", column="score", keep="gt", value=15.0, on="var")
    logger = get_logger()
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    try:
        stream = io.StringIO()
        stale_handler = logging.StreamHandler(stream)
        logger.handlers = [stale_handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        stream.close()
        capsys.readouterr()

        applied = apply_filter(filtered, modality="psm", on="var")

        captured = capsys.readouterr()
        last_key = max(applied.uns["_cmd"], key=lambda x: int(x))
        assert stale_handler not in logger.handlers
        assert "--- Logging error ---" not in captured.err
        assert "Applying var filters for psm:" in applied.uns["_cmd"][last_key]["stdout"]
    finally:
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_apply_filter_prunes_closed_child_stream_handler(filter_mdata, capsys):
    filtered = add_filter(filter_mdata, modality="psm", column="score", keep="gt", value=15.0, on="var")
    logger = logging.getLogger("msmu._preprocessing._filter")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    try:
        stream = io.StringIO()
        stale_handler = logging.StreamHandler(stream)
        logger.handlers = [stale_handler]
        logger.setLevel(logging.INFO)
        logger.propagate = True
        stream.close()
        capsys.readouterr()

        applied = apply_filter(filtered, modality="psm", on="var")

        captured = capsys.readouterr()
        last_key = max(applied.uns["_cmd"], key=lambda x: int(x))
        assert stale_handler not in logger.handlers
        assert "--- Logging error ---" not in captured.err
        assert "Applying var filters for psm:" in applied.uns["_cmd"][last_key]["stdout"]
    finally:
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_apply_filter_does_not_emit_to_closed_root_stream_handler(filter_mdata, capsys):
    filtered = add_filter(filter_mdata, modality="psm", column="score", keep="gt", value=15.0, on="var")
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    try:
        stream = io.StringIO()
        stale_handler = logging.StreamHandler(stream)
        root_logger.handlers = [stale_handler]
        root_logger.setLevel(logging.INFO)
        stream.close()
        capsys.readouterr()

        applied = apply_filter(filtered, modality="psm", on="var")

        captured = capsys.readouterr()
        last_key = max(applied.uns["_cmd"], key=lambda x: int(x))
        assert "--- Logging error ---" not in captured.err
        assert "Applying var filters for psm:" in applied.uns["_cmd"][last_key]["stdout"]
    finally:
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)
