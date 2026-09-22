from msmu._provenance import get_log
import io
import logging

import pandas as pd
import pytest

from msmu.logging_utils import get_logger
from msmu._preprocessing._filter import _mask_boolean_filter, add_filter, apply_filter


def test_mask_boolean_filter_ops():
    series = pd.Series(["a", "b", "aa"])
    assert _mask_boolean_filter(series, "contains", "a").tolist() == [True, False, True]
    assert _mask_boolean_filter(series, "not_contains", "b").tolist() == [
        True,
        False,
        True,
    ]
    assert _mask_boolean_filter(pd.Series([1, 2, 3]), "gt", 1).tolist() == [
        False,
        True,
        True,
    ]


def test_mask_boolean_filter_invalid():
    with pytest.raises(ValueError, match="Unknown filter operator"):
        _mask_boolean_filter(pd.Series([1, 2, 3]), "nope", 1)


def test_add_filter_and_apply_filter_with_decoy(filter_mdata):
    filtered = add_filter(filter_mdata, modality="psm", column="score", keep="gt", value=15.0)
    assert "filter" in filtered["psm"].varm.keys()
    assert filtered["psm"].varm["filter"].shape[1] == 1

    applied = apply_filter(filtered, modality="psm")
    assert applied["psm"].var_names.tolist() == ["v2", "v3"]
    assert applied["psm"].uns["decoy"].index.tolist() == ["v2", "v3"]


def test_add_filter_on_obs_stores_in_obsm(filter_mdata):
    mdata = filter_mdata.copy()
    mdata.mod["psm"].obs["group"] = ["A", "B"]
    out = add_filter(mdata, modality="psm", column="group", keep="eq", value="A", on="obs")
    assert "filter" in out["psm"].obsm.keys()
    assert out["psm"].obsm["filter"].shape[1] == 1
    assert out["psm"].obsm["filter"].iloc[:, 0].tolist() == [True, False]


def test_add_filter_preserves_order_without_duplicates(filter_mdata):
    mdata = filter_mdata
    for value in [15.0, 25.0, 15.0]:
        mdata = add_filter(mdata, modality="psm", column="score", keep="gt", value=value)
    assert mdata["psm"].uns["filter"] == ["score_gt_15.0", "score_gt_25.0"]


def test_add_filter_on_obsm_with_key_stores_in_obsm(filter_mdata):
    mdata = filter_mdata.copy()
    mdata.mod["psm"].obsm["qc"] = pd.DataFrame({"score": [0.1, 0.9]}, index=mdata.mod["psm"].obs_names)
    out = add_filter(mdata, modality="psm", column="score", keep="gt", value=0.5, on="obsm", key="qc")
    assert "filter" in out["psm"].obsm.keys()
    assert out["psm"].obsm["filter"].iloc[:, 0].tolist() == [False, True]


def test_add_filter_requires_key_for_obsm(filter_mdata):
    with pytest.raises(ValueError, match="key must be provided"):
        add_filter(
            filter_mdata,
            modality="psm",
            column="score",
            keep="gt",
            value=0.5,
            on="obsm",
        )


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

    entry = get_log(applied)["events"][-1]
    assert entry["function"] == "apply_filter"
    assert entry["parameters"]["columns"] == ["score_gt_15.0"]


def test_apply_filter_logs_filter_columns_to_console(filter_mdata, caplog):
    caplog.set_level(logging.INFO, logger="msmu")
    filtered = add_filter(filter_mdata, modality="psm", column="score", keep="gt", value=15.0, on="var")
    applied = apply_filter(filtered, modality="psm", on="var")

    entry = get_log(applied)["events"][-1]
    assert "stdout" not in entry
    assert "Applying var filters for psm:" in caplog.text
    assert "score_gt_15.0" in caplog.text


def test_apply_filter_all_with_var_filters_does_not_warn_for_missing_obs(filter_mdata, caplog):
    caplog.set_level(logging.INFO, logger="msmu")
    filtered = add_filter(filter_mdata, modality="psm", column="score", keep="gt", value=15.0)
    filtered = add_filter(filtered, modality="psm", column="score", keep="lt", value=25.0)

    applied = apply_filter(filtered, modality="psm")

    assert applied["psm"].var_names.tolist() == ["v2"]
    assert "Applying var filters for psm:" in caplog.text
    assert "obsm['filter']" not in caplog.text


def test_apply_filter_all_with_obs_filter_and_decoy_does_not_require_var_decoy_filter(
    filter_mdata,
):
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

        assert stale_handler not in logger.handlers
        assert "--- Logging error ---" not in captured.err
        assert "stdout" not in get_log(applied)["events"][-1]
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

        assert stale_handler not in logger.handlers
        assert "--- Logging error ---" not in captured.err
        assert "stdout" not in get_log(applied)["events"][-1]
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

        assert stale_handler not in logger.handlers
        assert "--- Logging error ---" not in captured.err
        assert "stdout" not in get_log(applied)["events"][-1]
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

        assert "--- Logging error ---" not in captured.err
        assert "stdout" not in get_log(applied)["events"][-1]
    finally:
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
@pytest.mark.parametrize("keep", ["eq", "ne", "lt", "le", "gt", "ge"])
def test_numeric_filters_reject_missing_values(dtype, keep):
    values = pd.Series([1., float("nan"), 3.], dtype=dtype)
    mask = _mask_boolean_filter(values, keep, 2.)
    assert not mask.iloc[1]
    assert not mask.isna().any()


@pytest.mark.parametrize("dtype", [object, "string"])
@pytest.mark.parametrize("keep", ["contains", "not_contains"])
def test_string_filters_reject_missing_values(dtype, keep):
    mask = _mask_boolean_filter(pd.Series(["abc", None, "xyz"], dtype=dtype), keep, "a")
    assert mask.tolist() == ([True, False, False] if keep == "contains" else [False, False, True])


def test_filter_missing_values_in_targets_and_decoys(filter_mdata):
    for table in (filter_mdata["psm"].var, filter_mdata["psm"].uns["decoy"]):
        table["score"] = pd.array([10., None, 30.], dtype="Float64")
    out = add_filter(filter_mdata, "psm", "score", "gt", 15.)
    out = apply_filter(out, "psm")
    assert out["psm"].var_names.tolist() == ["v3"]
    assert out["psm"].uns["decoy"].index.tolist() == ["v3"]


@pytest.mark.parametrize("axis, matrix", [("var", "varm"), ("obs", "obsm")])
def test_matrix_filters_do_not_overwrite_other_sources(filter_mdata, axis, matrix):
    mdata = filter_mdata.copy()
    mdata["psm"].uns.pop("decoy")
    table = getattr(mdata["psm"], axis)
    table["score"] = [10.] + [30.] * (len(table) - 1)
    for key in ("qc", "other"):
        getattr(mdata["psm"], matrix)[key] = pd.DataFrame(
            {"score": [30.] + [10.] * (len(table) - 1)}, index=table.index
        )
    out = add_filter(mdata, "psm", "score", "gt", 15., on=axis)
    for key in ("qc", "other"):
        out = add_filter(out, "psm", "score", "gt", 15., on=matrix, key=key)
    masks = getattr(out["psm"], matrix)["filter"]
    assert len(masks.columns) == 3
    selected = apply_filter(out, "psm", on=axis, columns=[f"{matrix}['qc'].score_gt_15.0"])
    assert getattr(selected["psm"], axis).index.tolist() == [table.index[0]]
    assert len(getattr(apply_filter(out, "psm", on=axis)["psm"], axis)) == 0


def test_apply_filter_rejects_invalid_axis(filter_mdata):
    with pytest.raises(ValueError, match="Unknown filter axis"):
        apply_filter(filter_mdata, "psm", on="vars")


@pytest.mark.parametrize("second_columns", [None, ["score_lt_28.0"]])
def test_sequential_filters_preserve_decoy_conditions(filter_mdata, second_columns):
    out = add_filter(filter_mdata, "psm", "score", "gt", 15.)
    out = add_filter(out, "psm", "score", "lt", 28.)
    first = apply_filter(out, "psm", on="var", columns=["score_gt_15.0"])
    assert first["psm"].uns["decoy_filter"].columns.tolist() == out["psm"].varm["filter"].columns.tolist()
    sequential = apply_filter(first, "psm", on="var", columns=second_columns)
    together = apply_filter(out, "psm", on="var")
    assert sequential["psm"].var_names.tolist() == together["psm"].var_names.tolist() == ["v2"]
    pd.testing.assert_frame_equal(sequential["psm"].uns["decoy"], together["psm"].uns["decoy"])
    assert sequential["psm"].uns["decoy"].index.tolist() == ["v2"]


@pytest.mark.parametrize("axis", ["var", "obs", "all"])
def test_missing_requested_filter_does_not_apply_partial_set(filter_mdata, axis):
    out = add_filter(filter_mdata, "psm", "score", "gt", 15.)
    out = add_filter(out, "psm", "score", "lt", 28.)
    if axis == "obs":
        out["psm"].obs["group"] = ["a", "b"]
        out = add_filter(out, "psm", "group", "eq", "a", on="obs")
        name = "group_eq_a"
    else:
        name = "score_gt_15.0"
    before = get_log(out)
    with pytest.raises(ValueError, match="Filter columns not found"):
        apply_filter(out, "psm", on=axis, columns=[name, "missing"])
    assert get_log(out) == before
    assert out["psm"].shape == (2, 3)


def test_missing_decoy_condition_raises(filter_mdata):
    out = add_filter(filter_mdata, "psm", "score", "gt", 15.)
    out = add_filter(out, "psm", "score", "lt", 28.)
    out["psm"].uns["decoy_filter"] = out["psm"].uns["decoy_filter"].drop(columns="score_lt_28.0")
    with pytest.raises(ValueError, match="Decoy filter columns not found"):
        apply_filter(out, "psm", on="var")


@pytest.mark.parametrize("filter_name", [None, "qc_score"])
def test_updated_filter_workflow_replays_and_generates_script(filter_mdata, tmp_path, filter_name):
    import msmu as mm

    source = tmp_path / "filters.h5mu"
    filter_mdata["psm"].obs["name"] = pd.array(["sample", None], dtype="string")
    filter_mdata["psm"].obsm["qc"] = pd.DataFrame({"score": [1., 0.]}, index=filter_mdata["psm"].obs_names)
    filter_mdata.write_h5mu(source)
    with mm.pv.options(hashing=True):
        original = mm.read_h5mu(source)
        before_hash, before_log = mm.pv.compute_hash(original), get_log(original)
        out = add_filter(original, "psm", "score", "gt", 15.)
        out = add_filter(out, "psm", "score", "lt", 28.)
        out = add_filter(out, "psm", "name", "not_contains", "blank", on="obs")
        out = add_filter(out, "psm", "score", "gt", .5, on="obsm", key="qc", name=filter_name)
        out = apply_filter(out, "psm", on="var", columns=["score_gt_15.0"])
        out = apply_filter(out, "psm")
    assert mm.pv.compute_hash(original) == before_hash
    assert get_log(original) == before_log
    assert out["psm"].shape == (1, 1)
    assert out["psm"].uns["decoy"].index.tolist() == ["v2"]
    assert mm.pv.compute_hash(mm.pv.replay(out, verify=True)) == mm.pv.compute_hash(out)
    namespace = {}
    exec(mm.pv.to_script(out, verify=True), namespace)
    assert mm.pv.compute_hash(namespace["mdata"]) == mm.pv.compute_hash(out)


def test_concat_undefined_masks_remain_unapplied(filter_mdata):
    import msmu as mm

    filter_mdata["psm"].obs["group"] = ["a", "b"]
    branches = {}
    for group in ("a", "b"):
        branch = add_filter(filter_mdata, "psm", "group", "eq", group, on="obs")
        branches[group] = apply_filter(branch, "psm", on="obs")
    merged = mm.dt.concat(branches)
    assert merged["psm"].obsm["filter"].isna().any().any()
    out = apply_filter(merged, "psm", on="obs")
    assert out["psm"].obs_names.tolist() == ["s1", "s2"]


def test_named_filter_selection_reuse_and_conflicts(filter_mdata):
    out = add_filter(filter_mdata, "psm", "score", "gt", 15., name="qc_score")
    repeated = add_filter(out, "psm", "score", "gt", 15., name="qc_score")
    assert repeated["psm"].varm["filter"].columns.tolist() == ["qc_score"]
    selected = apply_filter(repeated, "psm", columns=["qc_score"])
    assert selected["psm"].var_names.tolist() == ["v2", "v3"]
    assert selected["psm"].uns["decoy"].index.tolist() == ["v2", "v3"]
    assert out["psm"].uns["filter_conditions"]["qc_score"]["value"] == 15.
    before = get_log(out)
    with pytest.raises(ValueError, match="different condition"):
        add_filter(out, "psm", "score", "gt", 25., name="qc_score")
    assert get_log(out) == before
    pd.testing.assert_frame_equal(out["psm"].varm["filter"], repeated["psm"].varm["filter"])
    assert "filter_conditions" not in filter_mdata["psm"].uns


def test_named_filter_does_not_collide_with_other_axes_or_automatic_names(filter_mdata):
    out = add_filter(filter_mdata, "psm", "score", "gt", 15., name="score_gt_25.0")
    with pytest.raises(ValueError, match="different condition"):
        add_filter(out, "psm", "score", "gt", 25.)
    out["psm"].obs["score"] = [10., 20.]
    with pytest.raises(ValueError, match="different condition"):
        add_filter(out, "psm", "score", "gt", 15., on="obs", name="score_gt_25.0")
    auto = add_filter(filter_mdata, "psm", "score", "gt", 15.)
    with pytest.raises(ValueError, match="without a recorded condition"):
        add_filter(auto, "psm", "score", "gt", 25., name="score_gt_15.0")


@pytest.mark.parametrize("name", ["", "  ", 1, "qc/score"])
def test_named_filter_rejects_invalid_names(filter_mdata, name):
    with pytest.raises(ValueError, match="name must"):
        add_filter(filter_mdata, "psm", "score", "gt", 15., name=name)


def test_named_matrix_filter_h5mu_roundtrip(filter_mdata, tmp_path):
    import msmu as mm

    filter_mdata["psm"].obsm["qc"] = pd.DataFrame({"score": [1., 0.]}, index=filter_mdata["psm"].obs_names)
    out = add_filter(filter_mdata, "psm", "score", "gt", .01, on="obsm", key="qc", name="qc_score")
    out = add_filter(out, "psm", "score", "gt", 15., name="quality")
    path = tmp_path / "named.h5mu"
    out.write_h5mu(path)
    restored = mm.read_h5mu(path)
    assert restored["psm"].uns["filter_conditions"] == out["psm"].uns["filter_conditions"]
    restored = add_filter(restored, "psm", "score", "gt", 15., name="quality")
    with pytest.raises(ValueError, match="different condition"):
        add_filter(restored, "psm", "score", "gt", 25., name="quality")
    selected = apply_filter(restored, "psm", columns=["qc_score", "quality"])
    assert selected["psm"].shape == (1, 2)
