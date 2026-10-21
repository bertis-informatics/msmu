import copy
import logging

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

import msmu as mm
from msmu._core._provenance import options as legacy_options


def _data():
    return md.MuData({
        "protein": ad.AnnData(
            X=np.array([[2., 4., 8.], [4., 8., 16.], [3., 6., 12.], [6., 12., 24.]]),
            obs=pd.DataFrame({"group": pd.Categorical(["a", "a", "b", "b"])},
                             index=["s1", "s2", "s3", "s4"]),
            var=pd.DataFrame({"score": [1, 2, 3]}, index=["p1", "p2", "p3"]),
        ),
    })


def _roundtrip(mdata, tmp_path):
    path = tmp_path / "metadata.h5mu"
    mdata.write_h5mu(path)
    return md.read_h5mu(path)


def test_filter_name_list_hash_survives_h5mu(tmp_path):
    mdata = _data()
    mdata["protein"].uns["filter"] = ["score_ge_2"]
    digest = mm.pv.compute_hash(mdata)
    modality_digest = mm.pv.compute_hash(mdata["protein"])

    loaded = _roundtrip(mdata, tmp_path)

    assert isinstance(loaded["protein"].uns["filter"], np.ndarray)
    assert loaded["protein"].uns["filter"].tolist() == ["score_ge_2"]
    assert mm.pv.compute_hash(loaded) == digest
    assert mm.pv.compute_hash(loaded["protein"]) == modality_digest


@pytest.mark.parametrize("values", [
    ["first", "second"], [1, 2], [1.25, 2.5], [True, False], [], [[1, 2], [3, 4]],
], ids=["strings", "integers", "floats", "booleans", "empty", "matrix"])
def test_nested_global_and_modality_uns_lists_survive_h5mu(values, tmp_path):
    mdata = _data()
    for owner in (mdata, mdata["protein"]):
        owner.uns["metadata"] = {"nested": {"values": copy.deepcopy(values)}}
    digests = [mm.pv.compute_hash(owner) for owner in (mdata, mdata["protein"])]

    loaded = _roundtrip(mdata, tmp_path)

    for owner, digest in zip((loaded, loaded["protein"]), digests):
        restored = owner.uns["metadata"]["nested"]["values"]
        assert isinstance(restored, np.ndarray)
        np.testing.assert_array_equal(restored, np.asarray(values))
        assert mm.pv.compute_hash(owner) == digest


def test_filter_normalisation_pipeline_can_resume_after_h5mu(tmp_path, caplog):
    with mm.pv.options(hashing=True):
        mdata = mm.pp.add_filter(_data(), modality="protein", column="score", keep="ge", value=2,
                                 name="quality")
        mdata = mm.pp.apply_filter(mdata, modality="protein", on="var", columns=["quality"])
        mdata = mm.pp.normalise(mdata, method="median", modality="protein", group_obs="group")
        digest = mm.pv.compute_hash(mdata)
        history = mm.pv.get_log(mdata)
        blocks = mdata["protein"].uns["normalisation"]["blocks"]
        assert len(blocks) == 2
        assert all(isinstance(block["samples"], list) for block in blocks.values())

        loaded = _roundtrip(mdata, tmp_path)
        restored_blocks = loaded["protein"].uns["normalisation"]["blocks"]
        for key, block in blocks.items():
            assert isinstance(restored_blocks[key]["samples"], np.ndarray)
            assert restored_blocks[key]["samples"].tolist() == block["samples"]
        assert mm.pv.compute_hash(loaded) == digest
        assert mm.pv.get_log(loaded) == history

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="msmu"):
            resumed = mm.pp.normalise(loaded, method="median_center", modality="protein", group_obs="group")

    assert "Data changed outside the recorded workflow" not in caplog.text
    assert mm.pv.get_log(resumed)["events"][-1]["parents"] == [history["head"]]


@pytest.mark.parametrize("edit", ["value", "order", "X"])
def test_real_edits_after_h5mu_still_change_hash_and_warn(edit, tmp_path, caplog):
    mdata = _data()
    mdata.uns["labels"] = ["first", "second"]
    with mm.pv.options(hashing=True):
        mdata = mm.pp.add_filter(mdata, modality="protein", column="score", keep="ge", value=2)
        loaded = _roundtrip(mdata, tmp_path)
        digest = mm.pv.compute_hash(loaded)
        if edit == "value":
            loaded.uns["labels"][0] = "changed"
        elif edit == "order":
            loaded.uns["labels"] = loaded.uns["labels"][::-1].copy()
        else:
            loaded["protein"].X[0, 0] += 1
        assert mm.pv.compute_hash(loaded) != digest

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="msmu"):
            mm.pp.apply_filter(loaded, modality="protein", on="var")

    assert "Data changed outside the recorded workflow" in caplog.text


def test_hashing_uns_lists_does_not_mutate_metadata():
    mdata = _data()
    values = {"labels": ["a", "b"], "matrix": [[1, 2], [3, 4]], "array": np.array([1., np.nan])}
    for owner in (mdata, mdata["protein"]):
        owner.uns["metadata"] = copy.deepcopy(values)
    metadata = [owner.uns["metadata"] for owner in (mdata, mdata["protein"])]
    labels = [item["labels"] for item in metadata]
    array_bytes = [item["array"].tobytes() for item in metadata]

    mm.pv.compute_hash(mdata)

    for item, original_labels, original_bytes in zip(metadata, labels, array_bytes):
        assert item["labels"] is original_labels
        assert item["labels"] == values["labels"]
        assert isinstance(item["matrix"], list) and isinstance(item["matrix"][0], list)
        assert item["matrix"] == values["matrix"]
        assert item["array"].tobytes() == original_bytes


@pytest.mark.parametrize("replacement", [np.array([1., 2.], dtype=np.float32), np.array([[1., 2.]])],
                         ids=["dtype", "shape"])
def test_metadata_array_dtype_and_shape_remain_significant(replacement, tmp_path):
    mdata = _data()
    mdata["protein"].uns["values"] = np.array([1., 2.])
    loaded = _roundtrip(mdata, tmp_path)
    digest = mm.pv.compute_hash(loaded)
    loaded["protein"].uns["values"] = replacement

    assert mm.pv.compute_hash(loaded) != digest


def test_list_normalisation_is_limited_to_uns():
    assert mm.pv.compute_hash([1, 2]) != mm.pv.compute_hash(np.array([1, 2]))
    assert mm.pv.compute_hash({"values": [1, 2]}) != mm.pv.compute_hash({"values": np.array([1, 2])})


@pytest.mark.parametrize("edit", [None, "metadata", "X"], ids=["unchanged", "metadata-edit", "numeric-edit"])
def test_v1_to_v2_continuation_checks_old_hash_before_running(edit, caplog):
    mdata = _data()
    mdata.uns["labels"] = ["first", "second"]
    with legacy_options(hashing=True, normalization="msmu-v1"):
        mdata = mm.pp.log2_transform(mdata, modality="protein")
    previous = mm.pv.get_log(mdata)
    if edit == "metadata":
        mdata.uns["labels"][0] = "changed"
    elif edit == "X":
        mdata["protein"].X[0, 0] += 1
    warnings_at_entry = []

    @mm.pv.log
    def continue_workflow(mdata):
        warnings_at_entry.append("Data changed outside the recorded workflow" in caplog.text)
        return mdata

    caplog.clear()
    with mm.pv.options(hashing=True), caplog.at_level(logging.WARNING, logger="msmu"):
        result = continue_workflow(mdata)

    assert warnings_at_entry == [edit is not None]
    event = mm.pv.get_log(result)["events"][-1]
    assert event["parents"] == [previous["head"]]
    assert event["outputs"][0]["hash"]["normalization"] == "msmu-v2"


def test_loaded_v1_list_history_is_not_silently_relabelled(tmp_path, caplog):
    mdata = _data()
    mdata.uns["labels"] = ["first", "second"]
    with legacy_options(hashing=True, normalization="msmu-v1"):
        mdata = mm.pp.log2_transform(mdata, modality="protein")
    loaded = _roundtrip(mdata, tmp_path)
    assert isinstance(loaded.uns["labels"], np.ndarray)
    warnings_at_entry = []

    @mm.pv.log
    def continue_workflow(mdata):
        warnings_at_entry.append("Data changed outside the recorded workflow" in caplog.text)
        return mdata

    caplog.clear()
    with mm.pv.options(hashing=True), caplog.at_level(logging.WARNING, logger="msmu"):
        result = continue_workflow(loaded)

    assert warnings_at_entry == [True]
    events = mm.pv.get_log(result)["events"]
    assert events[0]["outputs"][0]["hash"]["normalization"] == "msmu-v1"
    assert events[-1]["outputs"][0]["hash"]["normalization"] == "msmu-v2"


@pytest.mark.parametrize("values", [
    [[1], [2, 3]], [None, {"sample": "s1"}], np.array([{"sample": "s1"}, None], dtype=object),
], ids=["ragged-list", "object-list", "object-array"])
def test_non_array_uns_metadata_remains_hashable_without_mutation(values):
    mdata = _data()
    mdata["protein"].uns["values"] = values
    original = copy.deepcopy(values)

    digest = mm.pv.compute_hash(mdata)

    assert mm.pv.compute_hash(mdata) == digest
    assert mdata["protein"].uns["values"] is values
    if isinstance(values, np.ndarray):
        np.testing.assert_array_equal(values, original)
    else:
        assert values == original
