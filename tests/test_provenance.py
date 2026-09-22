from msmu._core._provenance import _event_inputs
from pathlib import Path
import warnings
from types import SimpleNamespace

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import msmu as mm
from msmu._core import _provenance as core
from msmu._provenance import compute_hash, get_log, options


def data():
    a = ad.AnnData(
        X=sparse.csr_matrix(np.array([[1.0, 0.0, 3.0], [4.0, 5.0, 0.0]])),
        obs=pd.DataFrame({"group": ["a", "a"]}, index=["s1", "s2"]),
        var=pd.DataFrame({"label": ["a", "a", "b"]}, index=["v1", "v2", "v3"]),
    )
    return md.MuData({"protein": a})


@mm.pv.log
def identity(mdata, value=10, text="default"):
    return mdata


def test_hash_off_does_not_visit_content(monkeypatch):
    def forbidden(_):
        raise AssertionError("Hashing must not run")

    monkeypatch.setattr(core, "compute_hash", forbidden)
    with options(hashing=False):
        m = identity(data(), text="a" * 2000, value={str(i): i for i in range(40)})
    log = get_log(m)
    event = log["events"][0]
    assert "status" not in event
    assert "error" not in event
    assert event["parameters"]["text"] == "a" * 2000
    assert len(event["parameters"]["value"]) == 40
    assert "inputs" not in event
    assert event["parameters"]["mdata"]["source_event"] == ""
    assert _event_inputs(event)[0]["hash"] == {"status": "disabled"}
    assert log["environments"][event["environment_id"]]["packages"]["pandas"] == pd.__version__
    assert "_cmd" not in m.uns


@pytest.mark.parametrize("hashes", [False, True])
def test_data_parameters_are_only_recorded_as_entities(hashes):
    m = data()
    frame = pd.DataFrame({"x": [1, 2]})
    with options(hashing=hashes):
        identity(m, value={"frame": frame, "datasets": [m], "method": "median",
                           "columns": ["x"], "empty": [], "layer": None})
    event = get_log(m)["events"][-1]
    inputs = {entity["role"]: entity for entity in _event_inputs(event)}
    def reference(role, kind):
        return inputs[role]
    assert event["parameters"] == {
        "mdata": reference("arguments/mdata", "MuData"),
        "value": {"frame": reference("arguments/value/frame", "DataFrame"),
                  "datasets": [reference("arguments/value/datasets/0", "MuData")],
                  "method": "median", "columns": ["x"], "empty": [], "layer": None},
        "text": "default",
    }
    assert inputs["arguments/value/frame"]["type"] == "DataFrame"
    assert "inputs" not in event
    assert set(inputs["arguments/value/frame"]) == {"id", "role", "type", "hash"}
    assert inputs["arguments/mdata"]["source_event"] == ""
    assert inputs["arguments/value/datasets/0"]["type"] == "MuData"
    assert all(entity["hash"]["status"] == ("completed" if hashes else "disabled")
               for entity in inputs.values())
    assert event["outputs"][0]["type"] == "MuData"


def test_inplace_hashes_and_failed_mutation():
    m = data()
    original = compute_hash(m)
    error = ValueError("computation failed")

    @mm.pv.log
    def mutate(mdata, fail=False):
        mdata["protein"].X.data[0] += 1
        if fail:
            raise error
        return mdata

    with options(hashing=True):
        mutate(m)
        event = get_log(m)["events"][0]
        assert _event_inputs(event)[0]["hash"]["value"] == original
        assert event["outputs"][0]["hash"]["value"] == compute_hash(m) != original
        before_failure = get_log(m)
        with pytest.raises(ValueError) as caught:
            mutate(m, fail=True)
        assert caught.value is error
    assert get_log(m) == before_failure
    assert compute_hash(m) != event["outputs"][0]["hash"]["value"]
    mutate(m)
    assert len(get_log(m)["events"]) == 2
    assert get_log(m)["events"][-1]["parents"] == [event["id"]]

    fresh = data()
    with pytest.raises(ValueError):
        mutate(fresh, fail=True)
    assert "_log" not in fresh.uns



def test_nested_copy_history_and_environment():
    m = identity(data())
    before = get_log(m)

    @mm.pv.log
    def outer(mdata):
        return identity(mdata.copy())

    out = outer(m)
    assert get_log(m) == before
    log = get_log(out)
    assert [e["function"] for e in log["events"]] == ["identity", "outer"]
    assert log["events"][-1]["parents"] == [before["head"]]
    assert len(log["environments"]) == 1
    with np.errstate(over="raise"):
        identity(out)
    assert len(get_log(out)["environments"]) == 2
    log["events"][0]["function"] = "edited"
    assert get_log(out)["events"][0]["function"] == "identity"


def test_environment_collects_all_installed_packages_once(monkeypatch):
    calls = []

    def installed():
        calls.append(True)
        return [SimpleNamespace(metadata={"Name": "custom-analysis-addon"}, version="1.2.3")]

    monkeypatch.setattr(core, "distributions", installed)
    core._base_environment.cache_clear()
    try:
        assert core._environment()["packages"] == {"custom-analysis-addon": "1.2.3"}
        core._environment()
        assert len(calls) == 1
    finally:
        core._base_environment.cache_clear()


def test_h5mu_roundtrip_and_read_continuation(tmp_path):
    m = data()
    with options(hashing=True):
        identity(m)
    before = get_log(m)
    digest = compute_hash(m)
    path = tmp_path / "data.h5mu"
    m.write_h5mu(path)
    loaded = md.read_h5mu(path)
    assert get_log(loaded) == before
    assert compute_hash(m) == compute_hash(loaded) == digest
    continued = mm.read_h5mu(path)
    assert len(get_log(continued)["events"]) == 2
    assert get_log(continued)["events"][-1]["parents"] == [before["head"]]


def test_sparse_never_densifies_and_metadata_changes(monkeypatch):
    m = data()

    def forbidden(*args, **kwargs):
        raise AssertionError("No sparse densification")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbidden)
    first = compute_hash(m)
    copy = m.copy()
    copy.uns["_log"] = {"anything": "ignored"}
    assert compute_hash(copy) == first
    copy["protein"].var.iloc[0, 0] = "changed"
    assert compute_hash(copy) != first
    copy = m.copy()
    copy["protein"].layers["raw"] = copy["protein"].X.copy()
    assert compute_hash(copy) != first
    assert compute_hash({"a": {"b": "c"}}) != compute_hash({"a": {}, "b": "c"})
    assert compute_hash(np.array([complex(np.nan, 1)])) != compute_hash(np.array([complex(np.nan, 2)]))


def test_unavailable_hash_is_not_successful_verification():
    m = data()
    m.uns["unsupported"] = object()
    with options(hashing=True):
        identity(m)
    event = get_log(m)["events"][0]
    assert "error" not in event
    assert "status" not in event
    assert _event_inputs(event)[0]["hash"]["status"] == "unavailable"
    assert "value" not in _event_inputs(event)[0]["hash"]


def test_file_hash_snapshot_and_options_restore(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("before")
    digest = compute_hash(path)

    @mm.pv.log
    def reader(input_file):
        Path(input_file).write_text("after")
        return data()

    with options(hashing=True):
        with options(hashing=False):
            assert identity(data()).uns["_log"]
        out = reader(str(path))
    event = get_log(out)["events"][0]
    assert _event_inputs(event)[0]["hash"]["value"] == digest != compute_hash(path)
    assert get_log(identity(data()))["events"][0]["hashing"] is True
    with pytest.raises(TypeError):
        mm.pv.set_options(hashing="yes")


def test_set_options_and_temporary_hashing_restore():
    assert mm.pv.get_options() == {"hashing": True, "significant_digits": 12}
    mm.pv.set_options(hashing=True)
    try:
        snapshot = mm.pv.get_options()
        snapshot["hashing"] = False
        assert mm.pv.get_options() == {"hashing": True, "significant_digits": 12}
        with pytest.raises(ValueError):
            with options(hashing=False):
                assert mm.pv.get_options() == {"hashing": False, "significant_digits": 12}
                assert get_log(identity(data()))["events"][-1]["hashing"] is False
                raise ValueError("stop")
        assert get_log(identity(data()))["events"][-1]["hashing"] is True
        assert mm.pv.get_options() == {"hashing": True, "significant_digits": 12}
        with pytest.raises(TypeError):
            with options(hashing="yes"):
                pass
        mm.pv.set_options(hashing=False)
        assert get_log(identity(data()))["events"][-1]["hashing"] is False
    finally:
        mm.pv.set_options(hashing=True)
    assert get_log(identity(data()))["events"][-1]["hashing"] is True
    assert mm.pv.get_options() == {"hashing": True, "significant_digits": 12}


def test_merge_histories_have_distinct_entities_and_shared_parent():
    base = identity(data())
    left, right = identity(base.copy()), identity(base.copy())

    @mm.pv.log
    def combine(mdatas):
        return mdatas["left"].copy()

    result = combine({"left": left, "right": right})
    log = get_log(result)
    assert len(log["events"]) == 4
    last = log["events"][-1]
    assert len(last["parents"]) == 2
    assert len({v["id"] for v in _event_inputs(last) + last["outputs"]}) == 3
    assert len(get_log(left)["events"]) == 2


def test_hash_string_missing_values_and_ordered_categories(tmp_path):
    m = data()
    m["protein"].var["label"] = ["a", None, "a"]
    original = compute_hash(m)
    m.write_h5mu(tmp_path / "missing.h5mu")
    assert compute_hash(md.read_h5mu(tmp_path / "missing.h5mu")) == original
    first = pd.Series(pd.Categorical(["a", "b"], categories=["a", "b"], ordered=True))
    second = pd.Series(pd.Categorical(["a", "b"], categories=["b", "a"], ordered=True))
    assert compute_hash(first) != compute_hash(second)


def test_real_merge_preserves_both_histories(mdata_factory):
    left = identity(mdata_factory("a"))
    right = identity(mdata_factory("b"))
    before = (compute_hash(left), compute_hash(right))
    merged = mm.merge_mudata({"a": left, "b": right})
    assert (compute_hash(left), compute_hash(right)) == before
    log = get_log(merged)
    assert len(log["events"]) == 3
    assert set(log["events"][-1]["parents"]) == {get_log(left)["head"], get_log(right)["head"]}


def test_dea_result_is_recorded_on_input(mdata):
    with options(hashing=True):
        result = mm.tl.run_de(
            mdata, modality="protein", category="group", ctrl="A", expr="B", stat_method="welch", n_resamples=10
        )
    event = get_log(mdata)["events"][-1]
    assert event["function"] == "run_de"
    output = event["outputs"][0]
    assert output["role"] == "return/DeaResult"
    assert output["hash"]["value"] == compute_hash(vars(result))


def test_concurrent_calls_keep_histories_independent(capsys):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(2)

    @mm.pv.log
    def noisy(mdata, label):
        print(f"start-{label}")
        barrier.wait(timeout=10)
        print(f"end-{label}")
        return mdata

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda label: noisy(data(), label), ["first", "second"]))
    for label, result in zip(["first", "second"], results):
        event = get_log(result)["events"][0]
        assert event["parameters"]["label"] == label
        assert "stdout" not in event
    output = capsys.readouterr().out
    for label in ("first", "second"):
        assert f"start-{label}" in output
        assert f"end-{label}" in output


@pytest.mark.parametrize("prior_hashing,current_hashing,changed,expect_warning", [
    (True, True, True, True),
    (True, True, False, False),
    (False, True, True, False),
    (True, False, True, False),
])
def test_warns_before_execution_on_unrecorded_change(
    monkeypatch, caplog, prior_hashing, current_hashing, changed, expect_warning
):
    with options(hashing=prior_hashing):
        m = identity(data())
    head = get_log(m)["head"]
    if changed:
        m["protein"].var["label"] = ["changed", "a", "b"]
    calls = []
    original_hash = core.compute_hash

    def counted_hash(value, **kwargs):
        calls.append(value)
        return original_hash(value, **kwargs)

    monkeypatch.setattr(core, "compute_hash", counted_hash)

    @mm.pv.log
    def next_step(mdata):
        assert any("Data changed outside" in item.message for item in caplog.records) == expect_warning
        return mdata

    with options(hashing=current_hashing):
        result = next_step(m)
    if expect_warning:
        warning = next(item for item in caplog.records if "Data changed outside" in item.message)
        assert warning.levelname == "WARNING"
        assert "identity → next_step" in warning.message
    assert get_log(result)["events"][-1]["parents"] == [head]
    assert len(calls) == (2 if current_hashing else 0)


def test_ambiguous_previous_outputs_do_not_warn(caplog):
    with options(hashing=True):
        m = identity(data())
        m["protein"].var["label"] = ["changed", "a", "b"]
        head = m.uns["_log"]["head"]
        import json
        event = json.loads(m.uns["_log"]["events"][head])
        event["outputs"] *= 2
        m.uns["_log"]["events"][head] = json.dumps(event)
        identity(m)
    assert not any("Data changed outside" in item.message for item in caplog.records)


def test_unrecorded_change_logs_even_when_python_warnings_are_errors(caplog):
    with options(hashing=True):
        m = identity(data())
        m["protein"].var["label"] = ["changed", "a", "b"]
        before = get_log(m)["head"]
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = identity(m)
        assert "Data changed outside" in caplog.text
        assert get_log(result)["events"][-1]["parents"] == [before]
