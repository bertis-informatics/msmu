from msmu._core._provenance import _event_inputs
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import warnings

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

import msmu as mm
from msmu._core import _sources
from msmu._provenance import compute_hash, get_log, options, replay

pytestmark = pytest.mark.filterwarnings("ignore:Replay environment may differ:UserWarning")


def workflow(tmp_path, hashing=True):
    source = tmp_path / "source.h5mu"
    md.MuData({"protein": ad.AnnData(
        X=np.array([[2., 4.], [8., 16.]]),
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame({"score": [1., 2.]}, index=["p1", "p2"]),
    )}).write_h5mu(source)
    with options(hashing=hashing):
        m = mm.read_h5mu(source)
        m = mm.pp.log2_transform(m, modality="protein")
        m = mm.pp.add_filter(m, modality="protein", column="score", keep="gt", value=1.)
        m = mm.pp.apply_filter(m, modality="protein")
    return source, m


@pytest.mark.parametrize("decoded", [False, True])
def test_replay_roundtrip_leaves_original_unchanged(tmp_path, decoded):
    _, original = workflow(tmp_path)
    before = get_log(original)
    with options(hashing=False):
        result = replay(before if decoded else original)
        assert mm.pv.get_options() == {"hashing": False, "significant_digits": 12}
    assert compute_hash(result) == compute_hash(original)
    assert get_log(original) == before
    assert [e["function"] for e in get_log(result)["events"]] == [e["function"] for e in before["events"]]
    assert get_log(result)["head"] != before["head"]


@pytest.mark.parametrize("old_module", ["msmu._preprocessing._meta", "msmu.dt", "msmu._data"])
def test_data_namespace_and_legacy_function_paths(tmp_path, old_module):
    _, m = workflow(tmp_path)
    with options(hashing=True):
        m = mm.dt.assign(m, "group", "old", modality="protein")
        m = mm.dt.map(m, source="protein.var", target="protein.var", columns={"group": "label"})
        m = mm.dt.replace(m, target="protein.var", columns={"label": {"old": "new"}})
        m = mm.dt.drop(m, target="protein.uns", key=next(iter(m["protein"].uns)))
    history = get_log(m)
    old_names = {"assign": "set_column", "map": "map_column", "replace": "replace_values", "drop": "drop_key"}
    for event in history["events"]:
        if event["function_path"].startswith("msmu._data."):
            event["function_path"] = old_module + "." + old_names[event["function"]]
            event["function"] = old_names[event["function"]]
    assert compute_hash(replay(history)) == compute_hash(m)
    script = mm.pv.to_script(history)
    for name in old_names:
        assert f"mm.dt.{name}(" in script
        assert not hasattr(mm.pp, name)
    namespace = {}
    exec(script, namespace)
    assert compute_hash(namespace["mdata"]) == compute_hash(m)


def test_replay_source_replacement(tmp_path):
    source, original = workflow(tmp_path)
    replacement = tmp_path / "moved.h5mu"
    source.rename(replacement)
    result = replay(original, sources={str(source): replacement})
    assert compute_hash(result) == compute_hash(original)
    assert get_log(result)["events"][0]["parameters"]["h5mu_file"]["path"] == str(replacement)


def test_hashless_replay_requires_explicit_opt_out(tmp_path):
    _, original = workflow(tmp_path, hashing=False)
    with pytest.raises(ValueError, match="verify=False"):
        replay(original)
    result = replay(original, verify=False)
    assert compute_hash(result) == compute_hash(original)
    assert all(not e["hashing"] for e in get_log(result)["events"])


def test_changed_original_file_fails_before_reader(tmp_path, monkeypatch):
    source, original = workflow(tmp_path)
    with source.open("ab") as handle:
        handle.write(b"changed")

    def forbidden(*args, **kwargs):
        pytest.fail("A changed input must be rejected before reading")

    monkeypatch.setattr(md, "read_h5mu", forbidden)
    with pytest.raises(ValueError, match="hash mismatch at read_h5mu"):
        replay(original)


def test_output_mismatch_stops_replay(tmp_path):
    _, original = workflow(tmp_path)
    history = get_log(original)
    history["events"][-1]["outputs"][0]["hash"]["value"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch at apply_filter"):
        replay(history)


@pytest.mark.parametrize("problem,match", [
    ("external", "does not support function"),
    ("merge", "producers do not match"),
    ("gap", "Unrecorded data changes"),
    ("data", "Input content is not stored"),
    ("callable", "non-replayable"),
    ("return", "one returned MuData"),
    ("cycle", "cyclic"),
])
def test_unsupported_history_fails_before_any_execution(tmp_path, monkeypatch, problem, match):
    _, original = workflow(tmp_path)
    history = get_log(original)
    event = history["events"][-1]
    if problem == "external":
        event["function_path"] = "os.system"
    elif problem == "merge":
        event["parents"].append(history["events"][0]["id"])
    elif problem == "gap":
        _event_inputs(event)[0]["hash"]["value"] = "0" * 64
    elif problem == "data":
        event["parameters"]["table"] = {"id": "table", "role": "arguments/table", "type": "DataFrame", "hash": {"status": "disabled"}}
    elif problem == "callable":
        event["parameters"]["on"] = {"type": "callable", "replayable": False}
    elif problem == "return":
        event["outputs"][0]["type"] = "DeaResult"
    elif problem == "cycle":
        event["parents"] = [event["id"]]

    def forbidden(*args, **kwargs):
        pytest.fail("Preflight must reject unsupported history before reading")

    monkeypatch.setattr(md, "read_h5mu", forbidden)
    with pytest.raises(ValueError, match=match):
        replay(history)


def test_environment_difference_warns(tmp_path, caplog):
    _, original = workflow(tmp_path)
    history = get_log(original)
    for env in history["environments"].values():
        env["python"] = "0.0.0"
    replay(history)
    assert "Replay environment may differ:" in caplog.text
    assert "python" in caplog.text


def test_real_diann_url_replay_downloads_once_per_call(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "data/diann_dia/diann/report.parquet"
    url = "https://example.test/report.parquet"
    sdrf_url = "https://example.test/meta.sdrf.tsv"
    payloads = {
        url: path.read_bytes(),
        sdrf_url: (path.parent.parent / "meta.sdrf.tsv").read_bytes(),
    }
    downloads = []

    def download(source, timeout):
        downloads.append(source)
        return BytesIO(payloads[source])

    monkeypatch.setattr(_sources, "urlopen", download)
    with options(hashing=True):
        original = mm.read_diann(url)
        original = mm.pp.attach_sdrf(original, sdrf_url, validate=False)
        original = mm.pp.apply_sdrf_to_obs(original)
        original = mm.pp.log2_transform(original, modality="psm")
    before = deepcopy(get_log(original))
    result = replay(original)
    assert downloads == [url, sdrf_url, url, sdrf_url]
    assert compute_hash(result) == compute_hash(original)
    assert get_log(original) == before
    assert _sources.get_download_buffer(url) is None
    assert _sources.get_download_buffer(sdrf_url) is None


def test_unseeded_umap_is_rejected_before_execution(tmp_path):
    _, original = workflow(tmp_path)
    history = get_log(original)
    last = history["events"][-1]
    last["function_path"] = f"{mm.tl.umap.__module__}.{mm.tl.umap.__qualname__}"
    last["function"] = "umap"
    last["parameters"] = {"mdata": last["parameters"]["mdata"], "modality": "protein", "random_state": None}
    with pytest.raises(ValueError, match="random_state"):
        replay(history)


def test_logging_warning_preserves_original_and_options(tmp_path, caplog):
    _, original = workflow(tmp_path)
    before = get_log(original)
    history = deepcopy(before)
    for env in history["environments"].values():
        env["python"] = "0.0.0"
    with warnings.catch_warnings(), options(hashing=False):
        warnings.filterwarnings("error", message="Replay environment")
        replay(history)
        assert "Replay environment" in caplog.text
        assert get_log(original) == before
        assert mm.pv.get_options() == {"hashing": False, "significant_digits": 12}


def test_script_roundtrip_and_source_replacement(tmp_path):
    source, original = workflow(tmp_path)
    before = get_log(original)
    replacement = tmp_path / "moved ' source.h5mu"
    source.rename(replacement)
    script = mm.pv.to_script(before, sources={str(source): replacement})
    assert all(key not in script for key in ("duration_seconds", "started_at", "ended_at"))
    assert "mm.pv.set_options(hashing=True)" in script
    assert "mm.pv.options(" not in script
    assert "mdata = mm.read_h5mu(" in script
    assert "mdata = mm.pp.log2_transform(mdata=mdata," in script
    namespace = {}
    with options(hashing=False):
        exec(compile(script, "generated_workflow.py", "exec"), namespace)
        assert mm.pv.get_options() == {"hashing": True, "significant_digits": 12}
    assert compute_hash(namespace["mdata"]) == compute_hash(original)
    assert get_log(original) == before
    assert all(event["hashing"] for event in get_log(namespace["mdata"])["events"])


def test_script_generation_needs_no_source_and_does_not_run(tmp_path):
    source, original = workflow(tmp_path)
    source.unlink()
    script = mm.pv.to_script(original)
    compile(script, "generated_workflow.py", "exec")
    with pytest.raises(FileNotFoundError):
        exec(script, {})


@pytest.mark.parametrize("target", ["source", "output"])
def test_script_hash_mismatch_stops_execution(tmp_path, target):
    source, original = workflow(tmp_path)
    history = get_log(original)
    if target == "source":
        with source.open("ab") as handle:
            handle.write(b"changed")
    else:
        history["events"][-1]["outputs"][0]["hash"]["value"] = "0" * 64
    script = mm.pv.to_script(history)
    with options(hashing=False):
        with pytest.raises(ValueError, match="hash mismatch"):
            exec(script, {})
        assert mm.pv.get_options() == {"hashing": True, "significant_digits": 12}


def test_script_hashless_opt_out(tmp_path):
    _, original = workflow(tmp_path, hashing=False)
    with pytest.raises(ValueError, match="verify=False"):
        mm.pv.to_script(original)
    namespace = {}
    with options(hashing=False):
        exec(mm.pv.to_script(original, verify=False), namespace)
        assert mm.pv.get_options()["hashing"] is True
    assert all(event["hashing"] for event in get_log(namespace["mdata"])["events"])
    assert compute_hash(namespace["mdata"]) == compute_hash(original)


def test_script_parameter_literals_are_safe():
    import datetime
    from msmu._core._replay import _literal

    value = {"quoted": "'); raise RuntimeError('injected') #", "date": datetime.date(2026, 9, 15),
             "tuple": (1, "x"), "infinity": float("inf")}
    assert eval(_literal(value), {"datetime": datetime}) == value
    with pytest.raises(ValueError, match="parameter type"):
        _literal(object())


def test_script_file_export(tmp_path):
    _, original = workflow(tmp_path)
    target = tmp_path / "재현.py"
    expected = mm.pv.to_script(original)
    for filename in (target, str(target)):
        assert mm.pv.to_script(original, filename=filename) is None
        assert target.read_text(encoding="utf-8") == expected
    namespace = {}
    exec(compile(target.read_text(encoding="utf-8"), str(target), "exec"), namespace)
    assert compute_hash(namespace["mdata"]) == compute_hash(original)
    invalid = get_log(original)
    invalid["events"][-1]["function_path"] = "os.system"
    with pytest.raises(ValueError, match="does not support function"):
        mm.pv.to_script(invalid, filename=target)
    assert target.read_text(encoding="utf-8") == expected


def test_plotting_leaves_verified_replay_intact(tmp_path):
    _, original = workflow(tmp_path)
    before = get_log(original)
    with options(hashing=True):
        assert mm.pl.plot_id(original, modality="protein").data
        assert mm.pl.plot_intensity(original, modality="protein").data
    assert get_log(original) == before
    assert compute_hash(replay(original)) == compute_hash(original)


def merged_workflow(tmp_path, hashing=True):
    path = tmp_path / "branches.h5mu"
    md.MuData({"protein": ad.AnnData(
        np.array([[2., 4.], [8., 16.]]),
        obs=pd.DataFrame({"group": ["G1", "G2"]}, index=["a", "b"]),
        var=pd.DataFrame(index=["p1", "p2"]),
    )}).write_h5mu(path)
    with options(hashing=hashing):
        base = mm.pp.log2_transform(mm.read_h5mu(path), modality="protein")
        branches = {}
        for key, group in [("G/1", "G1"), ("G'2", "G2")]:
            branch = mm.pp.add_filter(base, "protein", "group", "eq", group, on="obs")
            branches[key] = mm.pp.apply_filter(branch, "protein", on="obs")
        return mm.dt.concat(branches)


@pytest.mark.parametrize("hashing", [True, False])
def test_merge_replay_and_script_share_ancestor_once(tmp_path, monkeypatch, hashing):
    original = merged_workflow(tmp_path, hashing)
    before = get_log(original)
    refs = before["events"][-1]["parameters"]["mdatas"]
    assert set(refs) == {"G/1", "G'2"}
    assert {ref["source_event"] for ref in refs.values()} == set(before["events"][-1]["parents"])
    read = md.read_h5mu
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return read(*args, **kwargs)
    monkeypatch.setattr(md, "read_h5mu", counted)
    result = replay(original, verify=hashing)
    namespace = {}
    script = mm.pv.to_script(original, verify=hashing)
    assert "mm.dt.concat(" in script
    exec(script, namespace)
    assert len(calls) == 2  # One reader each, despite two branches.
    for output in (result, namespace["mdata"]):
        assert compute_hash(output) == compute_hash(original)
        log = get_log(output)
        assert len(log["events"]) == 7
        assert len(log["events"][-1]["parents"]) == 2
    assert get_log(original) == before


def test_legacy_linear_log_without_parameter_references(tmp_path):
    _, original = workflow(tmp_path)
    history = get_log(original)
    for event in history["events"]:
        event["inputs"] = deepcopy(_event_inputs(event))
        event["parameters"].pop("mdata", None)
        for entity in event["inputs"]:
            if "path" in entity:
                event["parameters"][entity["role"].split("/")[1]] = entity["path"]
    assert compute_hash(replay(history)) == compute_hash(original)
    assert compute_hash(script_result(history)) == compute_hash(original)


def test_invalid_branch_reference_fails_before_reading(tmp_path, monkeypatch):
    original = merged_workflow(tmp_path)
    history = get_log(original)
    history["events"][-1]["parameters"]["mdatas"]["G/1"]["source_event"] = history["events"][0]["id"]
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid references must fail before reading")
    monkeypatch.setattr(md, "read_h5mu", forbidden)
    for execute in (replay, mm.pv.to_script):
        with pytest.raises(ValueError, match="producers do not match"):
            execute(history)


def test_equal_hash_inputs_keep_distinct_producers(tmp_path):
    source = tmp_path / "empty.h5mu"
    md.MuData({"protein": ad.AnnData(np.empty((0, 2)))}).write_h5mu(source)
    left, right = mm.read_h5mu(source), mm.read_h5mu(source)
    assert compute_hash(left) == compute_hash(right)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        original = mm.dt.concat({"second": left, "first": right})
        history = get_log(original)
        refs = history["events"][-1]["parameters"]["mdatas"]
        assert refs["second"]["hash"]["value"] == refs["first"]["hash"]["value"]
        assert refs["second"]["source_event"] != refs["first"]["source_event"]
        for execute in (replay, lambda h: script_result(h)):
            result = execute(history)
            assert compute_hash(result) == compute_hash(original)
            assert len(get_log(result)["events"]) == 3
        legacy = deepcopy(history)
        legacy["events"][-1]["inputs"] = deepcopy(_event_inputs(legacy["events"][-1]))
        legacy["events"][-1]["parameters"].pop("mdatas")
        with pytest.raises(ValueError, match="Ambiguous input producer"):
            replay(legacy)


def script_result(history):
    namespace = {}
    exec(mm.pv.to_script(history), namespace)
    return namespace["mdata"]


def test_fork_inputs_are_isolated_from_mutating_calls(tmp_path, monkeypatch):
    from functools import wraps
    original_filter = mm.pp.add_filter

    @wraps(original_filter)
    def mutating_filter(mdata, *args, **kwargs):
        output = original_filter(mdata, *args, **kwargs)
        mdata["protein"].X[:] = -999
        return output

    monkeypatch.setattr(mm.pp, "add_filter", mutating_filter)
    path = tmp_path / "fork.h5mu"
    md.MuData({"protein": ad.AnnData(
        np.array([[1.], [2.]]), obs=pd.DataFrame({"group": ["a", "b"]}, index=["a", "b"])
    )}).write_h5mu(path)
    base = mm.read_h5mu(path)
    branches = {}
    for group in ("a", "b"):
        branch = mm.pp.add_filter(base.copy(), "protein", "group", "eq", group, on="obs")
        branches[group] = mm.pp.apply_filter(branch, "protein", on="obs")
    merged = mm.dt.concat(branches)
    assert compute_hash(replay(merged)) == compute_hash(merged)
    assert compute_hash(script_result(merged)) == compute_hash(merged)


@pytest.mark.parametrize("old_name", ["merge_mudata", "concat"])
def test_legacy_merge_name_replays_as_concat(tmp_path, old_name):
    original = merged_workflow(tmp_path)
    history = get_log(original)
    event = history["events"][-1]
    event["function"] = old_name
    event["function_path"] = f"msmu._read_write._reader_utils.{old_name}"
    result = replay(history)
    script = mm.pv.to_script(history)
    assert "mm.dt.concat(" in script
    namespace = {}
    exec(script, namespace)
    assert compute_hash(result) == compute_hash(original)
    assert compute_hash(namespace["mdata"]) == compute_hash(original)



def test_filter_history_before_optional_name_still_replays(tmp_path):
    _, original = workflow(tmp_path)
    history = get_log(original)
    for event in history["events"]:
        if event["function"] == "add_filter":
            del event["parameters"]["name"]
    assert compute_hash(replay(history, verify=True)) == compute_hash(original)
    namespace = {}
    exec(mm.pv.to_script(history, verify=True), namespace)
    assert compute_hash(namespace["mdata"]) == compute_hash(original)
