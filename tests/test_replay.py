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
    for name in mm.dt.__all__:
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
    assert get_log(result)["events"][0]["parameters"]["h5mu_file"] == str(replacement)


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
    ("merge", "merged histories"),
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
        event["inputs"][0]["hash"]["value"] = "0" * 64
    elif problem == "data":
        event["inputs"].append({"role": "arguments/table", "type": "DataFrame"})
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
    last["parameters"] = {"modality": "protein", "random_state": None}
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
    assert "mdata = mm.read_h5mu(" in script
    assert "mdata = mm.pp.log2_transform(mdata=mdata," in script
    namespace = {}
    with options(hashing=False):
        exec(compile(script, "generated_workflow.py", "exec"), namespace)
        assert mm.pv.get_options() == {"hashing": False, "significant_digits": 12}
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
        assert mm.pv.get_options() == {"hashing": False, "significant_digits": 12}


def test_script_hashless_opt_out(tmp_path):
    _, original = workflow(tmp_path, hashing=False)
    with pytest.raises(ValueError, match="verify=False"):
        mm.pv.to_script(original)
    namespace = {}
    exec(mm.pv.to_script(original, verify=False), namespace)
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
