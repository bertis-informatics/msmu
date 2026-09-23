from copy import deepcopy
import json

import mudata as md
from packaging.requirements import Requirement
import pytest

import msmu as mm
from msmu._core import _provenance as core


def recorded_history():
    return {
        "schema_version": 1,
        "head": "event-a",
        "events": [{"id": "event-a", "environment_id": "env-a"}],
        "environments": {"env-a": {
            "python": "3.12.8", "implementation": "CPython", "os": "Linux", "architecture": "x86_64",
            "packages": {"NumPy": "2.2.0", "msmu": "0.3.1", "scikit_learn": "1.6.0"},
            "msmu_source": {"commit": "unavailable", "dirty": False},
            "thread_settings": {"OMP_NUM_THREADS": "2"},
        }},
    }


@pytest.mark.parametrize("format", ["uv", "conda"])
def test_export_uses_persisted_environment_without_live_inspection(tmp_path, monkeypatch, mdata, format):
    environment = recorded_history()["environments"]["env-a"]
    monkeypatch.setattr(core, "_environment", lambda: deepcopy(environment))

    @mm.pv.log
    def identity(mdata):
        return mdata

    original = identity(mdata)
    before = mm.pv.get_log(original)
    path = tmp_path / "analysis.h5mu"
    original.write_h5mu(path)
    loaded = md.read_h5mu(path)

    def forbidden():
        raise AssertionError("Export must not inspect the current environment")

    monkeypatch.setattr(core, "_environment", forbidden)
    text = mm.pv.to_env(loaded, format=format)
    assert text == mm.pv.to_env(before, format=format)
    assert mm.pv.get_log(original) == before
    assert mm.pv.get_log(loaded) == before
    destination = tmp_path / ("requirements.txt" if format == "uv" else "environment.yml")
    assert mm.pv.to_env(loaded, destination, format=format) is None
    assert destination.read_text() == text
    if format == "uv":
        requirements = [line for line in text.splitlines() if line and not line.startswith("#")]
        assert "uv venv --python 3.12.8" in text
    else:
        yaml = pytest.importorskip("yaml")
        document = yaml.safe_load(text)
        assert document["channels"] == ["conda-forge"]
        assert document["dependencies"][:2] == ["python=3.12.8", "pip"]
        requirements = document["dependencies"][2]["pip"]
    assert [str(Requirement(item)) for item in requirements] == [
        "msmu==0.3.1", "numpy==2.2.0", "scikit-learn==1.6.0",
    ]


def test_multiple_environments_require_selection_only_when_installation_differs():
    history = recorded_history()
    second = deepcopy(history["environments"]["env-a"])
    second["thread_settings"] = {"OMP_NUM_THREADS": "4"}
    history["environments"]["env-b"] = second
    history["events"].append({"id": "event-b", "environment_id": "env-b"})
    text = mm.pv.to_env(history)
    assert '"env-a","env-b"' in text
    assert '"OMP_NUM_THREADS":"2"' in text and '"OMP_NUM_THREADS":"4"' in text
    second["packages"]["NumPy"] = "2.3.0"
    with pytest.raises(ValueError, match="select environment_id"):
        mm.pv.to_env(history)
    assert "numpy==2.2.0" in mm.pv.to_env(history, environment_id="env-a")
    assert "numpy==2.3.0" in mm.pv.to_env(history, environment_id="env-b")
    # A persisted but unreferenced snapshot must not override the recorded calls.
    history["events"].pop()
    assert "numpy==2.2.0" in mm.pv.to_env(history)


@pytest.mark.parametrize("change, message", [
    ({"python": "3.12\n--index-url https://example.invalid"}, "Python version"),
    ({"implementation": "PyPy"}, "CPython"),
    ({"packages": {}}, "package versions"),
    ({"packages": {"numpy\n--extra-index-url": "2.2.0"}}, "package name"),
    ({"packages": {"numpy": "2.2.0\n-r other.txt"}}, "version"),
    ({"packages": {"numpy": "2.*"}}, "version"),
    ({"packages": {"NumPy": "2.2.0", "numpy": "2.3.0"}}, "Conflicting"),
])
def test_invalid_recordings_do_not_overwrite_files(tmp_path, change, message):
    history = recorded_history()
    history["environments"]["env-a"].update(change)
    destination = tmp_path / "requirements.txt"
    destination.write_text("preserve me\n")
    with pytest.raises(ValueError, match=message):
        mm.pv.to_env(history, destination)
    assert destination.read_text() == "preserve me\n"


def test_missing_and_unknown_environments_and_format():
    history = recorded_history()
    with pytest.raises(ValueError, match="format"):
        mm.pv.to_env(history, format="uv.lock")
    with pytest.raises(ValueError, match="Missing recorded environment"):
        mm.pv.to_env(history, environment_id="unknown")
    history["events"][0]["environment_id"] = "missing"
    with pytest.raises(ValueError, match="Missing recorded environment"):
        mm.pv.to_env(history)
    with pytest.raises(ValueError, match="No recorded environments"):
        mm.pv.to_env(md.MuData({}))


def test_dirty_source_warns_and_context_cannot_inject_requirements(caplog):
    history = recorded_history()
    history["environments"]["env-a"]["msmu_source"]["dirty"] = True
    history["environments"]["env-a"]["os"] = "Linux\n--index-url https://example.invalid"
    before = deepcopy(history)
    text = mm.pv.to_env(history)
    assert "uncommitted changes" in caplog.text
    assert not any(line.startswith("--") for line in text.splitlines())
    context = next(line.removeprefix("# Recorded context: ") for line in text.splitlines()
                   if line.startswith("# Recorded context: "))
    assert json.loads(context)["os"] == history["environments"]["env-a"]["os"]
    assert history == before
