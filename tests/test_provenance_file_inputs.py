from io import BytesIO

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

import msmu as mm
from msmu._core import _sources
from msmu._core._provenance import _event_inputs


def _data():
    return md.MuData({"protein": ad.AnnData(
        np.array([[2.], [4.], [8.]]),
        obs=pd.DataFrame(index=["s1", "s2", "s3"]),
        var=pd.DataFrame(index=["p1"]),
    )})


def test_declared_nested_file_inputs_keep_arguments_and_ordinary_text_unchanged(tmp_path):
    first, second = tmp_path / "first.tsv", tmp_path / "second.tsv"
    first.write_text("first")
    second.write_text("second")
    sources = {"left": [str(first)], "right": (str(second),)}

    @mm.pv.log(file_inputs=("sources",))
    def annotate(mdata, sources, metadata):
        assert sources is original_sources
        assert all(isinstance(values[0], str) for values in sources.values())
        assert metadata == "ordinary text"
        return mdata

    original_sources = sources
    result = annotate(_data(), sources, metadata="ordinary text")
    event = mm.pv.get_log(result)["events"][-1]
    files = {entity["role"]: entity for entity in _event_inputs(event) if "path" in entity}
    assert set(files) == {"arguments/sources/left/0", "arguments/sources/right/0"}
    for role, path in (("left", first), ("right", second)):
        entity = files[f"arguments/sources/{role}/0"]
        assert entity["path"] == str(path)
        assert entity["hash"]["value"] == mm.pv.compute_hash(path)
    assert event["parameters"]["metadata"] == "ordinary text"


@pytest.mark.parametrize("source_kind", ["local-string", "url"])
@pytest.mark.parametrize("function,argument,content,kwargs", [
    (mm.pp.add_meta, "metadata", "sample\tgroup\ns1\tA\ns2\tB\ns3\tB\n", {"metadata_on": "sample"}),
    (mm.pp.attach_sdrf, "sdrf", "source name\tcomment[data file]\ns1\trun1.raw\n", {"validate": False}),
    (mm.io.add_quant, "quant_data", "Sequence\tIntensity_s1\tIntensity_s2\tIntensity_s3\nAA\t1\t2\t3\n",
     {"quant_tool": "flashlfq"}),
], ids=["metadata", "sdrf", "flashlfq"])
def test_file_input_provenance_verifies_relocated_sources(
    tmp_path, monkeypatch, function, argument, content, kwargs, source_kind
):
    source = tmp_path / "source.h5mu"
    _data().write_h5mu(source)
    table = tmp_path / "input.tsv"
    table.write_text(content)
    location = str(table)
    downloads = []
    if source_kind == "url":
        location = "https://example.test/input.tsv"

        def download(url, timeout):
            downloads.append(url)
            return BytesIO(content.encode())

        monkeypatch.setattr(_sources, "urlopen", download)

    with mm.pv.options(hashing=True):
        original = mm.pv.log(mm.read_h5mu)(source)
        original = function(original, **{argument: location}, **kwargs)
    event = mm.pv.get_log(original)["events"][-1]
    entity = event["parameters"][argument]
    assert entity["role"] == f"arguments/{argument}"
    assert entity["path"] == location
    assert entity["hash"]["status"] == "completed"
    assert entity["hash"]["value"] == mm.pv.compute_hash(table)
    assert downloads == ([location] if source_kind == "url" else [])
    if function is mm.pp.attach_sdrf:
        assert original.uns["sdrf"].attrs == {}

    digest = mm.pv.compute_hash(original)
    moved = tmp_path / "moved.tsv"
    table.rename(moved)
    replacements = {location: str(moved)}
    replayed = mm.pv.replay(original, sources=replacements)
    script = mm.pv.to_script(original, sources=replacements)
    namespace = {}
    exec(script, namespace)
    for result in (replayed, namespace["mdata"]):
        assert mm.pv.compute_hash(result) == digest
        assert mm.pv.get_log(result)["events"][-1]["parameters"][argument]["path"] == str(moved)

    with moved.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match=f"hash mismatch at {function.__name__}"):
        mm.pv.replay(original, sources=replacements)
    with pytest.raises(ValueError, match=f"hash mismatch at {function.__name__}"):
        exec(script, {})
