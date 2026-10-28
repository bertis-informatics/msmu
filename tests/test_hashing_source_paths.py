import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

import msmu as mm
from msmu._core._hashing import compute_hash as legacy_hash
from msmu._core._provenance import options as legacy_options

from test_read_write_sage_parity import _clean_labels, _clean_scan_tokens, _write_pair


def _data():
    return md.MuData({"protein": ad.AnnData(np.array([[2., 4.]]))})


@pytest.mark.parametrize("key", ["identification_file", "quantification_file"])
@pytest.mark.parametrize("global_uns", [False, True], ids=["modality", "global"])
def test_source_paths_are_preserved_but_excluded_from_data_hash(key, global_uns):
    mdata = _data()
    owner = mdata if global_uns else mdata["protein"]
    before = mm.pv.compute_hash(mdata)
    owner_before = mm.pv.compute_hash(owner)

    for path in ("old/results.tsv", "/moved/results.tsv", "https://example.test/results.tsv"):
        owner.uns[key] = path
        assert mm.pv.compute_hash(mdata) == before
        assert mm.pv.compute_hash(owner) == owner_before
        assert owner.uns[key] == path


@pytest.mark.parametrize("location", ["mapping", "nested-uns", "obs", "var"])
@pytest.mark.parametrize("key", ["identification_file", "quantification_file"])
def test_source_path_exclusions_are_limited_to_top_level_uns(key, location):
    mdata = _data()
    if location == "mapping":
        value = {}
        target = value
    elif location == "nested-uns":
        value = mdata
        target = mdata["protein"].uns.setdefault("metadata", {})
    else:
        value = mdata
        target = getattr(mdata["protein"], location)
    target[key] = "old/results.tsv"
    before = mm.pv.compute_hash(value)
    target[key] = "moved/results.tsv"
    assert mm.pv.compute_hash(value) != before


@pytest.mark.parametrize("edit", ["X", "metadata"])
def test_source_path_exclusions_still_detect_data_and_other_metadata_edits(edit):
    mdata = _data()
    mdata["protein"].uns["identification_file"] = "old/results.tsv"
    before = mm.pv.compute_hash(mdata)
    if edit == "X":
        mdata["protein"].X[0, 0] += 1
    else:
        mdata["protein"].uns["search_engine"] = "changed"
    assert mm.pv.compute_hash(mdata) != before


@pytest.mark.parametrize("policy,expected", [
    ("msmu-v1", "7e1ae221ef481cba3750a076e8618876a0a8b6d3a84cf295be2ae930b8a0b057"),
    ("msmu-v2", "e8411d1bfe3213d382de04544269da465f119178a6832562e4486df8543f12a5"),
])
def test_historical_hash_policies_retain_source_paths(policy, expected):
    mdata = _data()
    mdata["protein"].uns.update(identification_file="old/report.tsv", quantification_file="old/quant.tsv",
                                labels=["alpha", "β", ""])
    assert legacy_hash(mdata, normalization=policy) == expected
    mdata["protein"].uns["identification_file"] = "moved/report.tsv"
    assert legacy_hash(mdata, normalization=policy) != expected


@pytest.fixture
def sage_sources(tmp_path):
    return _write_pair(tmp_path, _clean_scan_tokens(), _clean_labels())


@pytest.mark.parametrize("drop_search_result", [False, True])
def test_search_reader_replay_and_script_verify_relocated_sources(sage_sources, tmp_path, drop_search_result):
    identification, quantification = sage_sources
    with pd.option_context("future.infer_string", False), mm.pv.options(hashing=True):
        original = mm.read_sage(identification, label="tmt", quantification_file=quantification,
                                drop_search_result=drop_search_result)
        original = mm.pp.log2_transform(original, modality="psm")
        digest = mm.pv.compute_hash(original)
        replacements = {}
        for source in sage_sources:
            moved = tmp_path / f"moved-{source.name}"
            source.rename(moved)
            replacements[str(source)] = moved

        replayed = mm.pv.replay(original, sources=replacements)
        namespace = {}
        exec(mm.pv.to_script(original, sources=replacements), namespace)

    for result in (replayed, namespace["mdata"]):
        assert mm.pv.compute_hash(result) == digest
        for key, source in (("identification_file", identification), ("quantification_file", quantification)):
            assert str(replacements[str(source)]) in result["psm"].uns[key]
            assert str(source) in original["psm"].uns[key]

    with replacements[str(quantification)].open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="hash mismatch at read_sage"):
        mm.pv.replay(original, sources=replacements)
    with pytest.raises(ValueError, match="hash mismatch at read_sage"):
        exec(mm.pv.to_script(original, sources=replacements), {})


def test_historical_v2_search_reader_replay_and_script(sage_sources):
    identification, quantification = sage_sources
    with pd.option_context("future.infer_string", False), legacy_options(hashing=True, normalization="msmu-v2"):
        original = mm.read_sage(identification, label="tmt", quantification_file=quantification)
        original = mm.pp.log2_transform(original, modality="psm")
        digest = legacy_hash(original, normalization="msmu-v2")
        assert legacy_hash(mm.pv.replay(original), normalization="msmu-v2") == digest
        namespace = {}
        exec(mm.pv.to_script(original), namespace)
        assert legacy_hash(namespace["mdata"], normalization="msmu-v2") == digest
