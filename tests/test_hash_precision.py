import pandas as pd
import numpy as np
import pytest
from scipy import sparse
import anndata as ad
import mudata as md

import msmu as mm
from msmu._provenance import compute_hash, get_options, options, set_options
from msmu._core._hashing import compute_hash as legacy_hash
from msmu._core._provenance import options as legacy_options


def test_precision_preserves_small_values_and_original_arrays():
    left = np.array([0.002418128635502926, 1e-20, 1e-50, np.nan, np.inf, -np.inf, -0.0])
    right = left.copy()
    right[0] = np.nextafter(right[0], np.inf)
    original = left.tobytes()
    assert compute_hash(left) == compute_hash(right)
    assert legacy_hash(left, significant_digits=None) != legacy_hash(right, significant_digits=None)
    assert compute_hash(np.array([1e-20])) != compute_hash(np.array([1e-50]))
    assert compute_hash(pd.Index(left[:1])) != compute_hash(pd.Index(right[:1]))
    assert left.tobytes() == original
    assert compute_hash(sparse.csr_matrix(left)) == compute_hash(sparse.csr_matrix(right))


def test_precision_handles_extremes_and_rounding_boundary():
    for value in [np.nextafter(0.0, 1.0), np.finfo(float).max, 1e-308, -1e-308]:
        assert compute_hash(value) != compute_hash(0.0)
    assert compute_hash(9.9999999999999) == compute_hash(10.0)
    assert compute_hash(-9.9999999999999) == compute_hash(-10.0)
    # Quantization is not an allclose test: neighboring values may straddle a boundary.
    assert compute_hash(1.2345678901249) != compute_hash(1.2345678901251)
    assert compute_hash(np.array([1.0])) != compute_hash(np.array([1.0], dtype=np.float32))


def test_files_and_non_numeric_values_remain_exact(tmp_path):
    path = tmp_path / 'input.txt'
    path.write_text('0.002418128635502926')
    first = compute_hash(path)
    assert first == legacy_hash(path, significant_digits=None)
    path.write_text('0.0024181286355029266')
    assert first != compute_hash(path)
    assert compute_hash(['a', 1]) == legacy_hash(['a', 1], significant_digits=None)


def test_public_hash_precision_is_fixed():
    before = get_options()
    assert before == {"hashing": True}
    for precision in [None, 6, 12]:
        with pytest.raises(TypeError, match="significant_digits"):
            compute_hash(1.0, significant_digits=precision)
        with pytest.raises(TypeError, match="significant_digits"):
            set_options(hashing=True, significant_digits=precision)
        with pytest.raises(TypeError, match="significant_digits"):
            options(hashing=True, significant_digits=precision)
    assert get_options() == before


def test_legacy_v1_string_list_digest_remains_unchanged():
    metadata = {"labels": ["alpha", "β", ""]}
    mdata = md.MuData({"protein": ad.AnnData(np.array([[2., 4.]]))})
    mdata.uns.update(metadata)
    assert legacy_hash(metadata, normalization="msmu-v1") == "d48f0c93db40ae8fb79eda2908de289eae321f9dc6fde23440a02c76435562a2"
    assert legacy_hash(mdata, normalization="msmu-v1") == "bb835cc48578924c529c703c783710b713fb5c9685500bbbad65704e77055230"


@pytest.mark.parametrize("normalization,precision", [
    (None, None), ("significant-digits-v1", 6), ("significant-digits-v1", 12),
    ("msmu-v1", 12), ("msmu-v2", 12), ("msmu-v3", 12),
])
def test_exact_and_rounded_replay(tmp_path, normalization, precision):
    source = tmp_path / 'source.h5mu'
    md.MuData({'protein': ad.AnnData(np.array([[2., 4.]]))}).write_h5mu(source)
    with legacy_options(hashing=True, significant_digits=precision, normalization=normalization):
        original = mm.pv.log(mm.read_h5mu)(source)
        original = mm.pp.log2_transform(original, modality='protein')
    history = mm.pv.get_log(original)
    for event in history['events']:
        info = event['outputs'][0]['hash']
        assert info.get('normalization') == normalization
        if normalization == 'significant-digits-v1':
            assert info['significant_digits'] == precision
        else:
            assert 'significant_digits' not in info
    replayed = mm.pv.replay(original)
    expected = legacy_hash(original, significant_digits=precision, normalization=normalization)
    assert legacy_hash(replayed, significant_digits=precision, normalization=normalization) == expected
    namespace = {}
    script = mm.pv.to_script(original)
    if normalization == "msmu-v3":
        assert 'significant_digits=' not in script
    exec(script, namespace)
    assert get_options() == {'hashing': True}
    assert legacy_hash(namespace['mdata'], significant_digits=precision, normalization=normalization) == expected


@pytest.fixture
def policy_source(tmp_path):
    source = tmp_path / "source.h5mu"
    mdata = md.MuData({"protein": ad.AnnData(
        X=np.array([[2., 4.], [8., 16.]]),
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame({"score": [1., 2.]}, index=["p1", "p2"]),
    )})
    mdata.uns["labels"] = ["alpha", "β", ""]
    mdata["protein"].uns.update(identification_file="old/report.tsv", quantification_file="old/quant.tsv")
    mdata.write_h5mu(source)
    return source


@pytest.mark.parametrize("before,after", [
    ("msmu-v1", "msmu-v2"), ("msmu-v2", "msmu-v1"),
    ("msmu-v1", "msmu-v3"), ("msmu-v3", "msmu-v1"),
    ("msmu-v2", "msmu-v3"), ("msmu-v3", "msmu-v2"),
])
def test_mixed_fixed_policy_replay_and_script(policy_source, before, after):
    with legacy_options(hashing=True, normalization=before):
        original = mm.pv.log(mm.read_h5mu)(policy_source)
        original = mm.pp.log2_transform(original, modality="protein")
    with legacy_options(hashing=True, normalization=after):
        original = mm.pp.add_filter(original, "protein", "score", "gt", 1.)
        original = mm.pp.apply_filter(original, "protein")
    history = mm.pv.get_log(original)
    assert [event["outputs"][0]["hash"]["normalization"] for event in history["events"]] == [
        before, before, after, after,
    ]
    assert all("significant_digits" not in event["outputs"][0]["hash"] for event in history["events"])
    assert compute_hash(mm.pv.replay(original)) == compute_hash(original)
    namespace = {}
    exec(mm.pv.to_script(original), namespace)
    assert compute_hash(namespace["mdata"]) == compute_hash(original)
    for rerun in (mm.pv.replay, mm.pv.to_script):
        with pytest.raises(ValueError, match="change of hash precision"):
            rerun(original, verify=False)


@pytest.mark.parametrize("before,after", [
    ("msmu-v1", "msmu-v2"), ("msmu-v2", "msmu-v1"),
    ("msmu-v1", "msmu-v3"), ("msmu-v3", "msmu-v1"),
    ("msmu-v2", "msmu-v3"), ("msmu-v3", "msmu-v2"),
])
@pytest.mark.parametrize("edit", ["metadata", "numeric"])
def test_policy_transition_still_rejects_unrecorded_edits(policy_source, before, after, edit):
    with legacy_options(hashing=True, normalization=before):
        original = mm.pv.log(mm.read_h5mu)(policy_source)
        original = mm.pp.log2_transform(original, modality="protein")
    if edit == "metadata":
        original.uns["labels"] = ["changed", "β", ""]
    else:
        original["protein"].X[0, 0] += 1
    with legacy_options(hashing=True, normalization=after):
        original = mm.pp.add_filter(original, "protein", "score", "gt", 1.)
    with pytest.raises(ValueError, match="hash mismatch at add_filter"):
        mm.pv.replay(original)
    script = mm.pv.to_script(original)
    with pytest.raises(ValueError, match="hash mismatch at add_filter"):
        exec(script, {})


@pytest.mark.parametrize("normalization", ["msmu-v1", "msmu-v2", "msmu-v3"])
@pytest.mark.parametrize("precision", [None, 12])
def test_fixed_policy_metadata_rejects_explicit_precision(policy_source, normalization, precision):
    with legacy_options(hashing=True, normalization=normalization):
        original = mm.pv.log(mm.read_h5mu)(policy_source)
    history = mm.pv.get_log(original)
    history["events"][0]["outputs"][0]["hash"]["significant_digits"] = precision
    for rerun in (mm.pv.replay, mm.pv.to_script):
        with pytest.raises(ValueError, match=f"{normalization} has fixed precision"):
            rerun(history)


@pytest.mark.parametrize("before,after", [
    ((None, None), ("significant-digits-v1", 6)),
    (("significant-digits-v1", 6), ("significant-digits-v1", 12)),
    (("msmu-v1", 12), (None, None)),
])
@pytest.mark.parametrize("verify", [False, True])
def test_legacy_precision_transitions_remain_unsupported(policy_source, before, after, verify):
    with legacy_options(hashing=True, normalization=before[0], significant_digits=before[1]):
        original = mm.pv.log(mm.read_h5mu)(policy_source)
    with legacy_options(hashing=True, normalization=after[0], significant_digits=after[1]):
        original = mm.pp.log2_transform(original, modality="protein")
    for rerun in (mm.pv.replay, mm.pv.to_script):
        with pytest.raises(ValueError, match="change of hash precision"):
            rerun(original, verify=verify)


def test_rounding_chunks_and_array_layout_do_not_change_hash():
    values = np.linspace(-1e-30, 1e30, 131074).reshape(2, -1)
    assert compute_hash(values) == compute_hash(np.asfortranarray(values))
    complex_values = values.astype(complex) * (1 + 2j)
    assert compute_hash(complex_values) == compute_hash(np.asfortranarray(complex_values))
