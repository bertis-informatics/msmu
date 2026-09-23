import pandas as pd
import numpy as np
import pytest
from scipy import sparse

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


def test_exact_and_rounded_replay(tmp_path):
    import msmu as mm
    import mudata as md
    import anndata as ad

    source = tmp_path / 'source.h5mu'
    md.MuData({'protein': ad.AnnData(np.array([[2., 4.]]))}).write_h5mu(source)
    for normalization, precision in [(None, None), ("significant-digits-v1", 6), ("significant-digits-v1", 12), ("msmu-v1", 12)]:
        with legacy_options(hashing=True, significant_digits=precision, normalization=normalization):
            original = mm.read_h5mu(source)
            original = mm.pp.log2_transform(original, modality='protein')
        history = mm.pv.get_log(original)
        output_hash = history['events'][-1]['outputs'][0]['hash']
        assert output_hash.get('normalization') == normalization
        assert output_hash.get('significant_digits') == (precision if normalization == 'significant-digits-v1' else None)
        replayed = mm.pv.replay(original)
        assert legacy_hash(replayed, significant_digits=precision, normalization=normalization) == legacy_hash(original, significant_digits=precision, normalization=normalization)
        namespace = {}
        script = mm.pv.to_script(original)
        if normalization == "msmu-v1":
            assert 'significant_digits=' not in script
        exec(script, namespace)
        assert get_options() == {'hashing': True}
        assert legacy_hash(namespace['mdata'], significant_digits=precision, normalization=normalization) == legacy_hash(original, significant_digits=precision, normalization=normalization)


def test_rounding_chunks_and_array_layout_do_not_change_hash():
    values = np.linspace(-1e-30, 1e30, 131074).reshape(2, -1)
    assert compute_hash(values) == compute_hash(np.asfortranarray(values))
    complex_values = values.astype(complex) * (1 + 2j)
    assert compute_hash(complex_values) == compute_hash(np.asfortranarray(complex_values))
