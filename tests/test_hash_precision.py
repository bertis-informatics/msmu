import pandas as pd
import numpy as np
import pytest
from scipy import sparse

from msmu._provenance import compute_hash, get_options, options


def test_precision_preserves_small_values_and_original_arrays():
    left = np.array([0.002418128635502926, 1e-20, 1e-50, np.nan, np.inf, -np.inf, -0.0])
    right = left.copy()
    right[0] = np.nextafter(right[0], np.inf)
    original = left.tobytes()
    assert compute_hash(left) == compute_hash(right)
    assert compute_hash(left, significant_digits=None) != compute_hash(right, significant_digits=None)
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
    assert first == compute_hash(path, significant_digits=None)
    path.write_text('0.0024181286355029266')
    assert first != compute_hash(path)
    assert compute_hash(['a', 1]) == compute_hash(['a', 1], significant_digits=None)


def test_precision_settings_restore_and_validate():
    before = get_options()
    with options(hashing=True, significant_digits=None):
        assert get_options()['significant_digits'] is None
    assert get_options() == before
    for invalid in [0, 16, True, 12.5, '12']:
        with pytest.raises(ValueError):
            compute_hash(1.0, significant_digits=invalid)


def test_exact_and_rounded_replay(tmp_path):
    import msmu as mm
    import mudata as md
    import anndata as ad

    source = tmp_path / 'source.h5mu'
    md.MuData({'protein': ad.AnnData(np.array([[2., 4.]]))}).write_h5mu(source)
    for precision in [None, 12]:
        with options(hashing=True, significant_digits=precision):
            original = mm.read_h5mu(source)
            original = mm.pp.log2_transform(original, modality='protein')
        history = mm.pv.get_log(original)
        output_hash = history['events'][-1]['outputs'][0]['hash']
        assert output_hash.get('significant_digits') == precision
        replayed = mm.pv.replay(original)
        assert compute_hash(replayed, significant_digits=precision) == compute_hash(original, significant_digits=precision)
        namespace = {}
        exec(mm.pv.to_script(original), namespace)
        assert compute_hash(namespace['mdata'], significant_digits=precision) == compute_hash(original, significant_digits=precision)


def test_rounding_chunks_and_array_layout_do_not_change_hash():
    values = np.linspace(-1e-30, 1e30, 131074).reshape(2, -1)
    assert compute_hash(values) == compute_hash(np.asfortranarray(values))
    complex_values = values.astype(complex) * (1 + 2j)
    assert compute_hash(complex_values) == compute_hash(np.asfortranarray(complex_values))
