import numpy as np
import pandas as pd
import pytest

from msmu._core import _hashing

pa = pytest.importorskip("pyarrow")


def legacy_hash(value, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(_hashing, "pa", None)
        return _hashing.compute_hash(value)


@pytest.mark.parametrize("values", [
    [], [None, None], ["", "a", "한글", "🙂", "a\x00b", None, "tail"],
    ["long" * 10000, None, "", "x"],
])
def test_string_storage_and_chunking_preserve_existing_hash(values, monkeypatch):
    original = pd.Series(values, dtype=object)
    expected = legacy_hash(original, monkeypatch)
    for dtype in ("string[python]", "string[pyarrow]", "str"):
        value = pd.Series(values, dtype=dtype)
        assert _hashing.compute_hash(value) == expected
        assert _hashing.compute_hash(value) == legacy_hash(value, monkeypatch)
    arrow = pa.array(["discard", *values, "discard"], type=pa.large_string()).slice(1, len(values))
    # Slicing and repartitioning must not hash bytes outside the logical values.
    chunks = pa.chunked_array(
        [arrow.slice(0, 1), arrow.slice(1)] if values else [], type=pa.large_string()
    )
    value = pd.Series(pd.arrays.ArrowStringArray(chunks))
    assert _hashing.compute_hash(value) == expected


def test_batch_boundaries_preserve_hash(monkeypatch):
    value = pd.Series((["a", None, "한글", ""] * 32769), dtype="string[pyarrow]")
    assert _hashing.compute_hash(value) == legacy_hash(value, monkeypatch)


@pytest.mark.parametrize("ordered", [False, True])
def test_categorical_strings_preserve_hash(ordered, monkeypatch):
    value = pd.Series(pd.Categorical(["b", None, "a", ""], categories=["", "unused", "b", "a"], ordered=ordered))
    assert _hashing.compute_hash(value) == legacy_hash(value, monkeypatch)
    if not ordered:
        plain = pd.Series(["b", None, "a", ""], dtype="string[python]")
        assert _hashing.compute_hash(value) == _hashing.compute_hash(plain)


def test_string_boundaries_nulls_and_order_are_distinct():
    cases = [["ab", "c"], ["a", "bc"], ["ab", None], ["ab", ""], ["c", "ab"]]
    hashes = [_hashing.compute_hash(pd.Series(value, dtype="string[pyarrow]")) for value in cases]
    assert len(set(hashes)) == len(cases)


def test_optimized_string_path_does_not_convert_to_numpy(monkeypatch):
    value = pd.Series(["a", None, "한글"], dtype="string[pyarrow]")
    expected = _hashing.compute_hash(value)

    def forbidden(*args, **kwargs):
        pytest.fail("Arrow string hashing must not materialize Python objects")

    monkeypatch.setattr(pd.arrays.ArrowStringArray, "__array__", forbidden)
    assert _hashing.compute_hash(value) == expected


def test_mixed_object_fallback_is_unchanged(monkeypatch):
    value = pd.Series(["a", 1, True, np.nan, None, b"x"], dtype=object)
    assert _hashing.compute_hash(value) == legacy_hash(value, monkeypatch)


def test_unused_bytes_in_null_slots_are_ignored(monkeypatch):
    arrow = pa.Array.from_buffers(
        pa.large_string(), 2,
        [pa.py_buffer(b"\x02"), pa.py_buffer(np.array([0, 6, 7], dtype="<i8")), pa.py_buffer(b"unusedx")],
    )
    value = pd.Series(pd.arrays.ArrowStringArray(arrow))
    expected = pd.Series([None, "x"], dtype="string[python]")
    assert _hashing.compute_hash(value) == legacy_hash(expected, monkeypatch)
