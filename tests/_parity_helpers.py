"""Shared assertions for reader parity tests (polars path vs a frozen pandas-derived golden).

The pandas path is the de-facto specification for the readers that have no parity coverage
(Sage, DIA-NN, FragPipe, DELPI). The goldens under ``tests/goldens/`` capture that specification
as MuData written to ``.h5mu`` *while the pandas path still exists*, so the polars path stays
pinned to the reference after the pandas path is deleted (the cutover). Tests compare the polars
read against the loaded golden and never run the pandas path at test time.

Regenerating the goldens (only while the pandas path exists, or later from the surviving reader)::

    MSMU_REGEN_READER_GOLDENS=1 pytest tests/test_read_write_*_parity.py \\
        tests/test_read_write_maxquant_null_scan.py

Value-level, not byte-exact: ``.X`` and numeric ``var`` columns are compared with a tolerance
(polars parses CSV floats correctly-rounded, which differs from pandas by ~1 ULP on some inputs),
and dtypes are deliberately not asserted (float32-vs-float64 and Arrow-string-vs-object are noise
here, decided separately). ``.X`` is compared NaN-aware and sparse-aware (a sparse block-diagonal
``.X`` is restored to dense-with-NaN via :func:`msmu._core._blockdiag.dense_block`, never densified
to 0).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from msmu._core._blockdiag import dense_block

_GOLDEN_DIR = Path(__file__).parent / "goldens"
# Set MSMU_REGEN_READER_GOLDENS=1 to (re)write goldens from the pandas path instead of comparing.
_REGEN = os.environ.get("MSMU_REGEN_READER_GOLDENS", "").lower() not in ("", "0", "false", "no")


def load_golden(golden_name: str):
    """Load a frozen reader golden MuData from ``tests/goldens/<golden_name>.h5mu``."""
    from msmu import read_h5mu

    return read_h5mu(str(_GOLDEN_DIR / f"{golden_name}.h5mu"))


def assert_polars_matches_golden(
    read_fn, golden_name: str, *, ordered: bool = True, check_var: bool = True, rtol: float = 1e-5, atol: float = 1e-6
) -> None:
    """Assert the polars read equals the frozen golden for ``golden_name``.

    ``read_fn(as_polars=...)`` returns the reader MuData for the engine requested. In normal runs
    only the polars read happens and it is compared against the loaded golden. When
    ``MSMU_REGEN_READER_GOLDENS`` is set, the golden is first (re)written from ``read_fn`` -- from
    the pandas path (``as_polars=False``) so the reference stays the pandas specification.
    """
    if _REGEN:
        import anndata

        # polars->pandas gives nullable/Arrow-backed string columns; anndata 0.13 refuses to write
        # those under infer_string=False unless this is opted in. Values are unaffected (the compare
        # is dtype-agnostic); this only unblocks serialising the golden.
        anndata.settings.allow_write_nullable_strings = True

        _GOLDEN_DIR.mkdir(exist_ok=True)
        golden_path = _GOLDEN_DIR / f"{golden_name}.h5mu"
        if golden_path.exists():
            golden_path.unlink()
        golden = read_fn(as_polars=False)
        # Keep only what this helper compares (.X, obs/var names, var columns). The aligned mappings
        # are not compared, and some reader ``varm['search_result']`` columns (null-bearing object
        # dtype, e.g. a null scan) fail h5mu serialisation on the anndata-0.13/pandas-3 stack.
        for modality in golden.mod.values():
            for aligned in (modality.varm, modality.obsm, modality.varp, modality.obsp, modality.layers):
                aligned.clear()
        golden.write_h5mu(str(golden_path))

    assert_reader_mdata_equal(
        read_fn(as_polars=True), load_golden(golden_name), ordered=ordered, check_var=check_var, rtol=rtol, atol=atol
    )


def _dense(matrix) -> np.ndarray:
    """Dense NaN-aware view of an ``.X`` (sparse block-diagonal restored to NaN, not 0)."""
    if sp.issparse(matrix):
        return dense_block(matrix)
    return np.asarray(matrix, dtype=float)


def _both_nan(left, right) -> bool:
    try:
        return bool(pd.isna(left)) and bool(pd.isna(right))
    except (TypeError, ValueError):
        return False


def _canonical(adata):
    """View of ``adata`` with obs and var sorted by name (a stable order for order-insensitive compare)."""
    return adata[sorted(adata.obs_names), sorted(adata.var_names)]


def assert_reader_mdata_equal(
    actual, expected, *, rtol: float = 1e-5, atol: float = 1e-6, check_var: bool = True, ordered: bool = True
) -> None:
    """Assert two reader MuData outputs are equal at the value level.

    Compares, per modality: ``obs_names`` and ``var_names`` (exact), ``.X`` (NaN-aware,
    sparse-aware, within tolerance), and every ``var`` column (numeric within tolerance,
    other columns exact and NaN-aware). Dtypes are not asserted.

    ``ordered=False`` sorts both sides by obs/var name first -- use it for multi-file inputs, whose
    row order is non-deterministic on the pandas ProcessPool path (the polars path is deterministic).
    """
    assert set(actual.mod) == set(expected.mod), f"modalities differ: {set(actual.mod)} vs {set(expected.mod)}"

    for modality in expected.mod:
        actual_adata, expected_adata = actual[modality], expected[modality]
        if not ordered:
            actual_adata, expected_adata = _canonical(actual_adata), _canonical(expected_adata)

        assert list(actual_adata.obs_names) == list(expected_adata.obs_names), f"[{modality}] obs_names differ"
        assert list(actual_adata.var_names) == list(expected_adata.var_names), f"[{modality}] var_names differ"

        actual_x, expected_x = _dense(actual_adata.X), _dense(expected_adata.X)
        assert actual_x.shape == expected_x.shape, f"[{modality}] .X shape {actual_x.shape} vs {expected_x.shape}"
        np.testing.assert_allclose(
            actual_x, expected_x, rtol=rtol, atol=atol, equal_nan=True, err_msg=f"[{modality}] .X differs"
        )

        if check_var:
            assert list(actual_adata.var.columns) == list(expected_adata.var.columns), f"[{modality}] var columns differ"
            for column in expected_adata.var.columns:
                _assert_column_equal(actual_adata.var[column], expected_adata.var[column], modality, column, rtol, atol)


def _assert_column_equal(actual_col: pd.Series, expected_col: pd.Series, modality: str, column: str, rtol: float, atol: float) -> None:
    numeric = pd.api.types.is_numeric_dtype(expected_col) and not pd.api.types.is_bool_dtype(expected_col)
    if numeric:
        np.testing.assert_allclose(
            actual_col.to_numpy(dtype=float),
            expected_col.to_numpy(dtype=float),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            err_msg=f"[{modality}] var[{column!r}] numeric differs",
        )
        return

    actual_values, expected_values = list(actual_col), list(expected_col)
    assert len(actual_values) == len(expected_values), f"[{modality}] var[{column!r}] length differs"
    for row, (actual_value, expected_value) in enumerate(zip(actual_values, expected_values)):
        if _both_nan(actual_value, expected_value):
            continue
        assert actual_value == expected_value, f"[{modality}] var[{column!r}] row {row}: {actual_value!r} != {expected_value!r}"
