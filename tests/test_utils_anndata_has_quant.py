"""_has_quant_values must answer "are there stored values", not "is it sparse-structured".

§C-4: an all-absent (``nnz == 0``) sparse matrix is the sparse form of an all-NaN dense matrix, so
both must report False. Today the ``nnz < rows*cols`` shortcut returns True for the empty sparse
matrix -- the ``xfail`` pins the fixed contract and flips when Phase 3 corrects it.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from msmu._core._blockdiag import to_observed_sparse
from msmu._utils._anndata import _has_quant_values


def test_has_quant_values_none_is_false():
    assert _has_quant_values(None) is False


def test_has_quant_values_dense_all_nan_is_false():
    assert _has_quant_values(np.full((2, 3), np.nan)) is False


def test_has_quant_values_dense_with_a_value_is_true():
    dense = np.full((2, 3), np.nan)
    dense[0, 0] = 1.0
    assert _has_quant_values(dense) is True


def test_has_quant_values_partially_observed_sparse_is_true():
    dense = np.full((2, 3), np.nan)
    dense[0, 0] = 5.0
    dense[1, 2] = 7.0
    assert _has_quant_values(to_observed_sparse(dense, dtype=np.float64)) is True


def test_has_quant_values_all_absent_sparse_is_false():
    empty = sp.csr_matrix((2, 3), dtype=np.float64)  # nnz == 0, the sparse form of all-NaN
    assert _has_quant_values(empty) is False
