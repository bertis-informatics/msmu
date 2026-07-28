"""read_diann(sparse=True): the block-diagonal precursor .X, which becomes the only path.

This path had no coverage. The structural test is built from the reader's own output plus the input
report (no dense path involved), so it survives the deletion of the dense pivot; the parity test
additionally checks sparse == dense while both still exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest

import msmu as mm
from msmu._core._blockdiag import dense_block

from _parity_helpers import assert_reader_mdata_equal
from test_read_write_diann_parity import _clean_frame, _write


@pytest.fixture(autouse=True)
def _legacy_string_dtype():
    with pd.option_context("future.infer_string", False):
        yield


def _read(path, *, sparse):
    return mm.read_diann(path, sparse=sparse)


def test_read_diann_sparse_matches_dense(tmp_path):
    """sparse == dense end-to-end (checked while the dense pivot still exists)."""
    path = tmp_path / "report.tsv"
    _write(path, _clean_frame())

    assert_reader_mdata_equal(_read(path, sparse=True), _read(path, sparse=False))


def test_read_diann_sparse_is_block_diagonal(tmp_path):
    """Each precursor feature carries its quantity in exactly its own run, NaN in every other.

    Verified against the input report and the reader's own obs/var only -- no dense path -- so it
    keeps meaning once the dense pivot is deleted.
    """
    frame = _clean_frame()
    path = tmp_path / "report.tsv"
    _write(path, frame)

    adata = _read(path, sparse=True)["psm"]
    assert sp.issparse(adata.X), "read_diann(sparse=True) must store a sparse .X"

    dense = dense_block(adata.X)  # (n_runs, n_features), absent cells restored as NaN
    runs = list(adata.obs_names)
    feature_runs = list(adata.var["filename"])
    feature_peptides = list(adata.var["peptide"])
    feature_charges = list(adata.var["charge"])

    quantity_lookup = {
        (row["Run"], row["Modified.Sequence"], row["Precursor.Charge"]): row["Precursor.Quantity"]
        for _, row in frame.iterrows()
    }

    for feature_index in range(dense.shape[1]):
        feature_run = feature_runs[feature_index]
        expected_quantity = quantity_lookup[
            (feature_run, feature_peptides[feature_index], feature_charges[feature_index])
        ]
        for run_index, run in enumerate(runs):
            cell = dense[run_index, feature_index]
            if run == feature_run:
                assert np.isclose(cell, expected_quantity, rtol=1e-5), (
                    f"feature {feature_index} in {run}: {cell} != {expected_quantity}"
                )
            else:
                assert np.isnan(cell), f"feature {feature_index} must be absent (NaN) in {run}, got {cell}"


def test_read_diann_sparse_x_is_float32(tmp_path):
    """The sparse .X uses msmu's float32 convention."""
    path = tmp_path / "report.tsv"
    _write(path, _clean_frame())

    adata = _read(path, sparse=True)["psm"]
    assert adata.X.dtype == np.float32
