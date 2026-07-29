"""``collapse_obs`` and ``correct_batch_effect`` must handle a sparse block-diagonal ``.X``
NaN-aware. Both read ``.X`` and densify via ``dense_block``, which must restore structurally-absent
cells as NaN, not 0. A 0-fill would corrupt the obs-group aggregation (collapse) and the per-batch
medians (batch correction) -- so each op on a sparse ``.X`` must equal the same op on a dense ``.X``
holding the same values with NaN in the absent cells.

The fixture is built so one obs-group (``p2`` = s3,s4) is ENTIRELY absent for feature ``v5``:
that cell is ``nansum -> NaN`` correctly, but ``0 -> 0`` under a naive densify, which is what the
adversarial tests exploit to prove the guards are not vacuous.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from mudata import MuData

from msmu._core._blockdiag import to_dense_df
from msmu._preprocessing._batch_correction import correct_batch_effect
from msmu._preprocessing._collapse import collapse_obs

SAMPLES = [f"s{i}" for i in range(1, 7)]
FEATURES = ["v1", "v2", "v3", "v4", "v5"]

# Positive stored values; NaN = structurally-absent. v5 is absent for the whole p2 group (s3,s4).
_VALUES = np.array(
    [
        [4.0, 6.0, 1.0, 8.0, 2.0],       # s1  (sample p1, batch b1)
        [5.0, 7.0, 2.0, 9.0, np.nan],    # s2  (p1, b1)
        [6.0, 5.0, 3.0, 7.0, np.nan],    # s3  (p2, b1)
        [9.0, 2.0, 8.0, 3.0, np.nan],    # s4  (p2, b2)  -> p2's v5 is fully absent
        [8.0, 3.0, 9.0, 2.0, 7.0],       # s5  (p3, b2)
        [7.0, 4.0, 7.0, 4.0, 6.0],       # s6  (p3, b2)
    ]
)


def _sparse_with_absent_cells(dense: np.ndarray) -> sp.csr_matrix:
    rows, cols = np.nonzero(~np.isnan(dense))
    return sp.csr_matrix((dense[rows, cols], (rows, cols)), shape=dense.shape)


def _obs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample": ["p1", "p1", "p2", "p2", "p3", "p3"],
            "batch": ["b1", "b1", "b1", "b2", "b2", "b2"],
        },
        index=SAMPLES,
    )


def _mdata(x_array) -> MuData:
    mdata = MuData({"psm": AnnData(X=x_array, obs=_obs(), var=pd.DataFrame(index=FEATURES))})
    # correct_batch_effect reads the MuData-level obs (like the conftest simple_mdata fixture);
    # collapse_obs reads the modality obs. Set both so each consumer finds its key.
    mdata.obs["sample"] = _obs()["sample"].to_numpy()
    mdata.obs["batch"] = _obs()["batch"].to_numpy()
    return mdata


def _make_sparse() -> MuData:
    mdata = _mdata(_sparse_with_absent_cells(_VALUES))
    assert sp.issparse(mdata["psm"].X)
    return mdata


def _make_dense() -> MuData:
    return _mdata(_VALUES.copy())


def _make_zero_filled() -> MuData:
    return _mdata(np.nan_to_num(_VALUES, nan=0.0))


def _n_absent() -> int:
    return int(np.isnan(_VALUES).sum())


def test_fixture_has_a_fully_absent_group_cell():
    """Guard the guards: p2's v5 must be entirely unstored so 0-fill and NaN-aware genuinely differ."""
    assert _n_absent() == 3
    assert _make_sparse()["psm"].X.nnz == _VALUES.size - _n_absent()
    p2_v5 = _VALUES[[2, 3], 4]  # s3, s4 x v5
    assert np.isnan(p2_v5).all()


# ---------------------------------------------------------------- collapse_obs
def test_collapse_obs_sparse_matches_dense():
    sparse_out = collapse_obs(_make_sparse(), sample_key="sample", agg_method="sum")
    dense_out = collapse_obs(_make_dense(), sample_key="sample", agg_method="sum")
    pd.testing.assert_frame_equal(to_dense_df(sparse_out["psm"]), to_dense_df(dense_out["psm"]))


def test_collapse_obs_keeps_fully_absent_group_cell_nan():
    """The p2/v5 group-cell aggregates to NaN (all absent), never 0."""
    out = collapse_obs(_make_sparse(), sample_key="sample", agg_method="sum")
    collapsed = to_dense_df(out["psm"])
    # locate p2's row and v5's column regardless of orientation
    values = collapsed.to_numpy()
    assert np.isnan(values).any(), "a fully-absent group-cell was 0-filled instead of NaN"


def test_collapse_obs_zero_fill_would_differ():
    naive = collapse_obs(_make_zero_filled(), sample_key="sample", agg_method="sum")
    correct = collapse_obs(_make_dense(), sample_key="sample", agg_method="sum")
    assert not to_dense_df(naive["psm"]).equals(to_dense_df(correct["psm"])), (
        "0-fill and NaN-aware collapse coincided -- fixture cannot discriminate"
    )


# ---------------------------------------------------------- correct_batch_effect
def test_correct_batch_effect_sparse_matches_dense():
    sparse_out = correct_batch_effect(
        _make_sparse(), modality="psm", method="median_center", category="batch"
    )
    dense_out = correct_batch_effect(
        _make_dense(), modality="psm", method="median_center", category="batch"
    )
    pd.testing.assert_frame_equal(to_dense_df(sparse_out["psm"]), to_dense_df(dense_out["psm"]))


def test_correct_batch_effect_zero_fill_would_differ():
    naive = correct_batch_effect(
        _make_zero_filled(), modality="psm", method="median_center", category="batch"
    )
    correct = correct_batch_effect(
        _make_dense(), modality="psm", method="median_center", category="batch"
    )
    assert not to_dense_df(naive["psm"]).equals(to_dense_df(correct["psm"])), (
        "0-fill and NaN-aware batch correction coincided -- fixture cannot discriminate"
    )
