"""``_normalise`` must handle a sparse block-diagonal layer NaN-aware, never 0-filling the
structurally-absent cells.

``log2_transform`` is the LAST sparse-preserving step (per the port's design: read + log2 keep
the block-diagonal sparse, scale/normalise then densify), so it must (a) keep the layer sparse and
(b) transform only stored cells while absent cells stay NaN -- a 0-filled densify would turn absent
cells into ``log2(0) = -inf``. ``scale_data`` / ``normalise`` densify via ``dense_block`` (NaN
restore); the guarantee under test is that the sparse path yields the SAME result as a dense layer
holding the same values -- which only holds if absent cells become NaN, not 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData
from mudata import MuData

from msmu._core._blockdiag import to_dense_df
from msmu._preprocessing._normalise import log2_transform, normalise, scale_data

SAMPLES = [f"s{i}" for i in range(1, 7)]
FEATURES = ["v1", "v2", "v3", "v4", "v5"]

# All stored values are > 0 so log2 is finite; NaN marks a structurally-absent cell (never stored
# in the sparse form). v5 is mostly absent -- the block-diagonal shape this representation exists for.
_VALUES = np.array(
    [
        [4.0, 6.0, 1.0, 8.0, np.nan],
        [5.0, 7.0, 2.0, 9.0, 5.0],
        [6.0, 5.0, 3.0, 7.0, np.nan],
        [9.0, 2.0, 8.0, 3.0, 4.0],
        [8.0, 3.0, 9.0, 2.0, np.nan],
        [7.0, 4.0, 7.0, 4.0, 6.0],
    ]
)

# Deliberately different, and NaN-free (a valid dense .X). If any function ignored ``layer=`` and
# fell back to ``.X``, the sparse-vs-dense comparison would surface it.
_DECOY_X = (_VALUES[::-1] * 3.0 + 100.0)
_DECOY_X[np.isnan(_DECOY_X)] = 42.0


def _sparse_with_absent_cells(dense: np.ndarray) -> sp.csr_matrix:
    """CSR holding exactly the non-NaN cells; NaN positions are structurally absent (unstored)."""
    rows, cols = np.nonzero(~np.isnan(dense))
    return sp.csr_matrix((dense[rows, cols], (rows, cols)), shape=dense.shape)


def _mdata(layer_array) -> MuData:
    adata = AnnData(
        X=_DECOY_X.copy(),
        obs=pd.DataFrame(index=SAMPLES),
        var=pd.DataFrame(index=FEATURES),
    )
    adata.layers["raw"] = layer_array
    return MuData({"psm": adata})


def _make_sparse() -> MuData:
    mdata = _mdata(_sparse_with_absent_cells(_VALUES))
    assert sp.issparse(mdata["psm"].layers["raw"])
    return mdata


def _make_dense() -> MuData:
    return _mdata(_VALUES.copy())


def _n_absent() -> int:
    return int(np.isnan(_VALUES).sum())


def test_fixture_has_absent_cells_and_a_distinct_decoy_x():
    """Guard the guards: the sparse layer really has unstored cells and .X differs from the layer."""
    assert _n_absent() == 3
    assert _make_sparse()["psm"].layers["raw"].nnz == _VALUES.size - _n_absent()
    assert not np.allclose(_DECOY_X, np.nan_to_num(_VALUES), equal_nan=True)


def test_log2_keeps_layer_sparse():
    """log2 is elementwise, so it must leave the block-diagonal layer sparse (memory preserved)."""
    out = log2_transform(_make_sparse(), modality="psm", layer="raw")
    assert sp.issparse(out["psm"].layers["raw"]), "log2 densified the sparse layer"


def test_log2_is_nan_aware_never_zero_fills():
    """Absent cells must stay NaN; a 0-filled densify would make them log2(0) = -inf."""
    out = log2_transform(_make_sparse(), modality="psm", layer="raw")
    dense = to_dense_df(out["psm"], layer="raw").to_numpy()
    assert not np.isneginf(dense).any(), "absent cell became log2(0) = -inf (0-filled)"
    assert np.isnan(dense).sum() == _n_absent(), "absent-cell count changed"


@pytest.mark.parametrize(
    "func, kwargs",
    [
        (log2_transform, {}),
        (scale_data, {}),
        (normalise, {"method": "median"}),
        (normalise, {"method": "median_center"}),
        (normalise, {"method": "quantile"}),
        (normalise, {"method": "total_sum"}),
    ],
)
def test_sparse_layer_matches_dense(func, kwargs):
    """Every _normalise op on a sparse layer must equal the same op on a dense NaN layer."""
    sparse_out = func(_make_sparse(), modality="psm", layer="raw", **kwargs)
    dense_out = func(_make_dense(), modality="psm", layer="raw", **kwargs)
    pd.testing.assert_frame_equal(
        to_dense_df(sparse_out["psm"], layer="raw"),
        to_dense_df(dense_out["psm"], layer="raw"),
    )


def test_scale_re_sparsifies_and_keeps_absent_cells_nan():
    """scale_data densifies for per-feature mean/std, then re-sparsifies a sparse input back to sparse;
    absent cells stay NaN (never scaled from an imputed 0)."""
    out = scale_data(_make_sparse(), modality="psm", layer="raw")
    assert sp.issparse(out["psm"].layers["raw"]), "scale_data should re-sparsify a sparse input"
    dense = to_dense_df(out["psm"], layer="raw").to_numpy()
    assert np.isnan(dense).sum() == _n_absent()


def test_a_zero_filled_layer_would_change_the_result():
    """Prove the matches-dense guard is non-vacuous: 0-filling the absent cells (what a naive
    ``.toarray()`` densify does) genuinely changes the output, so the tests above would reject it."""
    zero_filled = np.nan_to_num(_VALUES, nan=0.0)
    naive = to_dense_df(scale_data(_mdata(zero_filled), modality="psm", layer="raw")["psm"], layer="raw")
    correct = to_dense_df(scale_data(_make_dense(), modality="psm", layer="raw")["psm"], layer="raw")
    assert not naive.equals(correct), "0-fill and NaN-aware scaling coincided -- fixture cannot discriminate"


@pytest.mark.parametrize("method", ["median", "median_center"])
def test_normalise_per_sample_median_keeps_layer_sparse(method):
    """Per-sample median normalisation (the common PSM-level TMT/DIA step) must stay sparse: the
    block-diagonal is centered per obs row in place, never materialised as a dense matrix."""
    out = normalise(_make_sparse(), method=method, modality="psm", layer="raw")
    assert sp.issparse(out["psm"].layers["raw"]), f"normalise({method}) densified the sparse block-diagonal"
    assert np.isnan(to_dense_df(out["psm"], layer="raw").to_numpy()).sum() == _n_absent()


def test_normalise_quantile_re_sparsifies():
    """quantile densifies to compute (its per-sample rank mapping couples all samples), but a sparse
    input is re-sparsified back to a sparse output -- absent cells dropped again."""
    out = normalise(_make_sparse(), method="quantile", modality="psm", layer="raw")
    assert sp.issparse(out["psm"].layers["raw"]), "quantile should re-sparsify a sparse input"
    assert np.isnan(to_dense_df(out["psm"], layer="raw").to_numpy()).sum() == _n_absent()


def test_normalise_total_sum_keeps_layer_sparse():
    """Per-sample total-intensity normalisation rescales each obs row's stored values in place, so it
    must stay sparse rather than materialise the dense matrix."""
    out = normalise(_make_sparse(), method="total_sum", modality="psm", layer="raw")
    assert sp.issparse(out["psm"].layers["raw"]), "normalise(total_sum) densified the sparse block-diagonal"
    assert np.isnan(to_dense_df(out["psm"], layer="raw").to_numpy()).sum() == _n_absent()


# obs "batch" and var "fraction" columns for grouped-normalisation tests. batch splits the 6 samples
# into two groups; fraction puts v5 (mostly absent) in its own group so some (row, fraction) blocks are
# all-absent and must be skipped -- exercising the block-skip path within a fraction.
_BATCH = ["b1", "b1", "b2", "b2", "b1", "b2"]
_FRACTION = ["f1", "f1", "f1", "f1", "f2"]


def _mdata_grouped(layer_array) -> MuData:
    adata = AnnData(
        X=_DECOY_X.copy(),
        obs=pd.DataFrame({"batch": _BATCH}, index=SAMPLES),
        var=pd.DataFrame({"fraction": _FRACTION}, index=FEATURES),
    )
    adata.layers["raw"] = layer_array
    return MuData({"psm": adata})


@pytest.mark.parametrize("method", ["median", "median_center", "total_sum"])
@pytest.mark.parametrize(
    "group_kwargs",
    [
        {"batch_key": "batch"},
        {"fraction_key": "fraction"},
        {"batch_key": "batch", "fraction_key": "fraction"},
    ],
)
def test_grouped_sparse_matches_dense(method, group_kwargs):
    """Grouped (batch / fraction / both) per-sample normalisation on a sparse layer must equal the dense
    result and stay sparse -- each (obs_group x var_group) block is rescaled in place, never densified."""
    sparse_out = normalise(
        _mdata_grouped(_sparse_with_absent_cells(_VALUES)), method=method, modality="psm", layer="raw", **group_kwargs
    )
    dense_out = normalise(_mdata_grouped(_VALUES.copy()), method=method, modality="psm", layer="raw", **group_kwargs)
    assert sp.issparse(sparse_out["psm"].layers["raw"]), f"grouped normalise({method}) densified the sparse layer"
    pd.testing.assert_frame_equal(
        to_dense_df(sparse_out["psm"], layer="raw"),
        to_dense_df(dense_out["psm"], layer="raw"),
    )


def test_grouped_normalise_actually_groups():
    """Non-vacuity guard: fraction grouping must change the result vs ungrouped, so the parity test
    cannot pass by both paths silently ignoring the grouping."""
    grouped = to_dense_df(
        normalise(
            _mdata_grouped(_sparse_with_absent_cells(_VALUES)),
            method="median",
            modality="psm",
            layer="raw",
            fraction_key="fraction",
        )["psm"],
        layer="raw",
    ).to_numpy()
    ungrouped = to_dense_df(
        normalise(_mdata_grouped(_sparse_with_absent_cells(_VALUES)), method="median", modality="psm", layer="raw")[
            "psm"
        ],
        layer="raw",
    ).to_numpy()
    assert not np.allclose(grouped, ungrouped, equal_nan=True)
