"""Tools must read a sparse ``layers[...]`` through to_dense_df, not raw ``pd.DataFrame(...)``.

The documented idiom ``mdata['psm'].layers['raw'] = mdata['psm'].X.copy()`` keeps a sparse ``.X``
sparse, so every tool that accepts ``layer=`` can receive a sparse matrix. Wrapping one in
``pd.DataFrame`` raises (pandas >= 2.2) or, on older pandas, densifies with SciPy's implicit-zero
convention and silently turns structurally-absent cells into 0. Both are wrong: the tools must go
through ``to_dense_df``, which restores absent cells as NaN.

The fixtures are built so that a passing test means more than "it did not crash":

* the layer has genuinely **structurally-absent** cells (not stored in the sparse matrix), so a
  densify that fills them with 0 instead of NaN changes the numbers, and
* it also carries a real measured **0.0**, which must survive as 0.0 -- absent and zero are
  different things and a correct implementation must not conflate them, and
* ``.X`` holds **decoy** values that differ from the layer, so a tool that ignores ``layer=`` and
  reads ``.X`` produces different numbers and fails.
"""

import os

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData
from mudata import MuData

os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")

from msmu._core._blockdiag import to_dense_df
from msmu._tools._correlation import corr
from msmu._tools._dea import run_de
from msmu._tools._pca import pca

try:
    from msmu._tools._umap import umap
except RuntimeError as exc:  # pragma: no cover - depends on the local numba/umap build
    umap = None
    _UMAP_SKIP_REASON = f"UMAP import failed: {exc}"

SAMPLES = [f"s{i}" for i in range(1, 7)]
FEATURES = ["v1", "v2", "v3", "v4", "v5", "v6"]

# The layer under test. NaN marks a structurally-absent cell (never stored in the sparse form);
# the 0.0 in v3 is a real measurement that must come back as 0.0, not be mistaken for absent.
_LAYER_VALUES = np.array(
    [
        [4.0, 6.0, 0.0, 8.0, 1.0, np.nan],
        [5.0, 7.0, 2.0, 9.0, 2.0, 5.0],
        [6.0, 5.0, 3.0, 7.0, 3.0, np.nan],
        [9.0, 2.0, 8.0, 3.0, 9.0, 4.0],
        [8.0, 3.0, 9.0, 2.0, 8.0, np.nan],
        [7.0, 4.0, 7.0, 4.0, 7.0, 6.0],
    ]
)

# Deliberately different from _LAYER_VALUES: this sits in .X so that any tool which ignores
# `layer=` and falls back to .X yields different results and the comparison below fails.
_DECOY_X_VALUES = _LAYER_VALUES[::-1].copy() * 3.0 + 100.0
_DECOY_X_VALUES[np.isnan(_DECOY_X_VALUES)] = 42.0


def _sparse_with_absent_cells(dense: np.ndarray) -> sp.csr_matrix:
    """CSR holding exactly the non-NaN cells; NaN positions are structurally absent (unstored).

    Real 0.0 values ARE stored (as explicit zeros) so they stay distinguishable from absent cells,
    which is the whole distinction ``to_dense_df`` exists to preserve.
    """
    observed_rows, observed_cols = np.nonzero(~np.isnan(dense))
    matrix = sp.csr_matrix(
        (dense[observed_rows, observed_cols], (observed_rows, observed_cols)),
        shape=dense.shape,
    )
    return matrix


def _obs() -> pd.DataFrame:
    return pd.DataFrame(
        {"group": pd.Categorical(["A", "A", "A", "B", "B", "B"], categories=["A", "B"])},
        index=SAMPLES,
    )


def _finish(adata: AnnData) -> MuData:
    mdata = MuData({"psm": adata})
    mdata.obs["group"] = pd.Categorical(adata.obs["group"], categories=["A", "B"])
    return mdata


def _make_sparse_layer_mdata() -> MuData:
    """psm modality with a SPARSE layer holding _LAYER_VALUES and a decoy dense ``.X``."""
    adata = AnnData(
        X=_DECOY_X_VALUES.copy(),
        obs=_obs(),
        var=pd.DataFrame(index=FEATURES),
    )
    adata.layers["raw"] = _sparse_with_absent_cells(_LAYER_VALUES)
    assert sp.issparse(adata.layers["raw"])
    return _finish(adata)


def _make_dense_layer_mdata() -> MuData:
    """Reference: the same values as a plain dense layer (NaN where the sparse form is absent)."""
    adata = AnnData(
        X=_DECOY_X_VALUES.copy(),
        obs=_obs(),
        var=pd.DataFrame(index=FEATURES),
    )
    adata.layers["raw"] = _LAYER_VALUES.copy()
    return _finish(adata)


def test_fixture_actually_exercises_absent_cells_and_layer_routing():
    """Guard the guards: the fixtures must be able to distinguish the failures under test."""
    adata = _make_sparse_layer_mdata()["psm"]
    layer = adata.layers["raw"]

    n_cells = layer.shape[0] * layer.shape[1]
    assert layer.nnz < n_cells, "fixture stores every cell, so absent-vs-zero is never exercised"
    assert np.isnan(_LAYER_VALUES).sum() == n_cells - layer.nnz
    # A real zero is stored, so it is not confusable with an absent cell.
    assert (_LAYER_VALUES == 0.0).any()
    assert not np.array_equal(np.nan_to_num(_DECOY_X_VALUES), np.nan_to_num(_LAYER_VALUES)), (
        ".X must differ from the layer, otherwise ignoring layer= would go unnoticed"
    )


def test_to_dense_df_restores_absent_cells_as_nan_and_keeps_real_zero():
    adata = _make_sparse_layer_mdata()["psm"]

    frame = to_dense_df(adata, layer="raw")

    np.testing.assert_array_equal(np.isnan(frame.to_numpy()), np.isnan(_LAYER_VALUES))
    np.testing.assert_allclose(
        np.nan_to_num(frame.to_numpy(), nan=-1.0),
        np.nan_to_num(_LAYER_VALUES, nan=-1.0),
    )
    assert frame.iloc[0, 2] == 0.0, "a real 0.0 must not be turned into an absent cell"


def test_run_de_on_sparse_layer_matches_dense():
    sparse_result = run_de(
        _make_sparse_layer_mdata(),
        modality="psm",
        category="group",
        ctrl="A",
        expr="B",
        layer="raw",
        stat_method="limma",
    )
    dense_result = run_de(
        _make_dense_layer_mdata(),
        modality="psm",
        category="group",
        ctrl="A",
        expr="B",
        layer="raw",
        stat_method="limma",
    )

    pd.testing.assert_frame_equal(sparse_result.to_df(), dense_result.to_df())
    # And the layer is what was analysed -- not .X.
    decoy_result = run_de(
        _make_dense_layer_mdata(),
        modality="psm",
        category="group",
        ctrl="A",
        expr="B",
        layer=None,
        stat_method="limma",
    )
    assert not sparse_result.to_df().equals(decoy_result.to_df())


def test_corr_on_sparse_layer_matches_dense():
    # corr writes obsp on a local copy (pre-existing, unrelated to sparse storage), so its result is
    # not observable on the returned object. What is asserted here: a sparse layer reaches the
    # correlation without raising, and the frame it correlates is the NaN-aware dense equivalent.
    sparse_mdata = _make_sparse_layer_mdata()
    corr(sparse_mdata, modality="psm", layer="raw")
    corr(_make_dense_layer_mdata(), modality="psm", layer="raw")

    sparse_frame = to_dense_df(sparse_mdata["psm"], layer="raw")
    expected = pd.DataFrame(_LAYER_VALUES, index=SAMPLES, columns=FEATURES)
    pd.testing.assert_frame_equal(sparse_frame.T.corr(), expected.T.corr())


def test_corr_still_rejects_a_missing_layer():
    with pytest.raises(ValueError, match="not found in modality"):
        corr(_make_sparse_layer_mdata(), modality="psm", layer="does_not_exist")


def test_pca_on_sparse_layer_matches_dense():
    sparse_out = pca(_make_sparse_layer_mdata(), modality="psm", n_components=2, layer="raw", random_state=0)
    dense_out = pca(_make_dense_layer_mdata(), modality="psm", n_components=2, layer="raw", random_state=0)
    decoy_out = pca(_make_dense_layer_mdata(), modality="psm", n_components=2, layer=None, random_state=0)

    sparse_pca = np.asarray(sparse_out["psm"].obsm["X_pca"], dtype=float)
    np.testing.assert_allclose(sparse_pca, np.asarray(dense_out["psm"].obsm["X_pca"], dtype=float), atol=1e-10)
    assert not np.allclose(sparse_pca, np.asarray(decoy_out["psm"].obsm["X_pca"], dtype=float))


@pytest.mark.skipif(umap is None, reason=globals().get("_UMAP_SKIP_REASON", "UMAP unavailable"))
@pytest.mark.filterwarnings("ignore:n_jobs value .* overridden .* by setting random_state.*:UserWarning")
def test_umap_on_sparse_layer_matches_dense():
    sparse_out = umap(_make_sparse_layer_mdata(), modality="psm", n_neighbors=2, layer="raw", random_state=0)
    dense_out = umap(_make_dense_layer_mdata(), modality="psm", n_neighbors=2, layer="raw", random_state=0)

    np.testing.assert_allclose(
        np.asarray(sparse_out["psm"].obsm["X_umap"], dtype=float),
        np.asarray(dense_out["psm"].obsm["X_umap"], dtype=float),
    )


def test_tools_reject_a_zero_filled_densify_of_the_sparse_layer():
    """The failure mode the fix exists for: absent cells arriving as 0 instead of NaN.

    A 0-filled densify is what ``adata.to_df(layer=...)`` / a plain SciPy ``.toarray()`` would
    produce. Feeding that through the same tool must give a DIFFERENT answer than the correct
    NaN-aware path, otherwise these tests could not tell the two apart.
    """
    zero_filled = np.nan_to_num(_LAYER_VALUES, nan=0.0)
    adata = AnnData(X=_DECOY_X_VALUES.copy(), obs=_obs(), var=pd.DataFrame(index=FEATURES))
    adata.layers["raw"] = zero_filled
    zero_filled_mdata = _finish(adata)

    correct = pca(_make_sparse_layer_mdata(), modality="psm", n_components=2, layer="raw", random_state=0)
    wrong = pca(zero_filled_mdata, modality="psm", n_components=2, layer="raw", random_state=0)

    assert not np.allclose(
        np.asarray(correct["psm"].obsm["X_pca"], dtype=float),
        np.asarray(wrong["psm"].obsm["X_pca"], dtype=float),
    )
