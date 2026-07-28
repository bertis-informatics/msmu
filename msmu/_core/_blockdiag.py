"""Grouped block-diagonal sparse representation for PSM/precursor-level quantification.

Several msmu quantification matrices are *grouped block-diagonal*: the feature axis
encodes a group (a TMT plex, or a DIA-NN run), and each feature is observed only in the
samples belonging to its own group. Concretely:

* after :func:`msmu.pp.split_tmt`, ``obs`` is ``channel x plex`` and each PSM feature carries
  values only in its plex's channels (NaN elsewhere);
* the DIA-NN reader's precursor matrix has ``var = run.precursor`` and each precursor feature
  carries a value only in its run.

Stored densely this is ``O(P^2)`` in the number of groups ``P`` (both axes grow with ``P``),
which is what makes large multi-plex / multi-run studies blow past memory. Only the block
diagonal carries data (``O(P)``); the rest is structural NaN.

This module stores such matrices as a SciPy sparse matrix that keeps **only the observed
cells**, and exposes helpers so consumers never materialise the full dense matrix:

* :func:`to_observed_sparse` -- dense (NaN = missing) -> CSC keeping observed cells only.
* :func:`dense_block` -- slice back to a small dense block, filling absent cells with NaN.
* :func:`aggregate_features_by_group` -- per-group column reduction (median/mean/sum) without
  ever densifying the whole matrix.

NaN / zero handling
-------------------
"Observed" is defined by the sparsity **pattern**, not by a ``value == 0`` test: an absent
cell is one that was never stored, and :func:`dense_block` restores it as NaN by scattering the
stored values into a NaN-filled array. A genuinely-zero observed intensity (should not occur in
practice -- msmu's readers already map 0 -> NaN) is therefore preserved rather than being
silently treated as missing. This is the one invariant that made a naive ``scipy`` retrofit
(``toarray()`` fills 0, then ``nanmedian`` treats the 0 as data) silently wrong; keeping it in
one place is the whole point of this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "to_observed_sparse",
    "is_sparse",
    "dense_block",
    "to_dense_df",
    "sparse_apply_elementwise",
    "aggregate_features_by_group",
    "SparseQuantFrame",
]


@dataclass
class SparseQuantFrame:
    """A sparse ``(features x samples)`` quantification carrier for the reader flow.

    Stands in for the dense quantification DataFrame a reader would otherwise build by pivoting
    a block-diagonal long table (DIA-NN precursors, where each ``run.precursor`` feature is
    observed in exactly one run). It stores only observed cells and exposes just enough of a
    DataFrame surface (``index``, ``columns``) for the base reader to align features, plus
    :meth:`anndata_x` to produce the sparse ``.X`` directly -- so the dense pivot is never built.
    """

    matrix: sp.spmatrix  # (n_features, n_samples), observed cells only
    index: pd.Index  # feature names (rows)
    columns: pd.Index  # sample names (cols)

    def anndata_x(self, feature_order: pd.Index):
        """Return the sparse ``(samples x features)`` matrix for ``AnnData(X=...)``.

        Features are selected/reordered to ``feature_order`` (the aligned var index); samples
        stay in ``columns`` order (the obs axis).
        """
        positions = self.index.get_indexer(feature_order)
        if (positions < 0).any():
            missing = feature_order[positions < 0]
            raise KeyError(f"SparseQuantFrame is missing features: {list(missing)[:5]}")
        selected = self.matrix[positions, :]  # (n_selected_features, n_samples)
        return selected.T.tocsr()  # (n_samples, n_features)

# Column-wise, NaN-aware reducers. These match pandas ``groupby(...).agg(method)`` on a dense
# frame with NaN for missing: median/mean of an all-NaN column -> NaN; sum of an all-NaN
# column -> 0 (numpy ``nansum`` and pandas ``sum`` agree here).
_REDUCERS = {
    "median": np.nanmedian,
    "mean": np.nanmean,
    "sum": np.nansum,
}


def is_sparse(x) -> bool:
    """Return True for a SciPy sparse matrix/array."""
    return sp.issparse(x)


def to_observed_sparse(dense: np.ndarray, *, dtype=np.float32) -> sp.csc_matrix:
    """Convert a dense ``(samples, features)`` matrix to CSC keeping only observed cells.

    A cell is observed iff it is finite; NaN cells are dropped (not stored). Genuine zeros are
    kept as explicit stored values so the observed pattern is exactly the finite mask.

    Args:
        dense: 2D array, NaN = missing.
        dtype: stored value dtype (default float32, matching msmu's ``.X``).

    Returns:
        CSC matrix of shape ``dense.shape`` storing only the observed values.
    """
    dense = np.asarray(dense)
    if dense.ndim != 2:
        raise ValueError("to_observed_sparse expects a 2D array.")
    observed = np.isfinite(dense)
    rows, cols = np.nonzero(observed)
    values = dense[rows, cols].astype(dtype, copy=False)
    out = sp.csc_matrix((values, (rows, cols)), shape=dense.shape, dtype=dtype)
    # Keep explicit zeros: an observed intensity of exactly 0 must stay in the pattern so it is
    # not later restored as NaN. (eliminate_zeros() would drop it -- deliberately not called.)
    return out


def dense_block(x, col_indices=None, row_indices=None) -> np.ndarray:
    """Return a dense sub-block with absent cells filled as NaN.

    Works on both sparse and dense ``x`` so callers stay dtype-agnostic. For sparse ``x`` the
    result is built by scattering stored values into a NaN-filled array, so cells that were
    never stored come back as NaN (not 0) and stored zeros are preserved.

    Args:
        x: ``(samples, features)`` sparse matrix or dense ndarray.
        col_indices: optional feature (column) selection.
        row_indices: optional sample (row) selection.

    Returns:
        Dense float64 block, NaN for absent cells.
    """
    if not sp.issparse(x):
        block = np.asarray(x, dtype=float)
        if row_indices is not None:
            block = block[row_indices, :]
        if col_indices is not None:
            block = block[:, col_indices]
        return block

    sub = x
    if col_indices is not None:
        sub = sub[:, col_indices]
    if row_indices is not None:
        sub = sub[row_indices, :]
    coo = sub.tocoo()
    out = np.full(coo.shape, np.nan, dtype=float)
    out[coo.row, coo.col] = coo.data
    return out


def to_dense_df(adata, layer: str | None = None) -> pd.DataFrame:
    """NaN-aware replacement for ``adata.to_df()`` that is safe on a sparse ``.X``.

    ``AnnData.to_df()`` densifies a sparse matrix with SciPy's implicit-zero convention, so
    structurally-absent cells come back as 0 -- silently wrong for any NaN-aware consumer
    (medians, distributions, exports). This restores absent cells as NaN via
    :func:`dense_block`, giving the exact frame the dense path would have produced.

    Args:
        adata: AnnData whose ``.X`` (or ``layers[layer]``) may be sparse or dense.
        layer: optional layer name; ``None`` uses ``.X``.

    Returns:
        ``(obs x var)`` DataFrame, NaN for absent cells.
    """
    matrix = adata.X if layer is None else adata.layers[layer]
    if sp.issparse(matrix):
        # dense_block already returns float64 with NaN for absent cells; keep that (do not re-cast to
        # matrix.dtype) so the sparse and dense branches return the same dtype, and so a non-float .X
        # cannot turn the NaN fill back into 0.
        values = dense_block(matrix)
    else:
        values = np.asarray(matrix, dtype=float)
    return pd.DataFrame(values, index=adata.obs_names, columns=adata.var_names)


def sparse_apply_elementwise(matrix, func):
    """Apply an elementwise ``func`` to the stored values only, preserving sparsity.

    For elementwise transforms (e.g. ``log2``) the absent (structurally-missing) cells stay
    absent, so the transform runs on ``O(nnz)`` stored values and never densifies. Dense input
    is transformed as usual.
    """
    if sp.issparse(matrix):
        out = matrix.copy()
        out.data = func(out.data)
        return out
    return func(np.asarray(matrix, dtype=float))


def aggregate_features_by_group(
    x,
    feature_groups: np.ndarray,
    method: str,
    group_order: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce features within each group, per sample, without densifying the whole matrix.

    For each group of feature columns, only that group's (small) block is densified -- so peak
    memory is one block at a time, never the full ``(samples, features)`` matrix.

    Args:
        x: ``(n_samples, n_features)`` sparse or dense matrix.
        feature_groups: length ``n_features`` array assigning each feature to a group.
        method: one of ``"median"``, ``"mean"``, ``"sum"`` (NaN-aware, column-wise).
        group_order: optional explicit group ordering for the output rows; defaults to the
            sorted unique groups (matching a default pandas ``groupby``).

    Returns:
        ``(groups, aggregated)`` where ``groups`` is the group order and ``aggregated`` is a
        ``(n_groups, n_samples)`` array (group-by-sample), NaN where a group has no observed
        value in a sample (0 for ``method="sum"``, matching pandas).
    """
    if method not in _REDUCERS:
        raise ValueError(f"aggregate method '{method}' not supported; choose from {sorted(_REDUCERS)}.")

    feature_groups = np.asarray(feature_groups)
    if group_order is None:
        group_order = np.unique(feature_groups)
    n_samples = x.shape[0]

    # Long form of the *observed* cells only (O(nnz), never the dense matrix). Grouping and
    # reducing is then a single vectorised pandas groupby -- far faster than a Python loop over
    # groups, and identical to ``groupby(group).agg(method)`` on the dense (NaN=missing) frame.
    if sp.issparse(x):
        coo = x.tocoo()
        sample_idx, feature_idx, values = coo.row, coo.col, coo.data
    else:
        arr = np.asarray(x, dtype=float)
        sample_idx, feature_idx = np.nonzero(np.isfinite(arr))
        values = arr[sample_idx, feature_idx]

    long = pd.DataFrame(
        {"_group": feature_groups[feature_idx], "_sample": sample_idx, "_value": values}
    )
    wide = long.groupby(["_group", "_sample"])["_value"].agg(method).unstack("_sample")
    wide = wide.reindex(index=group_order, columns=np.arange(n_samples))
    aggregated = wide.to_numpy(dtype=float)

    if method == "sum":
        # A group with no observed value in a sample sums to 0 under pandas' dense ``sum``
        # (all-NaN column -> 0); the long form leaves it absent, so fill those back to 0.
        aggregated = np.nan_to_num(aggregated, nan=0.0)

    return np.asarray(group_order), aggregated
