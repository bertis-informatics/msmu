from typing import Any, cast

from anndata import AnnData
from anndata.abc import CSCDataset, CSRDataset
from anndata.experimental.backed import Dataset2D
try:
    from anndata.typing import XDataType
except ImportError:  # anndata >=0.13 made XDataType private; XDataArray is the public replacement
    from anndata.typing import XDataArray as XDataType
import numpy as np
from pandas import DataFrame
from scipy import sparse as sp

from .._core._blockdiag import dense_block


def _require_columns(frame: DataFrame | Dataset2D, columns: list[str], context: str) -> None:
    """Raise a single, readable error when required columns are missing."""
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Required columns missing from {context}: {missing_columns}")


# Each level labels its accessions differently: a modality that has been through protein inference
# carries "protein_group", PTM sites carry the accessions they were localised on as
# "modified_protein", and peptides carry the search engine's own list as "proteins". Most preferred
# first.
ACCESSION_COLUMN_PREFERENCE: tuple[str, ...] = ("protein_group", "modified_protein", "proteins")


def _resolve_accession_column(frame: DataFrame | Dataset2D, context: str) -> str:
    """Return the var column carrying protein accessions for annotation helpers.

    Annotation only needs *some* accession string, so this takes the first available rather than
    requiring protein inference to have run -- the PTM path localises sites from the search engine's
    accessions and never attaches an inferred ``protein_group``.
    """
    for column in ACCESSION_COLUMN_PREFERENCE:
        if column in frame.columns:
            return column

    raise ValueError(f"No protein accession column found in {context}; expected one of {ACCESSION_COLUMN_PREFERENCE}.")


DECOY_COLUMN = "decoy"


def _get_target_feature_mask(var: DataFrame) -> np.ndarray:
    """Return True for the target features of a ``.var`` table and False for its decoys.

    A decoy is a row whose ``decoy`` column is 1. A table without that column holds targets only.
    """
    if DECOY_COLUMN not in var.columns:
        return np.ones(len(var), dtype=bool)
    return var[DECOY_COLUMN].ne(1).to_numpy()


def _drop_decoy_features(adata: AnnData) -> AnnData:
    """Return the modality without its decoy features, for analyses of the quantification.

    A modality without decoys is returned as it is; otherwise the result is a view.
    """
    is_target_feature = _get_target_feature_mask(adata.var)
    if is_target_feature.all():
        return adata
    return adata[:, is_target_feature]


def _replace_target_columns(full_matrix: Any, target_matrix: Any, is_target_feature: np.ndarray) -> Any:
    """Return a copy of ``full_matrix`` whose target columns hold ``target_matrix``.

    For computations that rewrite the target columns only: the decoy columns keep the values of
    ``full_matrix``. The result takes the format and dtype of ``target_matrix``; a sparse
    ``full_matrix`` read into a dense result gives NaN for its absent cells.
    """
    if is_target_feature.all():
        return target_matrix
    if sp.issparse(target_matrix):
        decoy_matrix = sp.csc_matrix(full_matrix)[:, ~is_target_feature]
        stacked_matrix = sp.hstack([sp.csc_matrix(target_matrix), decoy_matrix], format="csc")
        stacked_column_positions = np.concatenate(
            [np.flatnonzero(is_target_feature), np.flatnonzero(~is_target_feature)]
        )
        replaced_matrix = stacked_matrix[:, np.argsort(stacked_column_positions)]
        return replaced_matrix.asformat(target_matrix.format).astype(target_matrix.dtype)
    full_values = dense_block(full_matrix) if sp.issparse(full_matrix) else np.asarray(full_matrix)
    replaced_values = full_values.astype(target_matrix.dtype, copy=True)
    replaced_values[:, is_target_feature] = target_matrix
    return replaced_values


def _has_quant_values(matrix: XDataType | None) -> bool:
    """Return whether an AnnData matrix contains at least one non-NaN value."""
    if matrix is None:
        return False

    if isinstance(matrix, (CSRDataset, CSCDataset)):
        return _has_quant_values(matrix.to_memory())

    if sp.issparse(matrix):
        sparse_matrix = cast(Any, matrix)
        rows, cols = sparse_matrix.shape
        if rows == 0 or cols == 0:
            return False

        # An all-absent (nnz == 0) sparse matrix is the sparse form of an all-NaN dense matrix -> no
        # quant values. (The old "nnz < rows*cols -> True" shortcut answered "is it sparse", not "has
        # values", and wrongly reported True here.) Otherwise there are values iff any stored one is
        # non-NaN.
        if sparse_matrix.nnz == 0:
            return False

        return not bool(np.isnan(sparse_matrix.data).all())

    values = np.asarray(matrix)
    if values.size == 0:
        return False

    return not bool(np.isnan(values).all())


__all__ = [
    "_drop_decoy_features",
    "_get_target_feature_mask",
    "_has_quant_values",
    "_require_columns",
    "_replace_target_columns",
]
