from typing import Any, cast

from anndata.abc import CSCDataset, CSRDataset
from anndata.experimental.backed import Dataset2D
from anndata.typing import XDataType
import numpy as np
from pandas import DataFrame
from scipy import sparse as sp


def _require_columns(frame: DataFrame | Dataset2D, columns: list[str], context: str) -> None:
    """Raise a single, readable error when required columns are missing."""
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Required columns missing from {context}: {missing_columns}")


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

        if sparse_matrix.nnz < rows * cols:
            return True

        return not bool(np.isnan(sparse_matrix.data).all())

    values = np.asarray(matrix)
    if values.size == 0:
        return False

    return not bool(np.isnan(values).all())


__all__ = ["_has_quant_values", "_require_columns"]
