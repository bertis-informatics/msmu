from typing import Any, cast

from anndata.abc import CSCDataset, CSRDataset
from anndata.typing import XDataType
import numpy as np
from scipy import sparse as sp


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


__all__ = ["_has_quant_values"]
