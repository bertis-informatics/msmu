from typing import Any, cast

from anndata.abc import CSCDataset, CSRDataset
from anndata.experimental.backed import Dataset2D
try:
    from anndata.typing import XDataType
except ImportError:  # anndata >=0.13 made XDataType private; XDataArray is the public replacement
    from anndata.typing import XDataArray as XDataType
import numpy as np
from pandas import DataFrame
from scipy import sparse as sp


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


__all__ = ["_has_quant_values", "_require_columns"]
