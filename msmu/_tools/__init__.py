from ._pca import pca
from ._correlation import corr
from ._dea import run_de
from .._statistics._de_base import PermTestResult, StatTestResult


def umap(*args, **kwargs):
    from ._umap import umap as _umap

    return _umap(*args, **kwargs)


def compute_precursor_isolation_purity(*args, **kwargs):
    from ._precursor_purity import compute_precursor_isolation_purity as _compute

    return _compute(*args, **kwargs)


def compute_precursor_isolation_purity_from_mzml(*args, **kwargs):
    from ._precursor_purity import compute_precursor_isolation_purity_from_mzml as _compute_from_mzml

    return _compute_from_mzml(*args, **kwargs)


__all__ = [
    "compute_precursor_isolation_purity",
    "compute_precursor_isolation_purity_from_mzml",
    "pca",
    "umap",
    "corr",
    "run_de",
    "PermTestResult",
    "StatTestResult",
]
