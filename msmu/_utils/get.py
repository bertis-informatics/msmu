"""Backward-compatible accessors re-exported from :mod:`msmu._utils._mudata`."""

from ._mudata import get_anndata, get_anndata_mod, get_mudata, get_mudata_mod

get_adata = get_anndata
get_mdata = get_mudata

__all__ = ["get_adata", "get_anndata", "get_anndata_mod", "get_mdata", "get_mudata", "get_mudata_mod"]
