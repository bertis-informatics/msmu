"""Compatibility re-exports for MuData helpers moved to :mod:`msmu._utils._mudata`."""

from .._utils._mudata import (
    MutableMuDataMod,
    MutableMuDataObsMap,
    add_modality,
    get_anndata,
    get_anndata_mod,
    get_mudata,
    get_mudata_mod,
    get_mudata_mod_as_mutable,
    mutable_mudata_mod,
    mutable_mudata_obsmap,
)


__all__ = [
    "add_modality",
    "MutableMuDataMod",
    "MutableMuDataObsMap",
    "get_anndata",
    "get_anndata_mod",
    "get_mudata",
    "get_mudata_mod",
    "get_mudata_mod_as_mutable",
    "mutable_mudata_mod",
    "mutable_mudata_obsmap",
]
