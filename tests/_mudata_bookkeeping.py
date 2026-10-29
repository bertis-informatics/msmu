"""Shared assertions for msmu's MuData copy: mudata's own copy() is the oracle for the global bookkeeping."""

import mudata as md
import numpy as np

from msmu._core._hashing import compute_hash
from msmu._utils._mudata import copy_mudata


def _array_parts(mapping) -> dict:
    return {
        key: (np.asarray(mapping[key]).tolist(), str(np.asarray(mapping[key]).dtype), np.asarray(mapping[key]).shape)
        for key in mapping.keys()
    }


def global_bookkeeping(mdata: md.MuData) -> dict:
    """Everything mudata's update() maintains at the MuData level, as plain comparable values."""
    return {
        "var_index": mdata.var.index.tolist(),
        "var_index_dtype": str(mdata.var.index.dtype),
        "var_index_name": mdata.var.index.name,
        "var_columns": (mdata.var.columns.tolist(), str(mdata.var.columns.dtype)),
        "obs_index": mdata.obs.index.tolist(),
        "obs_columns": mdata.obs.columns.tolist(),
        "varmap": _array_parts(mdata.varmap),
        "varm": _array_parts(mdata.varm),
        "obsmap": _array_parts(mdata.obsmap),
        "obsm": _array_parts(mdata.obsm),
    }


def assert_copy_matches_mudata_copy(mdata: md.MuData, label: str) -> md.MuData:
    """copy_mudata(mdata) must equal MuData.copy() in bookkeeping and provenance hash; returns the copy."""
    reference = mdata.copy()
    copied = copy_mudata(mdata)
    assert global_bookkeeping(copied) == global_bookkeeping(reference), label
    assert compute_hash(copied) == compute_hash(reference), label
    return copied
