"""Shared assertions for msmu's MuData copy: mudata's own copy() is the oracle for the global bookkeeping."""

import mudata as md
import numpy as np

from msmu._core._hashing import compute_hash
from msmu._utils._mudata import copy_mudata


def _array_parts(mapping) -> dict:
    parts = {}
    for key in mapping.keys():
        array = np.asarray(mapping[key])
        values = array.tolist()
        if array.dtype.kind == "f":
            # mudata fills rows it cannot place with NaN, and NaN never compares equal to itself
            values = np.where(np.isnan(array), None, array.astype(object)).tolist()
        parts[key] = (values, str(array.dtype), array.shape)
    return parts


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


def assert_bookkeeping_is_final(mdata: md.MuData, label: str) -> None:
    """``mdata`` must already be in the state ``MuData.copy()`` produces (bookkeeping and provenance hash),
    and ``copy_mudata`` must reproduce that state."""
    reference = mdata.copy()
    assert global_bookkeeping(mdata) == global_bookkeeping(reference), label
    assert compute_hash(mdata) == compute_hash(reference), label
    copied = copy_mudata(mdata)
    assert global_bookkeeping(copied) == global_bookkeeping(reference), label
    assert compute_hash(copied) == compute_hash(reference), label
