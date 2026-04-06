from collections.abc import Mapping, Sequence
import contextlib
import datetime
import functools
import inspect
import io
import json
import logging
import os
import platform
import sys
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from ..logging_utils import get_logger

MAX_SEQ_ITEMS = 20
MAX_STRING_LEN = 500
MAX_DEPTH = 10


def _truncate_string(value: str, max_len: int = MAX_STRING_LEN) -> str:
    if len(value) <= max_len:
        return value
    return f"{value[:max_len]}...(truncated)"


def _get_msmu_version() -> str:
    try:
        return version("msmu")
    except PackageNotFoundError:
        return "unknown"


def get_bound_call_kwargs(func, *args, **kwargs) -> dict[str, object]:
    """
    Bind a function call to its signature and return parameter->value mapping,
    including defaults for omitted arguments.
    """
    bound = inspect.signature(func).bind_partial(*args, **kwargs)
    bound.apply_defaults()
    return {str(k): v for k, v in bound.arguments.items() if k not in {"self", "mdata"}}


def serialize(obj, *, depth: int = 0) -> object:
    if depth >= MAX_DEPTH:
        return {"__type__": "truncated", "reason": "max_depth"}

    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, str):
            return _truncate_string(obj)
        return obj

    if isinstance(obj, os.PathLike):
        return _truncate_string(os.fspath(obj))

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, datetime.datetime):
        return obj.isoformat()

    if isinstance(obj, Mapping):
        out = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= MAX_SEQ_ITEMS:
                out["__truncated__"] = True
                break
            out[str(k)] = serialize(v, depth=depth + 1)
        return out

    if isinstance(obj, (set, tuple, list)):
        values = list(obj)[:MAX_SEQ_ITEMS]
        out = [serialize(v, depth=depth + 1) for v in values]
        if len(obj) > MAX_SEQ_ITEMS:
            out.append({"__truncated__": True})
        return out

    if isinstance(obj, np.ndarray):
        return {
            "__type__": "ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }

    if isinstance(obj, pd.DataFrame):
        return {
            "__type__": "dataframe",
            "shape": [int(obj.shape[0]), int(obj.shape[1])],
            "columns": [str(c) for c in obj.columns[:MAX_SEQ_ITEMS]],
        }

    if isinstance(obj, pd.Series):
        return {
            "__type__": "series",
            "length": int(obj.shape[0]),
            "dtype": str(obj.dtype),
            "name": None if obj.name is None else str(obj.name),
        }

    if isinstance(obj, ad.AnnData):
        return {
            "__type__": "anndata",
            "shape": [int(obj.n_obs), int(obj.n_vars)],
        }

    if isinstance(obj, md.MuData):
        return {
            "__type__": "mudata",
            "mod_names": [str(x) for x in obj.mod_names],
        }

    if isinstance(obj, Sequence) and not isinstance(obj, str):
        values = list(obj)[:MAX_SEQ_ITEMS]
        out = [serialize(v, depth=depth + 1) for v in values]
        if len(obj) > MAX_SEQ_ITEMS:
            out.append({"__truncated__": True})
        return out

    return {"__type__": type(obj).__name__, "repr": _truncate_string(repr(obj))}


def append_cmd_log(
    mdata: md.MuData,
    *,
    function: str,
    payload: Mapping | None = None,
    stdout: str | None = None,
    input_dimensions: Mapping | None = None,
    output_dimensions: Mapping | None = None,
) -> md.MuData:
    if not isinstance(mdata, md.MuData):
        return mdata

    payload = serialize({} if payload is None else dict(payload))

    log_entry = {
        "function": function,
        "timestamp": datetime.datetime.now().isoformat(),
        "msmu_version": _get_msmu_version(),
        "python_version": platform.python_version(),
        "payload": payload,
    }
    if stdout:
        log_entry["stdout"] = _truncate_string(stdout)
    if input_dimensions is not None:
        log_entry["input_dimensions"] = serialize(dict(input_dimensions))
    if output_dimensions is not None:
        log_entry["output_dimensions"] = serialize(dict(output_dimensions))

    normalize_cmd_for_runtime(mdata)
    if "_cmd" not in mdata.uns_keys():
        mdata.uns["_cmd"] = {}

    cmd_logs: dict[str, dict] = mdata.uns["_cmd"]
    cmd_logs[_next_cmd_key(cmd_logs)] = log_entry
    return mdata


def _next_cmd_key(cmd_logs: Mapping[str, object]) -> str:
    numeric_keys = [int(k) for k in cmd_logs if str(k).isdigit()]
    if not numeric_keys:
        return "0"
    return str(max(numeric_keys) + 1)


def _coerce_cmd_entry_to_dict(entry) -> dict:
    if isinstance(entry, dict):
        # Keep runtime command entries stable across repeated normalization.
        # Re-serializing already-serialized dicts can introduce nested truncation.
        return deepcopy(entry)
    if isinstance(entry, str):
        try:
            parsed = json.loads(entry)
            if isinstance(parsed, dict):
                return _coerce_cmd_entry_to_dict(parsed)
            return {"function": "unknown", "payload": serialize(parsed)}
        except Exception:
            return {"function": "unknown", "payload": _truncate_string(entry)}
    return {"function": "unknown", "payload": _truncate_string(repr(entry))}


def normalize_cmd_for_runtime(mdata: md.MuData) -> md.MuData:
    """
    Ensure mdata.uns['_cmd'] is dict[str, dict] in runtime.
    """
    if not isinstance(mdata, md.MuData):
        return mdata
    if "_cmd" not in mdata.uns_keys():
        return mdata

    raw = mdata.uns["_cmd"]
    if isinstance(raw, dict):
        mdata.uns["_cmd"] = {str(k): _coerce_cmd_entry_to_dict(v) for k, v in raw.items()}
    elif isinstance(raw, list):
        mdata.uns["_cmd"] = {str(i): _coerce_cmd_entry_to_dict(x) for i, x in enumerate(raw)}
    else:
        raise TypeError("mdata.uns['_cmd'] must be dict or list for runtime normalization.")

    return mdata


class _StdoutTee:
    def __init__(self, original, buffer):
        self._original = original
        self._buffer = buffer

    def write(self, data):
        self._original.write(data)
        self._buffer.write(data)
        return len(data)

    def flush(self):
        self._original.flush()
        self._buffer.flush()


class _LogCaptureHandler(logging.Handler):
    def __init__(self, buffer: io.StringIO, tee_to_stdout: bool):
        super().__init__(level=logging.INFO)
        self._buffer = buffer
        self._tee_to_stdout = tee_to_stdout

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self._buffer.write(msg + "\n")
        if self._tee_to_stdout:
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()


@contextlib.contextmanager
def capture_provenance_output():
    """
    Capture stdout + msmu logger output while still showing messages on screen.
    Returns a StringIO buffer containing captured text.
    """
    stdout_buffer = io.StringIO()
    tee = _StdoutTee(sys.stdout, stdout_buffer)

    msmu_logger = get_logger()
    original_level = msmu_logger.level
    visible_handlers = [h for h in msmu_logger.handlers if not isinstance(h, logging.NullHandler)]
    has_info_visible_handler = any(h.level <= logging.INFO for h in visible_handlers)
    handler = _LogCaptureHandler(stdout_buffer, tee_to_stdout=not has_info_visible_handler)
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    msmu_logger.addHandler(handler)
    if msmu_logger.getEffectiveLevel() > logging.INFO:
        msmu_logger.setLevel(logging.INFO)

    try:
        with contextlib.redirect_stdout(tee):
            yield stdout_buffer
    finally:
        msmu_logger.removeHandler(handler)
        msmu_logger.setLevel(original_level)


def _get_mdata_dimensions(mdata: md.MuData) -> dict[str, object]:
    return {
        "n_obs": int(mdata.n_obs),
        "n_vars": int(mdata.n_vars),
        "modalities": {
            mod: {
                "n_obs": int(mdata.mod[mod].n_obs),
                "n_vars": int(mdata.mod[mod].n_vars),
                "layers": [str(layer) for layer in mdata.mod[mod].layers.keys()],
            }
            for mod in mdata.mod_names
        },
    }


def uns_logger(func):
    @functools.wraps(func)
    def wrapper(mdata, *args, **kwargs):
        with capture_provenance_output() as stdout_buffer:
            result = func(mdata, *args, **kwargs)

        if not isinstance(mdata, md.MuData) or not isinstance(result, md.MuData):
            return result

        full_kwargs = get_bound_call_kwargs(func, mdata, *args, **kwargs)
        captured_stdout = stdout_buffer.getvalue().strip()
        input_dimensions = _get_mdata_dimensions(mdata)
        output_dimensions = _get_mdata_dimensions(result)
        return append_cmd_log(
            result,
            function=func.__name__,
            payload=full_kwargs,
            stdout=captured_stdout if captured_stdout else None,
            input_dimensions=input_dimensions,
            output_dimensions=output_dimensions,
        )

    return wrapper
