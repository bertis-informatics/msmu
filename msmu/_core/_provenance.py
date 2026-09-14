"""Execution provenance stored entirely inside MuData.uns['_log']."""

from collections.abc import Mapping
from contextvars import ContextVar
from copy import deepcopy
import contextlib
import datetime
import functools
import inspect
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import time
import warnings
from importlib.metadata import distributions
from uuid import uuid4

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
from scipy import sparse

from ._hashing import ALGORITHM, compute_hash
from ._sources import get_download_buffer, is_url, source_scope
from ..logging_utils import get_logger, prune_closed_package_stream_handlers, prune_closed_stream_handlers

_hashing = ContextVar("msmu_provenance_hashing", default=False)
_active = ContextVar("msmu_provenance_active", default=False)
_OMIT = object()


def set_options(*, hashing: bool) -> None:
    """Enable or disable input/output hashing in this execution context (default False)."""
    if not isinstance(hashing, bool):
        raise TypeError("hashing must be a bool")
    _hashing.set(hashing)


def get_options() -> dict[str, bool]:
    """Return a snapshot of the settings in this execution context."""
    return {"hashing": _hashing.get()}


@contextlib.contextmanager
def options(*, hashing: bool):
    """Temporarily enable/disable hashing, restoring the previous setting on exit."""
    if not isinstance(hashing, bool):
        raise TypeError("hashing must be a bool")
    token = _hashing.set(hashing)
    try:
        yield
    finally:
        _hashing.reset(token)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _empty_log():
    return {"schema_version": 1, "events": {}, "environments": {}, "head": ""}


def _read_log(mdata):
    log = mdata.uns.get("_log", _empty_log())
    if not isinstance(log, dict) or log.get("schema_version") != 1:
        raise ValueError("Unsupported mdata.uns['_log'] schema")
    if not all(isinstance(log.get(k), dict) for k in ("events", "environments")):
        raise ValueError("Invalid mdata.uns['_log'] entries")
    if not isinstance(log.get("head"), str) or log["head"] and log["head"] not in log["events"]:
        raise ValueError("Invalid mdata.uns['_log'] head")
    for group in ("events", "environments"):
        for key, value in log[group].items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("Invalid mdata.uns['_log'] entry")
    return log


def get_log(mdata: md.MuData) -> dict:
    """Return detached, decoded provenance in lineage order, then execution-start order."""
    log = _read_log(mdata)
    return {
        "schema_version": int(log["schema_version"]),
        "head": log["head"],
        "events": sorted(
            (json.loads(v) for v in log["events"].values()), key=lambda e: (e["sequence"], e["started_at"], e["id"])
        ),
        "environments": {key: json.loads(value) for key, value in log["environments"].items()},
    }


def _merge_logs(target, source):
    for group in ("events", "environments"):
        for key, value in source[group].items():
            if key in target[group] and target[group][key] != value:
                raise ValueError(f"Conflicting provenance entry: {key}")
            target[group][key] = value


def _serialize_parameters(obj, _seen=None):
    """Serialize option values without truncation, omitting data objects."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else {"type": "float", "value": str(obj)}
    if isinstance(obj, np.generic):
        return _serialize_parameters(obj.item())
    if isinstance(obj, os.PathLike):
        return os.fsdecode(obj)
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return {"type": type(obj).__name__, "value": obj.isoformat()}
    if isinstance(obj, (md.MuData, ad.AnnData, pd.DataFrame, pd.Series, pd.Index, np.ndarray)) or sparse.issparse(obj):
        return _OMIT
    if callable(obj):
        return {
            "type": "callable",
            "path": f"{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', type(obj).__qualname__)}",
            "replayable": False,
        }
    seen = set() if _seen is None else _seen
    if id(obj) in seen:
        return {"type": "reference", "reason": "cyclic parameter", "replayable": False}
    seen.add(id(obj))
    try:
        if isinstance(obj, Mapping):
            items = [(k, _serialize_parameters(v, seen)) for k, v in obj.items()]
            items = [(k, v) for k, v in items if v is not _OMIT]
            if obj and not items:
                return _OMIT
            if all(isinstance(k, str) for k in obj):
                return dict(items)
            return {"type": "mapping", "items": [[_serialize_parameters(k, seen), v] for k, v in items]}
        if isinstance(obj, (list, tuple, set, frozenset)):
            values = [_serialize_parameters(v, seen) for v in obj]
            values = [v for v in values if v is not _OMIT]
            if obj and not values:
                return _OMIT
            return sorted(values, key=_json) if isinstance(obj, (set, frozenset)) else values
        return {
            "type": f"{type(obj).__module__}.{type(obj).__qualname__}",
            "representation": repr(obj),
            "replayable": False,
        }
    finally:
        seen.remove(id(obj))



@functools.lru_cache(maxsize=1)
def _base_environment():
    packages = {
        dist.metadata["Name"]: dist.version
        for dist in distributions()
        if dist.metadata["Name"]
    }
    try:
        from .._version import __commit_id__
    except ImportError:
        __commit_id__ = None
    source = {"commit": __commit_id__ or "unavailable", "dirty": None}
    root = Path(__file__).resolve().parents[2]
    if (root / ".git").exists():
        try:
            source["commit"] = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], timeout=2, stderr=subprocess.DEVNULL, text=True
            ).strip()
            source["dirty"] = bool(
                subprocess.check_output(
                    ["git", "-C", str(root), "status", "--porcelain", "--", "msmu"],
                    timeout=2,
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            )
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "packages": packages,
        "msmu_source": source,
    }


def _environment():
    result = deepcopy(_base_environment())
    result["numpy_errors"] = np.geterr()
    result["thread_settings"] = {
        name: os.environ[name]
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "PYTHONHASHSEED",
        )
        if name in os.environ
    }
    try:
        from threadpoolctl import threadpool_info

        result["threadpools"] = [{k: v for k, v in pool.items() if k != "filepath"} for pool in threadpool_info()]
    except ImportError:
        result["threadpools"] = "unavailable"
    return result


def _entities(value, path, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    if isinstance(value, (md.MuData, ad.AnnData, pd.DataFrame, pd.Series, pd.Index, np.ndarray)) or sparse.issparse(
        value
    ):
        yield path, value
    elif is_url(value):
        yield path, value
    elif isinstance(value, os.PathLike) or isinstance(value, str) and any(x in path.lower() for x in ("file", "path")):
        yield path, Path(value)
    elif isinstance(value, (Mapping, list, tuple)):
        seen.add(id(value))
        try:
            items = value.items() if isinstance(value, Mapping) else enumerate(value)
            for key, item in items:
                yield from _entities(item, f"{path}/{key}", seen)
        finally:
            seen.remove(id(value))


def _entity(path, value, hashing):
    entity = {"id": str(uuid4()), "role": path, "type": type(value).__name__, "hash": {"status": "disabled"}}
    if isinstance(value, Path) or is_url(value):
        entity["path"] = str(value)
    if is_url(value):
        entity["type"] = "URL"
    if hashing:
        started = time.perf_counter()
        try:
            content = get_download_buffer(value) if is_url(value) else value
            if content is None:
                raise ValueError("URL content was not read through the shared input loader")
            entity["hash"] = {"status": "completed", "algorithm": ALGORITHM, "value": compute_hash(content)}
        except Exception as error:
            entity["hash"] = {"status": "unavailable", "algorithm": ALGORITHM, "reason": str(error)}
        entity["hash"]["duration_seconds"] = time.perf_counter() - started
    return entity


def log_provenance(func):
    """Log one public call. Nested MSMU calls are represented by their outer call."""

    def run(*args, **kwargs):
        if _active.get():
            return func(*args, **kwargs)
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = list(_entities(bound.arguments, "arguments"))
        owners = [v for _, v in inputs if isinstance(v, md.MuData)]
        log = _empty_log()
        parents = []
        for owner in owners:
            previous = _read_log(owner)
            _merge_logs(log, previous)
            if previous["head"]:
                parents.append(previous["head"])
        hashing = _hashing.get()
        event = {
            "id": str(uuid4()),
            "function": func.__name__,
            "function_path": f"{func.__module__}.{func.__qualname__}",
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "parents": list(dict.fromkeys(parents)),
            "parameters": {
                key: value for key, argument in bound.arguments.items()
                if (value := _serialize_parameters(argument)) is not _OMIT
            },
            "inputs": [_entity(path, value, hashing) for path, value in inputs],
            "outputs": [],
            "hashing": hashing,
        }
        if hashing:
            for entity, (_, value) in zip(event["inputs"], inputs):
                if not isinstance(value, md.MuData):
                    continue
                previous = _read_log(value)
                current_hash = entity["hash"]
                if not previous["head"] or current_hash.get("status") != "completed":
                    continue
                head = json.loads(previous["events"][previous["head"]])
                outputs = [output for output in head["outputs"] if output["type"] == "MuData"]
                # Multiple MuData outputs have no persistent object-to-output mapping yet.
                if len(outputs) != 1:
                    continue
                previous_hash = outputs[0]["hash"]
                if (
                    previous_hash.get("status") == "completed"
                    and previous_hash.get("algorithm") == current_hash.get("algorithm")
                    and previous_hash.get("value") != current_hash["value"]
                ):
                    warnings.warn(
                        "Data changed outside the recorded workflow.\n"
                        f"  Between: {head['function']} → {func.__name__}\n"
                        "  If this step needs MSMU support, please open an issue describing the operation:\n"
                        "  https://github.com/bertis-informatics/msmu/issues",
                        UserWarning,
                        stacklevel=3,
                    )
        environment = _environment()
        environment_json = _json(environment)
        from hashlib import sha256

        env_id = sha256(environment_json.encode()).hexdigest()
        log["environments"][env_id] = environment_json
        event["environment_id"] = env_id
        token = _active.set(True)
        started = time.perf_counter()
        try:
            prune_closed_package_stream_handlers()
            prune_closed_stream_handlers(logging.getLogger(), only_msmu_handlers=False)
            result = func(*args, **kwargs)
        finally:
            _active.reset(token)
        event["duration_seconds"] = time.perf_counter() - started
        event["ended_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            if hashing:
                for entity, (path, value) in zip(event["inputs"], inputs):
                    if is_url(value):
                        entity["hash"] = _entity(path, value, True)["hash"]
            outputs = list(_entities(result, "return"))
            # DEA has a separate result object, while the history belongs to its input MuData.
            if result is not None and type(result).__name__ == "DeaResult":
                outputs = [("return/DeaResult", vars(result))]
            event["outputs"] = [_entity(path, value, hashing) for path, value in outputs]
            targets = [v for _, v in outputs if isinstance(v, md.MuData)] or owners
            # Readers, notably read_h5mu, return an existing history without an input MuData.
            if not owners:
                for target in targets:
                    previous = _read_log(target)
                    _merge_logs(log, previous)
                    if previous["head"]:
                        event["parents"].append(previous["head"])
            # A non-MuData-returning operation may still modify its input.
            if not any(isinstance(v, md.MuData) for _, v in outputs):
                event["outputs"].extend(_entity(f"after/{i}", owner, hashing) for i, owner in enumerate(owners))
            event["sequence"] = len(log["events"])
            log["events"][event["id"]] = _json(event)
            log["head"] = event["id"]
            for target in targets:
                target.uns["_log"] = deepcopy(log)
        except Exception as error:
            # Logging must not discard a successful computation result.
            get_logger().warning("Could not store provenance for %s: %s", func.__name__, error)
        return result

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with source_scope():
            return run(*args, **kwargs)

    return wrapper
