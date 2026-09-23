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
from typing import Any
import platform
import re
import subprocess
import time
from importlib.metadata import distributions
from io import BytesIO
from uuid import uuid4

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
from scipy import sparse

from ._hashing import ALGORITHM, HASH_POLICY, _validate_precision, compute_hash
from ._sources import get_download_buffer, is_url, source_scope
from ..logging_utils import get_logger, prune_closed_package_stream_handlers, prune_closed_stream_handlers

_hashing = ContextVar("msmu_provenance_hashing", default=True)
_active = ContextVar("msmu_provenance_active", default=False)
_OMIT = object()
_normalization: ContextVar[str | None] = ContextVar("msmu_hash_normalization", default=HASH_POLICY)
_significant_digits: ContextVar[int | None] = ContextVar("msmu_hash_significant_digits", default=12)


def set_options(*, hashing: bool) -> None:
    """Enable or disable input/output hashing in this execution context (default True).

    Parameters:
        hashing: Whether recorded calls compute input/output content hashes. Disabling hashing does not disable provenance recording.

    Returns:
        None. Updates settings for the current execution context.

    Notes:
        See [`compute_hash`][msmu.pv.compute_hash] for hash semantics and [`get_options`][msmu.pv.get_options] to inspect current settings. Hash verification during [`replay`][msmu.pv.replay] requires recorded hashes.

    Examples:
        ```python
        import msmu as mm
        mm.pv.set_options(hashing=True)
        ```
    """
    if not isinstance(hashing, bool):
        raise TypeError("hashing must be a bool")
    _hashing.set(hashing)


def get_options() -> dict[str, bool]:
    """Return a snapshot of the settings in this execution context.

    Returns:
        A new dictionary with `hashing` for the current execution context. Mutating it does not change settings.

    Examples:
        ```python
        import msmu as mm
        settings = mm.pv.get_options()
        ```

    Use [`set_options`][msmu.pv.set_options] or scoped [`options`][msmu.pv.options] to change settings.
    """
    return {"hashing": _hashing.get()}


@contextlib.contextmanager
def options(*, hashing: bool, significant_digits: int | None = 12, normalization: str | None = HASH_POLICY):
    """Temporarily enable/disable hashing, restoring the previous setting on exit.

    Parameters:
        hashing: Whether recorded calls compute input/output content hashes. Disabling hashing does not disable provenance recording.
        significant_digits: Compatibility setting for legacy histories (1 to 15, or None for exact hashing). Keep the default for new workflows.
        normalization: Hash policy; defaults to `"msmu-v1"`. Legacy values are retained for replay compatibility. Keep the default for new workflows.

    Returns:
        context (ContextManager): A context manager restoring previous settings on exit.

    Notes:
        See [`compute_hash`][msmu.pv.compute_hash] for hash semantics and [`get_options`][msmu.pv.get_options] to inspect current settings. Hash verification during [`replay`][msmu.pv.replay] requires recorded hashes.

    Examples:
        ```python
        import msmu as mm
        with mm.pv.options(hashing=False):
            history = mm.pv.get_log(mdata)
        ```
    """
    if not isinstance(hashing, bool):
        raise TypeError("hashing must be a bool")
    _validate_precision(significant_digits)
    normalization_token = _normalization.set(normalization)
    precision_token = _significant_digits.set(significant_digits)
    token = _hashing.set(hashing)
    try:
        yield
    finally:
        _hashing.reset(token)
        _significant_digits.reset(precision_token)
        _normalization.reset(normalization_token)


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
    """Return detached, decoded provenance in lineage order, then execution-start order.

    Parameters:
        mdata: MuData containing recorded history in `.uns["_log"]`.

    Returns:
        A detached dictionary with `schema_version`, `head`, `events` (decoded event list), and `environments` (ID-to-environment mapping). No history yields an empty event list and empty head.

    Notes:
        Changing this returned dictionary does not change the original log. Events contain parameters, inputs/outputs, timing, parent IDs, and environment references. See [`replay`][msmu.pv.replay] and [`to_env`][msmu.pv.to_env].

    Examples:
        ```python
        import msmu as mm
        history = mm.pv.get_log(mdata)
        for event in history["events"]:
            print(event["function_path"])
        ```
    """
    log = _read_log(mdata)
    return {
        "schema_version": int(log["schema_version"]),
        "head": log["head"],
        "events": sorted(
            (json.loads(v) for v in log["events"].values()), key=lambda e: (e["sequence"], e["started_at"], e["id"])
        ),
        "environments": {key: json.loads(value) for key, value in log["environments"].items()},
    }


def to_env(
    history: md.MuData | dict, filename: str | Path | None = None,
    *, format: str = "uv", environment_id: str | None = None,
) -> str | None:
    """Export recorded Python/package versions for uv or conda, without installing.

    ``format="uv"`` returns requirements.txt text; ``format="conda"`` returns
    environment.yml text with CPython/pip from conda-forge and Python packages
    in its pip section. With filename, write UTF-8 text (overwriting an existing
    file) and return None, as [`to_script`][msmu.pv.to_script] does.

    Accepts MuData or [`get_log()`][msmu.pv.get_log] output, independently of [`replay`][msmu.pv.replay] support or hashes.
    By default, all referenced environments must agree on Python, package pins,
    platform and `msmu` source. Otherwise select a recorded environment_id.
    Numerical/thread settings may differ and are included as comments only.

    These are version pins, not lockfiles: original package sources, conda builds,
    native dependencies and uncommitted code cannot be reconstructed. Only
    recorded CPython environments are supported. The supplied history is unchanged.

    Parameters:
        history: MuData or decoded history from [`get_log`][msmu.pv.get_log].
        filename: Optional output file path; an existing file is overwritten.
        format: `"uv"` for requirements.txt text or `"conda"` for environment.yml text.
        environment_id: Recorded environment ID to export. If omitted, all referenced environments must agree on the required version/source fields.

    Returns:
        Environment specification string when no filename is supplied; otherwise None.

    Notes:
        Use [`get_log`][msmu.pv.get_log] to find environment IDs. This does not install packages or mutate history.

    Examples:
        ```python
        import msmu as mm
        requirements = mm.pv.to_env(mdata, format="uv")
        print(requirements)
        ```
    """
    if format not in ("uv", "conda"):
        raise ValueError("format must be 'uv' or 'conda'")
    history = get_log(history) if isinstance(history, md.MuData) else history
    if not isinstance(history, dict) or history.get("schema_version") != 1:
        raise ValueError("Expected a MuData object or a decoded log from get_log()")
    environments = history.get("environments")
    if not isinstance(environments, dict) or not environments:
        raise ValueError("No recorded environments to export")
    if environment_id is None:
        events = history.get("events")
        if not isinstance(events, list) or not events:
            raise ValueError("No recorded events; select an environment_id explicitly")
        if any(not isinstance(event, dict) or not isinstance(event.get("environment_id"), str) for event in events):
            raise ValueError("Each event must reference a recorded environment_id")
        ids = list(dict.fromkeys(event["environment_id"] for event in events))
    else:
        if not isinstance(environment_id, str):
            raise TypeError("environment_id must be a string")
        ids = [environment_id]

    specifications = []
    contexts = []
    for key in ids:
        environment = environments.get(key)
        if not isinstance(environment, dict):
            raise ValueError(f"Missing recorded environment: {key!r}")
        python = environment.get("python")
        if not isinstance(python, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+)?", python):
            raise ValueError("A recorded Python version is required")
        if environment.get("implementation") != "CPython":
            raise ValueError("Environment export currently supports recorded CPython environments only")
        packages = environment.get("packages")
        if not isinstance(packages, dict) or not packages:
            raise ValueError("Recorded package versions are required")
        pins = {}
        for name, version in packages.items():
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", name):
                raise ValueError(f"Invalid recorded package name: {name!r}")
            # Only single version tokens; never accept requirement options or URLs.
            if not isinstance(version, str) or not re.fullmatch(r"[0-9][A-Za-z0-9.!+_-]*", version):
                raise ValueError(f"Invalid recorded version for {name!r}: {version!r}")
            name = re.sub(r"[-_.]+", "-", name).lower()
            if name in pins and pins[name] != version:
                raise ValueError(f"Conflicting recorded versions for {name!r}")
            pins[name] = version
        source = environment.get("msmu_source", {})
        if not isinstance(source, dict):
            raise ValueError("Invalid recorded MSMU source metadata")
        specifications.append((python, pins, source, environment.get("os"), environment.get("architecture")))
        contexts.append({name: value for name, value in environment.items() if name != "packages"})
    if any(spec != specifications[0] for spec in specifications[1:]):
        raise ValueError(
            f"Multiple recorded environments differ; select environment_id from {ids!r}. "
            "One export cannot reproduce all of them."
        )
    python, pins, source, _, _ = specifications[0]
    lines = [
        "# Generated by MSMU from recorded provenance, not the current environment.",
        f"# Environment IDs: {_json(ids)}",
        "# Version pins only: original package sources, conda builds and native dependencies are not captured.",
        "# Runtime settings and uncommitted source edits are not restored by this file.",
    ]
    lines.extend(f"# Recorded context: {_json(context)}" for context in contexts)
    if source.get("dirty"):
        get_logger().warning(
            "Recorded MSMU source has uncommitted changes. Package version pins cannot restore them; "
            "preserve and install the matching source separately.",
        )
    requirements = [f"{name}=={version}" for name, version in sorted(pins.items())]
    if format == "uv":
        lines.extend([
            f"# Create: uv venv --python {python} .venv-replay",
            "# Install: uv pip sync --python .venv-replay requirements.txt",
            "", *requirements,
        ])
    else:
        lines.extend([
            "# Create: conda env create --name msmu-replay --file environment.yml",
            "# Python packages use pip; their conda channels/builds were not recorded.",
            "channels:", "  - conda-forge", "dependencies:",
            f"  - python={python}", "  - pip", "  - pip:",
            *(f"      - {_json(requirement)}" for requirement in requirements),
        ])
    text = "\n".join(lines) + "\n"
    if filename is not None:
        Path(filename).write_text(text, encoding="utf-8")
        return None
    return text


def _merge_logs(target, source):
    for group in ("events", "environments"):
        for key, value in source[group].items():
            if key in target[group] and target[group][key] != value:
                raise ValueError(f"Conflicting provenance entry: {key}")
            target[group][key] = value


def _serialize_parameters(obj, _seen=None, *, store_data=False, entities=None, role="arguments"):
    """Serialize options; explicitly captured arguments also retain column data."""
    if not store_data and entities is not None and role in entities:
        return entities[role]
    if store_data:
        encode = functools.partial(_serialize_parameters, store_data=True)
        if obj is pd.NA or obj is pd.NaT:
            return {"type": "pandas.NA" if obj is pd.NA else "pandas.NaT"}
        if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return {"type": type(obj).__name__, "value": obj.isoformat()}
        if isinstance(obj, np.generic):
            return {"type": "numpy.scalar", "array": encode(np.asarray(obj))}
        if isinstance(obj, pd.MultiIndex):
            raise TypeError("MultiIndex column values are not supported")
        if isinstance(obj, (pd.Series, pd.Index, np.ndarray)):
            dtype = obj.dtype
            if isinstance(dtype, pd.CategoricalDtype):
                dtype = {"categories": encode(dtype.categories), "ordered": dtype.ordered}
            else:
                dtype = str(dtype)
            result = {"type": type(obj).__name__ if isinstance(obj, np.ndarray) else
                      "pandas.Series" if isinstance(obj, pd.Series) else "pandas.Index",
                      "values": encode(obj.tolist()), "dtype": dtype}
            if isinstance(obj, (pd.Series, pd.Index)):
                result["name"] = encode(obj.name)
            if isinstance(obj, pd.Series):
                result["index"] = encode(obj.index)
            return result
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else {"type": "float", "value": str(obj)}
    if isinstance(obj, np.generic):
        return _serialize_parameters(obj.item(), store_data=store_data)
    if isinstance(obj, os.PathLike):
        return os.fsdecode(obj)
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return {"type": type(obj).__name__, "value": obj.isoformat()}
    if isinstance(obj, md.MuData) and not store_data:
        return {"type": "MuData", "source_event": _read_log(obj)["head"], "hash": None}
    if isinstance(obj, (md.MuData, ad.AnnData, pd.DataFrame, pd.Series, pd.Index, np.ndarray)) or sparse.issparse(obj):
        if store_data:
            raise TypeError(f"Cannot store parameter type {type(obj).__name__}")
        return {"type": type(obj).__name__, "hash": None}
    if callable(obj):
        if store_data:
            raise TypeError("Cannot store callable column values")
        return {
            "type": "callable",
            "path": f"{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', type(obj).__qualname__)}",
            "replayable": False,
        }
    seen = set() if _seen is None else _seen
    if id(obj) in seen:
        if store_data:
            raise ValueError("Cannot store cyclic column values")
        return {"type": "reference", "reason": "cyclic parameter", "replayable": False}
    seen.add(id(obj))
    try:
        if isinstance(obj, Mapping):
            items = [(k, _serialize_parameters(v, seen, store_data=store_data, entities=entities, role=f"{role}/{k}")) for k, v in obj.items()]
            items = [(k, v) for k, v in items if v is not _OMIT]
            if obj and not items:
                return _OMIT
            if all(isinstance(k, str) for k in obj):
                return dict(items)
            return {"type": "mapping", "items": [[_serialize_parameters(k, seen, store_data=store_data), v] for k, v in items]}
        if isinstance(obj, (list, tuple, set, frozenset)):
            values = [_serialize_parameters(v, seen, store_data=store_data, entities=entities, role=f"{role}/{i}") for i, v in enumerate(obj)]
            values = [v for v in values if v is not _OMIT]
            if obj and not values:
                return _OMIT
            return sorted(values, key=_json) if isinstance(obj, (set, frozenset)) else values
        if store_data:
            raise TypeError(f"Cannot store parameter type {type(obj).__name__}")
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
    entity: dict[str, Any] = {"id": str(uuid4()), "role": path, "type": type(value).__name__, "hash": {"status": "disabled"}}
    if isinstance(value, md.MuData) and path.startswith("arguments/"):
        entity["source_event"] = _read_log(value)["head"]
    if isinstance(value, Path) or is_url(value):
        entity["path"] = str(value)
    if is_url(value):
        entity["type"] = "URL"
    if hashing:
        hash_info: dict[str, Any]
        started = time.perf_counter()
        try:
            content = get_download_buffer(value) if is_url(value) else value
            if content is None:
                raise ValueError("URL content was not read through the shared input loader")
            precision = None if isinstance(content, (Path, BytesIO)) else _significant_digits.get()
            hash_info = {"status": "completed", "algorithm": ALGORITHM,
                              "value": compute_hash(content, significant_digits=precision, normalization=_normalization.get())}
            if precision is not None:
                hash_info["normalization"] = _normalization.get()
                if _normalization.get() != HASH_POLICY:
                    hash_info["significant_digits"] = precision
        except Exception as error:
            hash_info = {"status": "unavailable", "algorithm": ALGORITHM, "reason": str(error)}
        hash_info["duration_seconds"] = time.perf_counter() - started
        entity["hash"] = hash_info
    return entity


def _parameter_references(value, role="arguments"):
    """Find complex-argument references without storing their data in parameters."""
    if isinstance(value, dict):
        if "type" in value and "hash" in value and set(value) <= {"id", "role", "type", "hash", "source_event", "path"}:
            yield value.get("role", role), value
        else:
            for key, item in value.items():
                yield from _parameter_references(item, f"{role}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _parameter_references(item, f"{role}/{index}")


def _event_inputs(event):
    """Read parameter descriptors, falling back to the previous inputs layout."""
    if "inputs" in event:
        return event["inputs"]
    return [reference for _, reference in _parameter_references(event["parameters"])
            if "id" in reference and "role" in reference]


def log_provenance(func=None, *, capture=()):
    """Log one public call. Nested `msmu` calls are represented by their outer call.

    Parameters:
        func (Callable | None): Function to decorate. Omit when using [`@mm.pv.log(capture=(... ,))`][msmu.pv.log].
        capture (tuple[str, ...]): Tuple of parameter names whose values are stored in full rather than summarized, for example `("values",)`.

    Returns:
        wrapped (Callable): A wrapped function, or a decorator when `func` is omitted. Successful calls attach history to participating MuData objects.

    Notes:
        Nested recorded calls are represented by the outer call. Full captures may increase history size. Recording a custom function does not make it replayable: [`replay`][msmu.pv.replay] only executes supported public `msmu` functions. [`assign`][msmu.dt.assign] is the supported operation for recording column values.

    Examples:
        ```python
        import msmu as mm
        @mm.pv.log(capture=("label",))
        def annotate(mdata, label):
            mdata.uns["label"] = label
            return mdata
        ```
    """

    if func is None:
        return functools.partial(log_provenance, capture=capture)

    def run(*args, **kwargs):
        if _active.get():
            return func(*args, **kwargs)
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = list(_entities({k: v for k, v in bound.arguments.items() if k not in capture}, "arguments"))
        owners = [v for _, v in inputs if isinstance(v, md.MuData)]
        log = _empty_log()
        parents = []
        for owner in owners:
            previous = _read_log(owner)
            _merge_logs(log, previous)
            if previous["head"]:
                parents.append(previous["head"])
        hashing = _hashing.get()
        input_entities = [_entity(path, value, hashing) for path, value in inputs]
        entities_by_role = {entity["role"]: entity for entity in input_entities}
        event = {
            "id": str(uuid4()),
            "function": func.__name__,
            "function_path": f"{func.__module__}.{func.__qualname__}",
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "parents": list(dict.fromkeys(parents)),
            "parameters": {
                key: value for key, argument in bound.arguments.items()
                if (value := _serialize_parameters(argument, store_data=key in capture, entities=entities_by_role, role=f"arguments/{key}")) is not _OMIT
            },
            "outputs": [],
            "hashing": hashing,
        }
        if hashing:
            for entity, (_, value) in zip(input_entities, inputs):
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
                    and previous_hash.get("normalization") == current_hash.get("normalization")
                    and previous_hash.get("significant_digits") == current_hash.get("significant_digits")
                    and previous_hash.get("value") != current_hash["value"]
                ):
                    get_logger().warning(
                        "Data changed outside the recorded workflow.\n"
                        f"  Between: {head['function']} → {func.__name__}\n"
                        "  If this step needs MSMU support, please open an issue describing the operation:\n"
                        "  https://github.com/bertis-informatics/msmu/issues",
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
                for entity, (path, value) in zip(input_entities, inputs):
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
            log["events"][event["id"]] = json.dumps(event, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
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
