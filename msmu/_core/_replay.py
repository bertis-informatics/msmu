"""Replay a linear, file-backed MSMU workflow from its recorded history."""

from copy import deepcopy
import datetime
import inspect
from pathlib import Path

import mudata as md
import numpy as np
import pandas as pd

from ..logging_utils import get_logger

from ._hashing import ALGORITHM, FLOAT_NORMALIZATION, _validate_precision, compute_hash
from ._provenance import _environment, _serialize_parameters, get_log, options
from ._sources import is_url, open_source, source_scope


def _functions():
    # Resolve only public MSMU functions; never import code named by a log.
    import msmu as mm

    functions = {
        f"{func.__module__}.{func.__qualname__}": func
        for module in (mm, mm.io, mm.pp, mm.dt, mm.tl, mm.utils)
        for name in module.__all__
        if inspect.isfunction(func := getattr(module, name)) and hasattr(func, "__wrapped__")
    }

    # Resolve recorded paths from earlier API names without exposing public aliases.
    functions.update({
        f"{module}.{old}": getattr(mm.dt, new)
        for module in ("msmu._preprocessing._meta", "msmu.dt", "msmu._data")
        for old, new in {
            "set_column": "assign", "map_column": "map",
            "replace_values": "replace", "drop_key": "drop",
        }.items()
    })
    return functions


def _decode(value):
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("replayable") is False:
        raise ValueError("A parameter was recorded as non-replayable")
    if value == {"type": "pandas.NA"}:
        return pd.NA
    if value == {"type": "pandas.NaT"}:
        return pd.NaT
    if set(value) == {"type", "array"} and value["type"] == "numpy.scalar":
        return _decode(value["array"])[()]
    if value.get("type") in ("pandas.Series", "pandas.Index", "ndarray") and "values" in value:
        dtype = value["dtype"]
        if isinstance(dtype, dict):
            dtype = pd.CategoricalDtype(_decode(dtype["categories"]), ordered=dtype["ordered"])
        values = _decode(value["values"])
        if value["type"] == "ndarray":
            return np.array(values, dtype=dtype)
        if value["type"] == "pandas.Index":
            return pd.Index(values, dtype=dtype, name=_decode(value["name"]))
        return pd.Series(values, index=_decode(value["index"]), dtype=dtype, name=_decode(value["name"]))
    if set(value) == {"type", "value"}:
        if value["type"] == "float":
            return float(value["value"])
        if value["type"] in ("Timestamp", "Timedelta"):
            return getattr(pd, value["type"])(value["value"])
        if value["type"] in ("date", "datetime"):
            return getattr(datetime, value["type"]).fromisoformat(value["value"])
    if set(value) == {"type", "items"} and value["type"] == "mapping":
        return {_decode(key): _decode(item) for key, item in value["items"]}
    return {key: _decode(item) for key, item in value.items()}


def _set_source(parameters, role, value):
    parts = role.split("/")
    if parts.pop(0) != "arguments":
        raise ValueError(f"Unsupported input role: {role}")
    target = parameters
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    key = int(parts[-1]) if isinstance(target, list) else parts[-1]
    target[key] = value


def _digest(entity):
    info = entity.get("hash", {})
    if info.get("status") in ("completed", "computed") and info.get("algorithm") == ALGORITHM:
        return info.get("value")
    return None


def _precision(event):
    info = event["outputs"][0].get("hash", {})
    normalization = info.get("normalization")
    if normalization is None:
        if "significant_digits" in info:
            raise ValueError("Hash precision requires a normalization version")
        return None  # Existing histories used exact hashes.
    if normalization != FLOAT_NORMALIZATION or info.get("significant_digits") is None:
        raise ValueError(f"Unsupported hash normalization: {normalization}")
    precision = info["significant_digits"]
    _validate_precision(precision)
    return precision


def _check_digest(actual, entity, event):
    if actual != _digest(entity):
        raise ValueError(
            f"Replay hash mismatch at {event['function']} ({entity['role']}). "
            "The input data, processing result, or execution conditions differ from the recording."
        )


def _plan(history, sources, verify, *, check_files=True):
    if history.get("schema_version") != 1 or not isinstance(history.get("events"), list):
        raise ValueError("Expected a MuData object or a decoded log from get_log()")
    events = history["events"]
    by_id = {event["id"]: event for event in events}
    if not events or len(by_id) != len(events):
        raise ValueError("Replay requires a nonempty history with unique event IDs")
    chain = []
    current = history.get("head")
    seen = set()
    while current:
        if current not in by_id or current in seen:
            raise ValueError("Invalid or cyclic provenance history")
        seen.add(current)
        event = by_id[current]
        chain.append(event)
        parents = event["parents"]
        if len(parents) > 1:
            raise ValueError("Replay currently supports a single MuData chain; merged histories are unsupported")
        current = parents[0] if parents else None
    if len(chain) != len(events):
        raise ValueError("Replay currently supports a single MuData chain without branches")
    chain.reverse()
    functions = _functions()
    plan = []
    used_sources = set()
    for index, event in enumerate(chain):
        func = functions.get(event["function_path"])
        if func is None:
            raise ValueError(f"Replay does not support function {event['function_path']}")
        outputs = event["outputs"]
        if len(outputs) != 1 or outputs[0]["type"] != "MuData" or outputs[0]["role"] != "return":
            raise ValueError(f"Replay requires one returned MuData per call: {event['function']}")
        _precision(event)
        parameters = _decode(event["parameters"])
        mudata_inputs = [entity for entity in event["inputs"] if entity["type"] == "MuData"]
        if index == 0:
            if mudata_inputs or func.__name__ not in (
                "read_sage", "read_diann", "read_maxquant", "read_fragpipe", "read_delpi", "read_h5mu"
            ):
                raise ValueError("Replay must start with a file reader; the original input is not stored in the log")
        elif len(mudata_inputs) != 1 or mudata_inputs[0]["role"].count("/") != 1:
            raise ValueError(f"Replay requires one direct MuData input: {event['function']}")
        mudata_argument = mudata_inputs[0]["role"].split("/")[1] if mudata_inputs else None
        if mudata_argument:
            previous_info = chain[index - 1]["outputs"][0].get("hash", {})
            input_info = mudata_inputs[0].get("hash", {})
            if _digest(chain[index - 1]["outputs"][0]) and _digest(mudata_inputs[0]) and any(
                previous_info.get(key) != input_info.get(key) for key in ("normalization", "significant_digits")
            ):
                raise ValueError("Replay cannot verify a change of hash precision between consecutive steps")
            previous_hash = _digest(chain[index - 1]["outputs"][0])
            input_hash = _digest(mudata_inputs[0])
            if previous_hash and input_hash and previous_hash != input_hash:
                raise ValueError(f"Unrecorded data changes before {event['function']} cannot be replayed")
            parameters[mudata_argument] = None  # Replaced by the previous result during execution.
        files = []
        for entity in event["inputs"]:
            if entity["type"] == "MuData":
                continue
            if "path" not in entity:
                raise ValueError(f"Input content is not stored: {event['function']} ({entity['role']})")
            original = entity["path"]
            source = sources.get(original, original)
            used_sources.add(original)
            _set_source(parameters, entity["role"], source)
            if check_files and not is_url(source) and not Path(source).is_file():
                raise ValueError(f"Replay input file does not exist: {source}")
            files.append((entity, source))
        if index == 0 and not files:
            raise ValueError("Replay requires an original input file")
        if verify and any(not _digest(entity) for entity in event["inputs"] + outputs):
            raise ValueError(
                f"Recorded hashes are missing for {event['function']}. "
                "Use verify=False to rerun without hash verification."
            )
        if func.__name__ in ("pca", "umap") and type(parameters.get("random_state")) is not int:
            raise ValueError(f"Replay requires a recorded integer random_state for {func.__name__}")
        signature = inspect.signature(func)
        unknown = parameters.keys() - signature.parameters.keys()
        missing = signature.parameters.keys() - parameters.keys()
        if unknown or missing:
            raise ValueError(
                f"Recorded parameters no longer match {event['function']}: "
                f"unknown={sorted(unknown)}, missing={sorted(missing)}"
            )
        bound = inspect.BoundArguments(signature, parameters)
        signature.bind(*bound.args, **bound.kwargs)
        plan.append((event, func, bound, mudata_argument, files))
    unused = sources.keys() - used_sources
    if unused:
        raise ValueError(f"Source replacements do not match recorded inputs: {sorted(unused)}")
    return plan


def replay(history: md.MuData | dict, *, sources: dict | None = None, verify: bool = True) -> md.MuData:
    """Rerun a linear file-backed workflow, returning a new MuData with fresh history.

    Args:
        history: MuData containing the original log, or the decoded result of get_log().
        sources: Optional mapping from recorded paths/URLs to replacement paths/URLs.
        verify: Require recorded hashes and verify inputs/outputs (default True).
            False reruns without hash verification; known unrecorded edits still fail.

    Branches, merges, data-valued parameters and non-MuData returns are unsupported.
    Only public MSMU functions can run. Environment differences are logged as warnings;
    packages are never installed or changed. The supplied object/log is not modified.
    """
    if not isinstance(verify, bool):
        raise TypeError("verify must be a bool")
    history = get_log(history) if isinstance(history, md.MuData) else deepcopy(history)
    plan = _plan(history, dict(sources or {}), verify)
    _check_environment([history.get("environments", {}).get(event["environment_id"]) for event, *_ in plan])
    result = None
    for event, func, bound, mudata_argument, files in plan:
        previous_head = get_log(result)["head"] if result is not None else None
        if mudata_argument:
            bound.arguments[mudata_argument] = result
        with source_scope(), options(hashing=verify, significant_digits=_precision(event)):
            if verify:
                for entity, source in files:
                    with open_source(source) as buffer:
                        _check_digest(compute_hash(buffer if is_url(source) else Path(source)), entity, event)
                # Preflight matched this input to the preceding output, already verified below.
            result = func(*bound.args, **bound.kwargs)
            if not isinstance(result, md.MuData):
                raise ValueError(f"Replay expected a MuData return from {event['function']}")
            if verify:
                _verify_output(result, event, previous_head)
    return result


def _check_environment(environments):
    current_environment = _environment()
    differences = set()
    for recorded in environments:
        if recorded is None:
            differences.add("missing environment record")
        else:
            differences.update(key for key, value in current_environment.items() if recorded.get(key) != value)
            if recorded.get("msmu_source", {}).get("dirty"):
                differences.add("uncommitted source changes in recording")
    if differences:
        get_logger().warning(
            f"Replay environment may differ: {', '.join(sorted(differences))}. "
            "Recorded package versions and source code are not restored automatically.",
        )


def _verify_output(result, event, previous_head):
    fresh = get_log(result)
    if not fresh["head"] or fresh["head"] in (previous_head, event["id"]):
        raise ValueError(f"Replay could not record a new event for {event['function']}")
    recorded = next(item for item in fresh["events"] if item["id"] == fresh["head"])
    if recorded["function_path"] != event["function_path"] and (
        _functions().get(recorded["function_path"]) is not _functions().get(event["function_path"])
    ):
        raise ValueError(f"Replay could not verify the output of {event['function']}")
    _check_digest(_digest(recorded["outputs"][0]), event["outputs"][0], event)


def _literal(value):
    if isinstance(value, (pd.Series, pd.Index, np.ndarray, np.generic, pd.Timestamp, pd.Timedelta)) or value is pd.NA or value is pd.NaT:
        return f"_decode({_literal(_serialize_parameters(value, store_data=True))})"
    """Render supported values without trusting arbitrary repr implementations."""
    if value is None or type(value) in (bool, int, str):
        return repr(value)
    if type(value) is float:
        return f"float({str(value)!r})"
    if isinstance(value, Path):
        return repr(str(value))
    if type(value) in (datetime.date, datetime.datetime):
        return f"datetime.{type(value).__name__}.fromisoformat({value.isoformat()!r})"
    if type(value) in (list, tuple):
        items = ", ".join(_literal(item) for item in value)
        return "[" + items + "]" if type(value) is list else "(" + items + ",)" if value else "()"
    if type(value) is dict:
        return "{" + ", ".join(f"{_literal(key)}: {_literal(item)}" for key, item in value.items()) + "}"
    raise ValueError(f"Cannot generate a script for parameter type {type(value).__name__}")


def to_script(
    history: md.MuData | dict, filename: str | Path | None = None,
    *, sources: dict | None = None, verify: bool = True,
) -> str | None:
    """Generate editable Python calls with source/output hash checks.

    If filename is supplied, write UTF-8 Python source (overwriting an existing
    file) and return None. Otherwise return the source text.

    Accepts the same linear histories and options as replay(). The generated script
    requires MSMU, checks the recorded environment at execution, and leaves its
    final MuData in ``mdata``. Generation neither reads inputs nor executes calls.
    """
    import msmu as mm

    if not isinstance(verify, bool):
        raise TypeError("verify must be a bool")
    history = get_log(history) if isinstance(history, md.MuData) else deepcopy(history)
    plan = _plan(history, dict(sources or {}), verify, check_files=False)
    names = {}
    for prefix, module in (("mm", mm), ("mm.io", mm.io), ("mm.pp", mm.pp), ("mm.dt", mm.dt), ("mm.tl", mm.tl), ("mm.utils", mm.utils)):
        for name in module.__all__:
            func = getattr(module, name)
            if inspect.isfunction(func):
                names.setdefault(func, f"{prefix}.{name}")
    environments = [history.get("environments", {}).get(key) for key in
                    dict.fromkeys(event["environment_id"] for event, *_ in plan)]
    lines = [
        "# Generated by MSMU. Edit input paths and calls as needed.",
        "import datetime", "from pathlib import Path", "import msmu as mm",
        "from msmu._core._replay import _check_environment, _check_digest, _verify_output, _decode",
        "from msmu._core._sources import source_scope, open_source, is_url", "",
        f"_check_environment({_literal(environments)})", "", "mdata = None",
    ]
    def script_entity(entity):
        return {
            **entity,
            "hash": {key: value for key, value in entity.get("hash", {}).items() if key != "duration_seconds"},
        }

    for event, func, bound, mudata_argument, files in plan:
        metadata = {key: event[key] for key in ("id", "function", "function_path")}
        metadata["outputs"] = [script_entity(entity) for entity in event["outputs"]]
        lines.extend(["", f"# {names[func]}", f"with source_scope(), mm.pv.options(hashing={verify}, significant_digits={_precision(event)!r}):"])
        if verify:
            lines.append(f"    event = {_literal(metadata)}")
            lines.append('    previous_head = mm.pv.get_log(mdata)["head"] if mdata is not None else None')
            for entity, source in files:
                lines.extend([
                    f"    source = {_literal(source)}",
                    "    with open_source(source) as buffer:",
                    f"        _check_digest(mm.pv.compute_hash(buffer if is_url(source) else Path(source)), {_literal(script_entity(entity))}, event)",
                ])
        arguments = []
        for name, parameter in bound.signature.parameters.items():
            value = "mdata" if name == mudata_argument else _literal(bound.arguments[name])
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                arguments.append("*" + value)
            elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
                arguments.append("**" + value)
            elif parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
                arguments.append(value)
            else:
                arguments.append(f"{name}={value}")
        lines.append(f"    mdata = {names[func]}({', '.join(arguments)})")
        if verify:
            lines.append("    _verify_output(mdata, event, previous_head)")
    script = "\n".join(lines) + "\n"
    if filename is not None:
        Path(filename).write_text(script, encoding="utf-8")
        return None
    return script
