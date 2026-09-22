"""Replay file-backed MSMU workflows, including recorded branches and merges."""

from copy import deepcopy
import datetime
import inspect
from collections import Counter
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any

import mudata as md
import numpy as np
import pandas as pd

from ..logging_utils import get_logger

from ._hashing import ALGORITHM, FLOAT_NORMALIZATION, _validate_precision, compute_hash
from ._provenance import _environment, _serialize_parameters, _parameter_references, _event_inputs, get_log, options
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


def _decode(value) -> Any:
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
        dtype: Any = value["dtype"]
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
    reachable = set()
    pending = [history.get("head")]
    while pending:
        current = pending.pop()
        if current not in by_id:
            raise ValueError("Invalid or cyclic provenance history")
        if current in reachable:
            continue
        reachable.add(current)
        pending.extend(by_id[current]["parents"])
    try:
        order = TopologicalSorter({key: event["parents"] for key, event in by_id.items()}).static_order()
        chain = [by_id[key] for key in order]
    except CycleError as error:
        raise ValueError("Invalid or cyclic provenance history") from error
    if reachable != set(by_id):
        raise ValueError("Replay requires all events to be ancestors of the history head")
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
        mudata_inputs = [entity for entity in _event_inputs(event) if entity["type"] == "MuData"]
        references = dict(_parameter_references(parameters))
        if func.__name__ == "merge_mudata":
            if any(not isinstance(key, str) for key in parameters.get("mdatas", {})):
                raise ValueError("Replay requires string dataset names in merge_mudata")
            parameters["mdatas"] = {}  # Input-entity order preserves first-value precedence.
        bindings = {}
        parents = event["parents"]
        if not parents:
            if mudata_inputs or func.__name__ not in (
                "read_sage", "read_diann", "read_maxquant", "read_fragpipe", "read_delpi", "read_h5mu"
            ):
                raise ValueError("Replay must start with a file reader; the original input is not stored in the log")
        elif not mudata_inputs:
            raise ValueError("Replay does not support intermediate readers without MuData inputs")
        for entity in mudata_inputs:
            role = entity["role"]
            if func.__name__ == "merge_mudata" and role.startswith("arguments/mdatas/"):
                parameters.setdefault("mdatas", {})[role.removeprefix("arguments/mdatas/")] = None
            elif role.startswith("arguments/") and role.count("/") == 1 and len(mudata_inputs) == 1:
                parameters[role.split("/")[1]] = None
            else:
                raise ValueError(f"Replay requires one direct MuData input or merge_mudata: {event['function']}")
            reference = references.get(role)
            producer = reference.get("source_event") if reference is not None else None
            if reference is not None and {key: value for key, value in reference.get("hash", {}).items() if key != "duration_seconds"} != {
                key: value for key, value in entity.get("hash", {}).items() if key != "duration_seconds"
            }:
                raise ValueError(f"Unrecorded data changes or inconsistent parameter hash before {event['function']}")
            if reference is None:
                # Older linear logs are unambiguous; older merges need unique hashes.
                candidates = parents if len(parents) == 1 else [
                    parent for parent in parents
                    if _digest(entity) and _digest(by_id[parent]["outputs"][0]) == _digest(entity)
                ]
                if len(candidates) != 1:
                    raise ValueError("Ambiguous input producer; record the workflow again to capture source_event")
                producer = candidates[0]
            if producer not in parents or role in bindings:
                raise ValueError("MuData input producers do not match event parents")
            bindings[role] = producer
            previous = by_id[producer]["outputs"][0]
            previous_info, input_info = previous.get("hash", {}), entity.get("hash", {})
            if _digest(previous) and _digest(entity) and any(
                previous_info.get(key) != input_info.get(key) for key in ("normalization", "significant_digits")
            ):
                raise ValueError("Replay cannot verify a change of hash precision between consecutive steps")
            if _digest(previous) and _digest(entity) and _digest(previous) != _digest(entity):
                raise ValueError(f"Unrecorded data changes before {event['function']} cannot be replayed")
        if set(bindings.values()) != set(parents):
            raise ValueError("MuData input producers do not match event parents")
        files = []
        for entity in _event_inputs(event):
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
        if not parents and not files:
            raise ValueError("Replay requires an original input file")
        if verify and any(not _digest(entity) for entity in _event_inputs(event) + outputs):
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
        plan.append((event, func, bound, bindings, files))
    unused = sources.keys() - used_sources
    if unused:
        raise ValueError(f"Source replacements do not match recorded inputs: {sorted(unused)}")
    return plan


def replay(history: md.MuData | dict, *, sources: dict | None = None, verify: bool = True) -> md.MuData:
    """Rerun a file-backed workflow, returning a new MuData with fresh history.

    Args:
        history: MuData containing the original log, or the decoded result of get_log().
        sources: Optional mapping from recorded paths/URLs to replacement paths/URLs.
        verify: Require recorded hashes and verify inputs/outputs (default True).
            False reruns without hash verification; known unrecorded edits still fail.

    Branches and merge_mudata are supported. Shared results are copied at forks.
    Other nested MuData inputs and non-MuData returns are unsupported.
    Only public MSMU functions can run. Environment differences are logged as warnings;
    packages are never installed or changed. The supplied object/log is not modified.
    """
    if not isinstance(verify, bool):
        raise TypeError("verify must be a bool")
    history = get_log(history) if isinstance(history, md.MuData) else deepcopy(history)
    plan = _plan(history, dict(sources or {}), verify)
    _check_environment([history.get("environments", {}).get(event["environment_id"]) for event, *_ in plan])
    results = {}
    uses = Counter(producer for _, _, _, bindings, _ in plan for producer in bindings.values())
    remaining = uses.copy()
    for event, func, bound, bindings, files in plan:
        previous_heads = []
        for role, producer in bindings.items():
            value = results[producer].copy() if uses[producer] > 1 else results[producer]
            # Merge dataset names may themselves contain '/'.
            if role.startswith("arguments/mdatas/"):
                bound.arguments["mdatas"][role.removeprefix("arguments/mdatas/")] = value
            else:
                _set_source(bound.arguments, role, value)
            previous_heads.append(get_log(value)["head"])
        with source_scope(), options(hashing=verify, significant_digits=_precision(event)):
            if verify:
                for entity, source in files:
                    with open_source(source) as buffer:
                        _check_digest(compute_hash(buffer if is_url(source) else Path(source)), entity, event)
                for entity in _event_inputs(event):
                    if entity["type"] == "MuData":
                        _check_digest(compute_hash(results[bindings[entity["role"]]], significant_digits=_precision(event)), entity, event)
            result = func(*bound.args, **bound.kwargs)
            if not isinstance(result, md.MuData):
                raise ValueError(f"Replay expected a MuData return from {event['function']}")
            if verify:
                _verify_output(result, event, previous_heads)
            results[event["id"]] = result
        bound.arguments.clear()
        value = None
        for producer in bindings.values():
            remaining[producer] -= 1
            if not remaining[producer]:
                del results[producer]
    return results[history["head"]]


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
    previous_heads = previous_head if isinstance(previous_head, list) else [previous_head]
    if not fresh["head"] or fresh["head"] in [*previous_heads, event["id"]]:
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

    Accepts the same file-backed histories and options as replay(). The generated script
    requires MSMU, checks the recorded environment at execution, and leaves its
    final MuData in ``mdata``. Scripts enable hashing even when verification is disabled.
    Generation neither reads inputs nor executes calls.
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
        f"_check_environment({_literal(environments)})", "", "mm.pv.set_options(hashing=True)", "mdata = None",
    ]
    def script_entity(entity):
        return {
            **entity,
            "hash": {key: value for key, value in entity.get("hash", {}).items() if key != "duration_seconds"},
        }

    uses = Counter(producer for _, _, _, bindings, _ in plan for producer in bindings.values())
    remaining = uses.copy()
    graph = any(len(bindings) > 1 for _, _, _, bindings, _ in plan) or any(count > 1 for count in uses.values())
    variables = {event["id"]: f"mdata_{index}" if graph else "mdata" for index, (event, *_) in enumerate(plan)}
    for event, func, bound, bindings, files in plan:
        metadata = {key: event[key] for key in ("id", "function", "function_path")}
        metadata["outputs"] = [script_entity(entity) for entity in event["outputs"]]
        context = "source_scope()"
        if verify and _precision(event) != 12:
            context += f", mm.pv.options(hashing=True, significant_digits={_precision(event)!r})"
        lines.extend(["", f"# {names[func]}", f"with {context}:"])
        if verify:
            lines.append(f"    event = {_literal(metadata)}")
            parent_variables = [variables[producer] for producer in dict.fromkeys(bindings.values())]
            lines.append("    previous_head = [" + ", ".join(f'mm.pv.get_log({value})["head"]' for value in parent_variables) + "]")
            for entity in _event_inputs(event):
                if entity["type"] == "MuData":
                    variable = variables[bindings[entity["role"]]]
                    lines.append(f"    _check_digest(mm.pv.compute_hash({variable}, significant_digits={_precision(event)!r}), {_literal(script_entity(entity))}, event)")
            for entity, source in files:
                lines.extend([
                    f"    source = {_literal(source)}",
                    "    with open_source(source) as buffer:",
                    f"        _check_digest(mm.pv.compute_hash(buffer if is_url(source) else Path(source)), {_literal(script_entity(entity))}, event)",
                ])
        arguments = []
        for name, parameter in bound.signature.parameters.items():
            direct_role = f"arguments/{name}"
            def input_expression(producer):
                return variables[producer] + (".copy()" if uses[producer] > 1 else "")
            if direct_role in bindings:
                value = input_expression(bindings[direct_role])
            elif name == "mdatas" and bindings:
                value = "{" + ", ".join(
                    f"{_literal(role.removeprefix('arguments/mdatas/'))}: {input_expression(producer)}"
                    for role, producer in bindings.items()
                ) + "}"
            else:
                value = _literal(bound.arguments[name])
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                arguments.append("*" + value)
            elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
                arguments.append("**" + value)
            elif parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
                arguments.append(value)
            else:
                arguments.append(f"{name}={value}")
        variable = variables[event["id"]]
        lines.append(f"    {variable} = {names[func]}({', '.join(arguments)})")
        if verify:
            lines.append(f"    _verify_output({variable}, event, previous_head)")
        for producer in bindings.values():
            remaining[producer] -= 1
            if graph and not remaining[producer]:
                lines.append(f"    del {variables[producer]}")
    if graph:
        lines.append(f"\nmdata = {variables[history['head']]}")
    script = "\n".join(lines) + "\n"
    if filename is not None:
        Path(filename).write_text(script, encoding="utf-8")
        return None
    return script
