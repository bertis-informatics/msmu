"""Recorded data manipulation for MuData tables and metadata."""

from typing import Literal

import pandas as pd
from mudata import MuData

from ._core._provenance import log_provenance

__all__ = ["assign", "map", "replace", "drop"]


def _column_table(mdata: MuData, path: str) -> pd.DataFrame:
    parts = path.split(".", 2)
    if path in ("obs", "var"):
        table = getattr(mdata, path)
    elif len(parts) == 2 and parts[1] in ("obs", "var"):
        table = getattr(mdata[parts[0]], parts[1])
    elif len(parts) == 3 and parts[1] in ("obsm", "varm"):
        table = getattr(mdata[parts[0]], parts[1])[parts[2]]
    else:
        raise ValueError(f"Invalid table path: {path!r}")
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"{path!r} must contain a pandas DataFrame")
    if not table.columns.is_unique:
        raise ValueError(f"{path!r} must have unique column names")
    return table


@log_provenance
def map(
    mdata: MuData,
    *,
    source: str,
    target: str,
    columns: dict[str, str],
    source_index: str | None = None,
    target_index: str | None = None,
) -> MuData:
    """Map source columns onto a target table in place, returning the same MuData.

    Table paths are ``obs``, ``var``, ``<modality>.obs/var``, or
    ``<modality>.obsm/varm.<key>`` (DataFrames only). Index arguments name
    matching columns; None uses the table index without changing it.
    ``columns`` maps source column names to destination column names.
    Identical source key/value rows are collapsed; conflicting keys raise.
    Unmatched rows receive missing values. Existing destination columns are
    overwritten. Matching follows pandas, including matching missing keys.
    Provenance records these instructions, not the mapped column contents.
    """
    if not isinstance(columns, dict) or not columns or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in columns.items()
    ):
        raise ValueError("columns must be a nonempty mapping of column names")
    if len(set(columns.values())) != len(columns):
        raise ValueError("Destination column names must be unique")
    src, dst = _column_table(mdata, source), _column_table(mdata, target)
    mapping = src[list(columns)].copy()
    key = object()  # Temporary label cannot collide with a source column name.
    mapping[key] = src.index if source_index is None else src[source_index].array
    mapping = mapping.drop_duplicates().set_index(key)
    if not mapping.index.is_unique:
        raise ValueError("Source keys have conflicting values in the selected columns")
    keys = dst.index if target_index is None else dst[target_index].array
    keys = pd.Series(keys, index=dst.index)
    # Resolve every column before writing, including when source and target overlap.
    mapped = {destination: keys.map(mapping[column]) for column, destination in columns.items()}
    for destination, values in mapped.items():
        dst[destination] = values
    return mdata


@log_provenance
def replace(
    mdata: MuData,
    *,
    target: str,
    columns: dict[str, dict],
) -> MuData:
    """Replace column values in place using pandas-style replacement mappings.

    ``columns`` maps column names to ``{old: new}`` rules. Unmatched values
    (including missing values unless explicitly mapped) remain unchanged.
    Restricted dtypes such as nullable booleans and categories are converted
    to object when needed to accommodate new values. Only rules are recorded.
    Table paths follow :func:`map`. Returns the same MuData.
    """
    if not isinstance(columns, dict) or not columns or not all(
        isinstance(name, str) and isinstance(rules, dict) for name, rules in columns.items()
    ):
        raise ValueError("columns must map column names to replacement dictionaries")
    table = _column_table(mdata, target)
    replaced = {}
    for name, rules in columns.items():
        column = table[name]
        try:
            replaced[name] = column.replace(rules)
        except TypeError:
            replaced[name] = column.astype(object).replace(rules)
    for name, values in replaced.items():
        table[name] = values
    return mdata


@log_provenance
def drop(mdata: MuData, *, target: str, key: str) -> MuData:
    """Delete an uns entry in place, recording only its location for replay.

    ``target`` is ``uns`` or ``<modality>.uns``. Missing keys raise KeyError,
    just like ``del``. Returns the same MuData; provenance cannot be deleted.
    """
    if key == "_log":
        raise ValueError("Cannot delete provenance with drop")
    if target == "uns":
        container = mdata.uns
    elif target.endswith(".uns"):
        container = mdata[target[:-4]].uns
    else:
        raise ValueError("target must be 'uns' or '<modality>.uns'")
    del container[key]
    return mdata


@log_provenance(capture=("values",))
def assign(
    mdata: MuData,
    column: str,
    values,
    *,
    modality: str,
    on: Literal["obs", "var"] = "var",
) -> MuData:
    """Assign a column in place and record its values for replay.

    Equivalent to ``mdata[modality].var[column] = values`` (or ``obs``).
    Pandas handles scalar broadcasting, positional arrays and Series index
    alignment. Returns the same MuData. Values are stored in full in provenance;
    the computation that produced them is not recorded.
    """
    if on not in ("obs", "var"):
        raise ValueError("on must be 'obs' or 'var'")
    getattr(mdata[modality], on)[column] = values
    return mdata
