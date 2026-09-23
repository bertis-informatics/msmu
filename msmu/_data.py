"""Recorded data manipulation for MuData objects, tables and metadata."""

from copy import deepcopy
from typing import Any, Literal, cast

import anndata as ad
import mudata as md

import pandas as pd
from mudata import MuData

from ._core._provenance import log_provenance

__all__ = ["assign", "map", "replace", "drop", "concat", "save_layer", "load_layer"]


@log_provenance
def save_layer(
    mdata: MuData, *, modality: str, layer: str, overwrite: bool = False
) -> MuData:
    """Save a copy of a modality's `.X` in a named layer, in place.

    Parameters:
        mdata: MuData to modify.
        modality: Modality containing the matrix.
        layer: Destination layer name.
        overwrite: Replace an existing layer when True; otherwise raise ValueError.

    Returns:
        The same MuData with `.X` copied into `layers[layer]`.
    """
    if not isinstance(layer, str) or not layer:
        raise ValueError("layer must be a nonempty string")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")
    adata = mdata[modality]
    if adata.isbacked:
        raise ValueError("save_layer does not support backed modalities; read the .h5mu into memory")
    if adata.X is None:
        raise ValueError(f"{modality!r} has no .X to save")
    if layer in adata.layers and not overwrite:
        raise ValueError(f"Layer {layer!r} already exists in {modality!r}")
    adata.layers[layer] = adata.X.copy()
    return mdata


@log_provenance
def load_layer(mdata: MuData, *, modality: str, layer: str) -> MuData:
    """Load a named layer into a modality's `.X`, in place; other annotations are unchanged.

    Parameters:
        mdata: MuData to modify.
        modality: Modality containing the layer.
        layer: Source layer name; missing names raise KeyError.

    Returns:
        The same MuData with `layers[layer]` copied into `.X`.
    """
    if not isinstance(layer, str) or not layer:
        raise ValueError("layer must be a nonempty string")
    adata = mdata[modality]
    if adata.isbacked:
        raise ValueError("load_layer does not support backed modalities; read the .h5mu into memory")
    adata.X = adata.layers[layer].copy()
    return mdata


def _column_table(mdata: MuData, path: str) -> pd.DataFrame:
    parts = path.split(".", 2)
    if path in ("obs", "var"):
        table = getattr(mdata, path)
    elif path.startswith(("obsm.", "varm.")):
        attribute, key = path.split(".", 1)
        table = getattr(mdata, attribute)[key]
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
    ``[<modality>.]obsm/varm.<key>`` (DataFrames only). Index arguments name
    matching columns; None uses the table index without changing it.
    ``columns`` maps source column names to destination column names.
    Identical source key/value rows are collapsed; conflicting keys raise.
    Unmatched rows receive missing values. Existing destination columns are
    overwritten. Matching follows pandas, including matching missing keys.
    Provenance records these instructions, not the mapped column contents.

    Parameters:
        mdata: MuData to modify.
        source: Source table path: `obs`, `var`, `<modality>.obs`, `<modality>.var`, or `[<modality>.]obsm/varm.<key>` containing a DataFrame.
        target: Destination table path using the same syntax.
        columns: Nonempty mapping from source column names to unique destination column names.
        source_index: Source matching column; `None` uses its index.
        target_index: Destination matching column; `None` uses its index.

    Returns:
        The same MuData; destination columns are written in place.

    Examples:
        ```python
        import msmu as mm
        mm.dt.map(mdata, source="obs", target="protein.obs", columns={"condition": "condition"})
        ```
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
    Table paths follow [`map`][msmu.dt.map]. Returns the same MuData.

    Parameters:
        mdata: MuData to modify.
        target: Table path as described by [`map`][msmu.dt.map].
        columns: Nonempty mapping from each column name to an `{old_value: new_value}` dictionary.

    Returns:
        The same MuData, modified in place.

    Examples:
        ```python
        import msmu as mm
        mm.dt.replace(mdata, target="protein.obs", columns={"condition": {"ctrl": "control"}})
        ```
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
    """Delete a metadata entry or table column in place, recording its location for replay.

    ``target`` is ``uns``, ``obsm``, ``varm``, a modality-specific equivalent,
    or a table path accepted by [`map`][msmu.dt.map]. A mapping target deletes
    its ``key``; a table target deletes a column. Missing keys raise KeyError.
    Returns the same MuData; provenance and global modality masks cannot be deleted.

    Parameters:
        mdata: MuData to modify.
        target: Metadata mapping or table path.
        key: Entry or column to delete. Missing keys raise KeyError; `_log` and global modality masks are protected.

    Returns:
        The same MuData, modified in place.

    Examples:
        ```python
        import msmu as mm
        mdata.uns["temporary_note"] = "reviewed"
        mm.dt.drop(mdata, target="uns", key="temporary_note")
        ```
    """
    if key == "_log":
        raise ValueError("Cannot delete provenance with drop")
    if target in ("obsm", "varm") and key in mdata.mod:
        raise ValueError("Cannot delete a modality mask with drop")
    if target in ("uns", "obsm", "varm"):
        container = getattr(mdata, target)
    elif target.endswith((".uns", ".obsm", ".varm")):
        modality, attribute = target.rsplit(".", 1)
        container = getattr(mdata[modality], attribute)
    else:
        container = _column_table(mdata, target)
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

    Parameters:
        mdata: MuData to modify.
        column: Destination column name; an existing column is overwritten.
        values (Any): Scalar, positional array, or pandas Series. Series align by index. Values are captured in full for replay.
        modality: Modality containing the target table.
        on: `"var"` or `"obs"`.

    Returns:
        The same MuData, modified in place.

    Examples:
        ```python
        import msmu as mm
        mm.dt.assign(mdata, "reviewed", True, modality="protein")
        ```
    """
    if on not in ("obs", "var"):
        raise ValueError("on must be 'obs' or 'var'")
    getattr(mdata[modality], on)[column] = values
    return mdata


@log_provenance
def concat(mdatas: dict[str, md.MuData]) -> md.MuData:
    """Concatenate samples, retaining modalities in first-seen order.

    Features and observation annotations use an outer join. Feature metadata
    and uns use the first available value, without filling metadata nulls or
    concatenating lists. Provenance joins both input histories separately.
    At least two flat, sample-aligned (axis=0) MuData inputs are required.

    Parameters:
        mdatas: Dictionary mapping dataset names to at least two flat, sample-aligned (`axis=0`) MuData objects. Keys become the `dataset` observation annotation.

    Returns:
        A new MuData containing the union of modalities/features and concatenated samples. Input objects are not modified.

    Examples:
        ```python
        import msmu as mm
        combined = mm.dt.concat({"batch_a": first, "batch_b": second})
        ```
    """
    if len(mdatas) < 2:
        raise ValueError("At least two MuData objects are required.")
    modalities: dict[str, ad.AnnData] = {}
    for name, mdata in mdatas.items():
        if not isinstance(mdata, md.MuData):
            raise TypeError(f"Expected MuData object, got {type(mdata)} for {name}.")
        if mdata.axis != 0:
            raise ValueError("concat requires sample-aligned MuData objects (axis=0).")
        for mod, adata in mdata.mod.items():
            if not isinstance(adata, ad.AnnData):
                raise TypeError(f"Expected AnnData modality, got {type(adata)} for {name}/{mod}.")
            modalities.setdefault(mod, adata)
    if not modalities:
        raise ValueError("At least one modality is required.")

    # Lightweight containers own their masks; concat cannot remove input masks.
    # Empty modalities preserve the union despite md.concat taking an intersection.
    inputs = {}
    for name, mdata in mdatas.items():
        mods = {
            mod: cast(ad.AnnData, mdata.mod[mod])
            if mod in mdata.mod
            else ad.AnnData(
                X=template.X[:0, :0] if template.X is not None else None,
            )
            for mod, template in modalities.items()
        }
        inputs[name] = md.MuData(mods, **_container_annotations(mdata))

    # MuData 0.4.1 incorrectly annotates concat's input as AnnData.
    result = md.concat(
        cast(Any, inputs), join="outer", label="dataset", merge="first", uns_merge="first", pairwise=True
    )
    result = md.MuData(
        {mod: cast(ad.AnnData, result.mod[mod]) for mod in modalities}, **_container_annotations(result)
    )
    # First-value metadata may still reference the inputs; do not copy X/layers.
    for data in [result, *result.mod.values()]:
        data.uns = deepcopy({key: value for key, value in data.uns.items() if key != "_log"})
        for attr in ("varm", "varp"):
            mapping = getattr(data, attr)
            for key in mapping:
                mapping[key] = mapping[key].copy()
    return result


def _container_annotations(mdata: md.MuData) -> dict:
    """Copy containers, not matrices; modality masks are rebuilt by MuData."""
    return {
        "obs": mdata.obs.copy(),
        "var": mdata.var.copy(),
        "uns": {key: value for key, value in mdata.uns.items() if key != "_log"},
        **{
            attr: {
                key: value
                for key, value in getattr(mdata, attr).items()
                if attr not in ("obsm", "varm") or key not in mdata.mod
            }
            for attr in ("obsm", "varm", "obsp", "varp")
        },
    }
