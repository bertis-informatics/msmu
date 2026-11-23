from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData

from .._utils._mudata import copy_mudata, get_anndata_mod, replace_modality
from .._core._provenance import log_provenance
from .._core._status import MuDataStatus
from ..logging_utils import get_logger

logger = get_logger(__name__)

# numpy dtype kinds of a string matrix: object (as an h5mu file returns strings), unicode and bytes.
_STRING_DTYPE_KINDS = frozenset("OUS")


@log_provenance
def add_filter(
    mdata: MuData,
    modality: str,
    column: str,
    keep: Literal["eq", "ne", "lt", "le", "gt", "ge", "contains", "not_contains"],
    value: str | float | None,
    on: Literal["var", "obs", "varm", "obsm"] = "var",
    key: str | None = None,
    *,
    name: str | None = None,
) -> MuData:
    """
    Adds a filter to the specified modality in the `MuData` object based on the given condition.

    Missing source values never pass, including for negated conditions. Matrix
    filters use names prefixed with ``varm[key].`` or ``obsm[key].`` (key repr)
    to distinguish their source from ordinary `.obs`/`.var` columns.

    Parameters:
        mdata: `MuData` object to which the filter will be added.
        modality: The modality within the `MuData` object to which the filter will be applied
        column: The column in the selected table to apply the filter on.
        keep: The condition to apply for filtering.
        value: The value to compare against for filtering.
        on: Target table to filter on. One of `'var'`, `'obs'`, `'varm'`, or `'obsm'`.
        key: Key to select table from `.varm`/`.obsm` when `on` is `'varm'`/`'obsm'`.
        name: Optional filter name for selection with [`apply_filter(columns=...)`][msmu.pp.apply_filter].
            Must be nonblank and contain no slash (HDF5 key restriction). An existing
            name can only be reused with the same recorded condition.

    Returns:
        `MuData` object with the added filter.
    """

    if name is not None and (not isinstance(name, str) or not name.strip() or "/" in name):
        raise ValueError("name must be a nonblank string without slashes")

    mdata = copy_mudata(mdata)
    mstatus = MuDataStatus(mdata)

    _require_key_for_matrix_source(on, key)

    filter_name = f"{column}_{keep}_{value}"
    if on in {"varm", "obsm"}:
        filter_name = f"{on}[{key!r}].{filter_name}"
    if name is not None:
        filter_name = name
    adata = get_anndata_mod(mdata, modality)

    column_values = _get_filter_source_column(adata, modality, column, on, key)
    store_axis = "varm" if _is_variable_filter(on) else "obsm"

    has_decoy = mstatus.__getattribute__(modality).has_decoy
    if on == "var" and has_decoy:
        _require_decoy_column(adata, modality, column, filter_kind="variable")

    condition = {"on": on, "key": key if on in {"varm", "obsm"} else None,
                 "column": column, "keep": keep, "value": value}
    definitions = adata.uns.get("filter_conditions", {})
    previous = definitions.get(filter_name)
    if previous is not None and previous != condition:
        raise ValueError(f"Filter name {filter_name!r} already identifies a different condition")
    if name is not None and previous is None and any(
        "filter" in mapping and filter_name in mapping["filter"].columns
        for mapping in (adata.varm, adata.obsm)
    ):
        raise ValueError(f"Filter name {filter_name!r} already exists without a recorded condition")

    mask = _mask_boolean_filter(values_to_mask=column_values, keep=keep, value=value)

    if store_axis == "varm":
        if "filter" not in adata.varm.keys():
            adata.varm["filter"] = mask.to_frame(name=filter_name)
        else:
            adata.varm["filter"][filter_name] = mask
    else:
        if "filter" not in adata.obsm.keys():
            adata.obsm["filter"] = mask.to_frame(name=filter_name)
        else:
            adata.obsm["filter"][filter_name] = mask

    if "filter" not in adata.uns:
        adata.uns["filter"] = [filter_name]
    else:
        adata.uns["filter"] = list(dict.fromkeys([*adata.uns["filter"], filter_name]))

    # add filter for decoy (only supported for variable-level filters)
    if store_axis == "varm" and has_decoy:
        decoy_mask = _mask_decoys(adata.uns["decoy"], mask, column, keep, value, on)

        if "decoy_filter" not in adata.uns:
            adata.uns["decoy_filter"] = decoy_mask.to_frame(name=filter_name)
        else:
            adata.uns["decoy_filter"][filter_name] = decoy_mask

    if name is not None:
        adata.uns.setdefault("filter_conditions", {})[filter_name] = condition

    return mdata


def _require_key_for_matrix_source(on: str, key: str | None) -> None:
    if on in {"varm", "obsm"} and key is None:
        raise ValueError("key must be provided when on is 'varm' or 'obsm'.")


def _is_variable_filter(on: str) -> bool:
    """Whether a filter source is aligned with the variables (`.var` or a `.varm` table)."""
    return on in {"var", "varm"}


def _get_filter_source_column(adata: AnnData, modality: str, column: str, on: str, key: str | None) -> pd.Series:
    """The column a `.var`/`.obs`/`.varm`/`.obsm` filter reads, indexed by the axis it filters."""
    if on == "var":
        source_df = adata.var
    elif on == "obs":
        source_df = adata.obs
    elif on == "varm":
        if key not in adata.varm:
            raise ValueError(f"Key {key} not found in {modality}.varm")
        source_df = pd.DataFrame(adata.varm[key], index=adata.var_names)
    elif on == "obsm":
        if key not in adata.obsm:
            raise ValueError(f"Key {key} not found in {modality}.obsm")
        source_df = pd.DataFrame(adata.obsm[key], index=adata.obs_names)
    else:
        raise ValueError(f"Unknown filter target: {on}")

    if column not in source_df.columns:
        raise ValueError(f"Column '{column}' not found in {modality}.{on}")

    column_values = source_df[column]
    if not isinstance(column_values, pd.Series):
        raise ValueError(f"Column '{column}' must identify a single column in {modality}.{on}")

    return column_values


def _require_decoy_column(adata: AnnData, modality: str, column: str, filter_kind: str) -> None:
    if column not in adata.uns["decoy"].columns:
        raise ValueError(
            f"Column '{column}' is missing from {modality}.uns['decoy']; {filter_kind} filters apply to "
            f"decoys too. Add it with mm.dt.assign(..., modality='{modality}', on='decoy') or "
            f"mm.dt.map(..., target='{modality}.decoy') before adding the filter."
        )


def _mask_decoys(decoy_df: pd.DataFrame, target_mask: pd.Series, column: str, keep, value, on: str) -> pd.Series:
    """Decoy rows passing a variable filter.

    A `.var` column is evaluated on the decoy table's own values. A `.varm` table has no decoy rows, so
    a decoy passes only when it shares its name with a passing target.
    """
    if on == "var":
        return _mask_boolean_filter(values_to_mask=decoy_df[column], keep=keep, value=value)
    return target_mask.reindex(decoy_df.index).fillna(False)


def _mask_boolean_filter(values_to_mask: pd.Series | pd.DataFrame, keep, value):
    # Missing source values never satisfy a condition, including negated conditions.
    values = values_to_mask
    if keep == "eq":
        mask = values == value
    elif keep == "ne":
        mask = values != value
    elif keep == "lt":
        mask = values < value
    elif keep == "le":
        mask = values <= value
    elif keep == "gt":
        mask = values > value
    elif keep == "ge":
        mask = values >= value
    elif keep == "contains":
        mask = values.str.contains(str(value), na=False)
    elif keep == "not_contains":
        mask = ~values.str.contains(str(value), na=False)
    else:
        raise ValueError(f"Unknown filter operator: {keep}")
    return mask.fillna(False) & values.notna()


@log_provenance
def apply_filter(
    mdata: MuData,
    modality: str,
    on: Literal["all", "var", "obs"] = "all",
    columns: list[str] | None = None,
) -> MuData:
    """
    Applies the filter to the specified modality in the `MuData` object.

    Parameters:
        mdata: `MuData` object to which the filter will be applied.
        modality: The modality within the `MuData` object to which the filter will be applied.
        on: Which axis to apply filters on. One of:
            - `"var"`: apply only variable filters from `.varm["filter"]`
            - `"obs"`: apply only observation filters from `.obsm["filter"]`
            - `"all"`: apply both variable and observation filters
        columns: Optional list of filter column names to apply. When omitted, all
            available filter columns for the selected axis are applied. Missing
            requested columns raise `ValueError` rather than applying a partial set.

    Returns:
        `MuData` object with the filter applied.
    """
    if on not in {"all", "var", "obs"}:
        raise ValueError(f"Unknown filter axis: {on}")

    mdata = copy_mudata(mdata)
    mstatus = MuDataStatus(mdata)

    adata_to_filter = get_anndata_mod(mdata, modality)
    apply_var = on in {"var", "all"}
    apply_obs = on in {"obs", "all"}
    var_mask = slice(None)
    obs_mask = slice(None)
    var_filter_columns: list[str] = []
    obs_filter_columns: list[str] = []
    var_filter_df = None
    obs_filter_df = None
    missing_filter_columns: list[str] = []

    if apply_var:
        if "filter" not in adata_to_filter.varm.keys():
            if on == "var":
                logger.warning("No filter found in %s.varm['filter'].", modality)
                raise ValueError("No filter found in the modality's varm.")
        else:
            var_filter_df = adata_to_filter.varm["filter"]
            available_var_columns = var_filter_df.columns.to_list()
            if columns is None:
                var_filter_columns = available_var_columns
            else:
                var_filter_columns = [col for col in columns if col in available_var_columns]
                missing_var_columns = [col for col in columns if col not in available_var_columns]
                if missing_var_columns and on == "var":
                    logger.warning(
                        "Var filter columns not found in %s.varm['filter']: %s",
                        modality,
                        missing_var_columns,
                    )
                missing_filter_columns.extend(missing_var_columns)
                if len(var_filter_columns) == 0:
                    if on == "var":
                        raise ValueError(f"No matching var filter columns found in {modality}.varm['filter'].")
            if var_filter_columns:
                logger.info("Applying var filters for %s: %s", modality, var_filter_columns)
                var_mask = var_filter_df[var_filter_columns].all(axis=1)

    if apply_obs:
        if "filter" not in adata_to_filter.obsm.keys():
            if on == "obs":
                logger.warning("No filter found in %s.obsm['filter'].", modality)
                raise ValueError("No filter found in the modality's obsm.")
        else:
            obs_filter_df = adata_to_filter.obsm["filter"]
            available_obs_columns = obs_filter_df.columns.to_list()
            if columns is None:
                obs_filter_columns = available_obs_columns
            else:
                obs_filter_columns = [col for col in columns if col in available_obs_columns]
                missing_obs_columns = [col for col in columns if col not in available_obs_columns]
                if missing_obs_columns and on == "obs":
                    logger.warning(
                        "Obs filter columns not found in %s.obsm['filter']: %s",
                        modality,
                        missing_obs_columns,
                    )
                missing_filter_columns.extend(missing_obs_columns)
                if len(obs_filter_columns) == 0:
                    if on == "obs":
                        raise ValueError(f"No matching obs filter columns found in {modality}.obsm['filter'].")
            if obs_filter_columns:
                logger.info("Applying obs filters for %s: %s", modality, obs_filter_columns)
                obs_mask = obs_filter_df[obs_filter_columns].all(axis=1)

    if on == "all":
        has_any_filter_table = var_filter_df is not None or obs_filter_df is not None
        selected_filter_columns = [*var_filter_columns, *obs_filter_columns]
        if columns is None and not has_any_filter_table:
            logger.warning(
                "No filters found in %s.varm['filter'] or %s.obsm['filter'].",
                modality,
                modality,
            )
        elif columns is not None:
            all_available_columns = set()
            if var_filter_df is not None:
                all_available_columns.update(var_filter_df.columns)
            if obs_filter_df is not None:
                all_available_columns.update(obs_filter_df.columns)
            missing_filter_columns = [col for col in columns if col not in all_available_columns]
            if missing_filter_columns:
                logger.warning(
                    "Filter columns not found in %s: %s",
                    modality,
                    missing_filter_columns,
                )
            if not selected_filter_columns:
                raise ValueError(f"No matching filter columns found in {modality}.")

    if missing_filter_columns:
        raise ValueError(f"Filter columns not found in {modality}: {missing_filter_columns}")

    filtered_adata = adata_to_filter[obs_mask, var_mask].copy()

    if mstatus.__getattribute__(modality).has_decoy and var_filter_columns:
        decoy_df = adata_to_filter.uns["decoy"]
        if "decoy_filter" not in adata_to_filter.uns:
            raise ValueError("No decoy filter found in the modality's uns.")
        decoy_filter = adata_to_filter.uns["decoy_filter"]
        missing_decoy_columns = [col for col in var_filter_columns if col not in decoy_filter.columns]
        if missing_decoy_columns:
            raise ValueError(f"Decoy filter columns not found: {missing_decoy_columns}")

        decoy_filtered_df = decoy_df[decoy_filter[var_filter_columns].all(axis=1)].copy()
        decoy_filter = decoy_filter.loc[decoy_filtered_df.index].copy()

        filtered_adata.uns["decoy"] = decoy_filtered_df
        filtered_adata.uns["decoy_filter"] = decoy_filter

    return replace_modality(mdata, modality, filtered_adata)


@log_provenance
def filter(
    mdata: MuData,
    modality: str,
    on: Literal["var", "obs", "varm", "obsm", "layers"],
    column: str,
    keep: Literal["eq", "ne", "lt", "le", "gt", "ge", "contains", "not_contains"],
    value: str | float | None,
    key: str | None = None,
) -> MuData:
    """
    Filters the specified modality on one condition and returns the filtered `MuData`.

    The condition is applied at once: no mask is stored, and the provenance log is the record of what
    was filtered. Missing source values never pass, including for negated conditions.

    - `on="var"` / `on="varm"`: variables failing the condition are removed. Decoys in
      `.uns["decoy"]` are filtered with them, on their own column of the same name (`"var"`) or by
      sharing a name with a passing variable (`"varm"`).
    - `on="obs"` / `on="obsm"`: observations failing the condition are removed.
    - `on="layers"`: `column` names a layer holding one value per cell, such as the per-run
      `q_value` of a wide `precursor` modality (observations = runs, variables = precursors). Cells
      failing the condition are emptied (set to NaN) in `.X` and in every layer, and variables left
      without a passing cell are removed; observations are kept. An integer or boolean layer cannot
      hold NaN and becomes floating-point; a layer of strings cannot hold an emptied cell and is
      refused. Decoys in `.uns["decoy"]` are filtered on their own column of the same name.

    Parameters:
        mdata: `MuData` object to filter.
        modality: The modality within the `MuData` object to filter.
        on: Source of the values to filter on. One of `'var'`, `'obs'`, `'varm'`, `'obsm'`, or `'layers'`.
        column: The column in the selected table to filter on, or the layer name when `on` is `'layers'`.
        keep: The condition a value must satisfy to be kept. Layers support the numeric comparisons only.
        value: The value to compare against.
        key: Key to select the table from `.varm`/`.obsm` when `on` is `'varm'`/`'obsm'`.

    Returns:
        `MuData` object with the modality filtered.
    """
    _require_key_for_matrix_source(on, key)

    mdata = copy_mudata(mdata)
    adata = get_anndata_mod(mdata, modality)

    if on == "layers":
        filtered_adata = _filter_cells_by_layer(adata, modality, column, keep, value)
    else:
        filtered_adata = _filter_axis_by_column(adata, modality, column, keep, value, on, key)

    return replace_modality(mdata, modality, filtered_adata)


def _filter_axis_by_column(
    adata: AnnData, modality: str, column: str, keep, value, on: str, key: str | None
) -> AnnData:
    """Remove the variables or observations failing a condition on a `.var`/`.obs`/`.varm`/`.obsm` column."""
    column_values = _get_filter_source_column(adata, modality, column, on, key)
    has_decoy = "decoy" in adata.uns
    if on == "var" and has_decoy:
        _require_decoy_column(adata, modality, column, filter_kind="variable")

    mask = _mask_boolean_filter(values_to_mask=column_values, keep=keep, value=value)

    if not _is_variable_filter(on):
        filtered_adata = adata[mask, :].copy()
        logger.info(
            "Filtered %s on %s '%s' %s %s: observations %d -> %d",
            modality, on, column, keep, value, adata.n_obs, filtered_adata.n_obs,
        )
        return filtered_adata

    filtered_adata = adata[:, mask].copy()
    logger.info(
        "Filtered %s on %s '%s' %s %s: variables %d -> %d",
        modality, on, column, keep, value, adata.n_vars, filtered_adata.n_vars,
    )
    if has_decoy:
        _keep_passing_decoys(filtered_adata, _mask_decoys(adata.uns["decoy"], mask, column, keep, value, on))

    return filtered_adata


def _filter_cells_by_layer(adata: AnnData, modality: str, column: str, keep, value) -> AnnData:
    """Empty the cells failing a condition on one layer, then remove the variables left without a passing cell."""
    if keep in {"contains", "not_contains"}:
        raise ValueError(f"Layer filters compare numbers; '{keep}' is not supported when on is 'layers'.")
    if column not in adata.layers:
        raise ValueError(f"Layer '{column}' not found in {modality}.layers")
    has_decoy = "decoy" in adata.uns
    if has_decoy:
        _require_decoy_column(adata, modality, column, filter_kind="layer")

    # anndata lists `.X` among the layers under the key None; it is handled by name here.
    layer_names = [layer_name for layer_name in adata.layers.keys() if layer_name is not None]

    # The comparisons are elementwise, so the whole (observations x variables) frame is masked in one call.
    source_values = pd.DataFrame(adata.layers[column])
    is_passing_cell = _mask_boolean_filter(values_to_mask=source_values, keep=keep, value=value).to_numpy()

    # A failing cell is an identification that is gone, so it is emptied wherever the cell has a value.
    adata.X = _empty_failing_cells(adata.X, is_passing_cell, f"{modality}.X")
    for layer_name in layer_names:
        adata.layers[layer_name] = _empty_failing_cells(
            adata.layers[layer_name], is_passing_cell, f"{modality}.layers['{layer_name}']"
        )
    filtered_adata = adata[:, is_passing_cell.any(axis=0)].copy()

    cell_count_with_value = int(source_values.notna().to_numpy().sum())
    logger.info(
        "Filtered %s on layers['%s'] %s %s: emptied %d of %d cells, variables %d -> %d",
        modality, column, keep, value,
        cell_count_with_value - int(is_passing_cell.sum()), cell_count_with_value,
        adata.n_vars, filtered_adata.n_vars,
    )
    if has_decoy:
        decoy_mask = _mask_boolean_filter(values_to_mask=adata.uns["decoy"][column], keep=keep, value=value)
        _keep_passing_decoys(filtered_adata, decoy_mask)

    return filtered_adata


def _empty_failing_cells(matrix: np.ndarray, is_passing_cell: np.ndarray, location: str) -> np.ndarray:
    """Set the failing cells of a matrix to NaN.

    An integer or boolean matrix cannot hold NaN and is returned as float64. Strings have no missing
    value that an h5mu file can store, so a matrix of strings cannot be emptied.
    """
    values = np.asarray(matrix)
    if values.dtype.kind in _STRING_DTYPE_KINDS:
        raise ValueError(
            f"{location} (dtype {values.dtype}) cannot hold an emptied cell: strings have no missing value "
            "that an h5mu file can store. A layer filter empties the failing cells in .X and in every layer."
        )
    emptied_dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.float64
    emptied_matrix = values.astype(emptied_dtype)
    emptied_matrix[~is_passing_cell] = np.nan
    return emptied_matrix


def _keep_passing_decoys(filtered_adata: AnnData, decoy_mask: pd.Series) -> None:
    """Keep the passing rows of the decoy table, and a pending `decoy_filter` aligned with them."""
    passing_decoy_df = filtered_adata.uns["decoy"][decoy_mask.astype(bool)].copy()
    filtered_adata.uns["decoy"] = passing_decoy_df
    if "decoy_filter" in filtered_adata.uns:
        filtered_adata.uns["decoy_filter"] = filtered_adata.uns["decoy_filter"].loc[passing_decoy_df.index].copy()
