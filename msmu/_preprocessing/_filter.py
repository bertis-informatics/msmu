from __future__ import annotations

import os
import warnings
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData

from .._utils._anndata import DECOY_COLUMN, _get_target_feature_mask
from .._utils._mudata import copy_mudata, get_anndata_mod, replace_modality
from .._core._provenance import log_provenance
from ..logging_utils import get_logger

logger = get_logger(__name__)

# A deprecation warning is attributed to the first frame outside msmu's own files, so it points at
# the caller's line however many frames the provenance decorator puts in between.
_MSMU_PACKAGE_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "")

# numpy dtype kinds of a string matrix: object (as an h5mu file returns strings), unicode and bytes.
_STRING_DTYPE_KINDS = frozenset("OUS")


def _warn_deprecated_filter_function(function_name: str) -> None:
    warnings.warn(
        f"{function_name} is deprecated; use msmu.pp.filter instead.",
        DeprecationWarning,
        skip_file_prefixes=(_MSMU_PACKAGE_DIRECTORY,),
    )


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
    Deprecated; use [`filter`][msmu.pp.filter], which evaluates the condition and filters in one call.

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

    Notes:
        Calling this function emits `DeprecationWarning`.
    """
    _warn_deprecated_filter_function("add_filter")

    if name is not None and (not isinstance(name, str) or not name.strip() or "/" in name):
        raise ValueError("name must be a nonblank string without slashes")

    mdata = copy_mudata(mdata)

    _require_key_for_matrix_source(on, key)

    filter_name = f"{column}_{keep}_{value}"
    if on in {"varm", "obsm"}:
        filter_name = f"{on}[{key!r}].{filter_name}"
    if name is not None:
        filter_name = name
    adata = get_anndata_mod(mdata, modality)

    column_values = _get_filter_source_column(adata, modality, column, on, key)
    store_axis = "varm" if _is_variable_filter(on) else "obsm"

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
    Deprecated; use [`filter`][msmu.pp.filter], which evaluates the condition and filters in one call.

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

    Notes:
        Calling this function emits `DeprecationWarning`.
    """
    _warn_deprecated_filter_function("apply_filter")

    if on not in {"all", "var", "obs"}:
        raise ValueError(f"Unknown filter axis: {on}")

    mdata = copy_mudata(mdata)

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
    _warn_if_every_decoy_removed(adata_to_filter, filtered_adata, modality)

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
    include_decoy: bool = True,
) -> MuData:
    """
    Filters the specified modality on one condition and returns the filtered `MuData`.

    The condition is applied at once: no mask is stored, and the provenance log is the record of what
    was filtered. Missing source values never pass, including for negated conditions.

    - `on="var"` / `on="varm"`: variables failing the condition are removed.
    - `on="obs"` / `on="obsm"`: observations failing the condition are removed.
    - `on="layers"`: `column` names a layer holding one value per cell, such as the per-run
      `q_value` of a wide `precursor` modality (observations = runs, variables = precursors). Cells
      failing the condition are emptied (set to NaN) in `.X` and in every layer, and variables left
      without a passing cell are removed; observations are kept. An integer or boolean layer cannot
      hold NaN and becomes floating-point; a layer of strings cannot hold an emptied cell and is
      refused.

    Decoy features (`.var["decoy"] == 1`) are filtered like targets: a decoy without a value fails.
    Removing decoys before the q-values are calculated makes those q-values too small, and removing
    every decoy leaves none to calculate them from (a warning says so). Filter on identification
    scores first and on quantification after, or pass `include_decoy=False` to keep every decoy.

    Parameters:
        mdata: `MuData` object to filter.
        modality: The modality within the `MuData` object to filter.
        on: Source of the values to filter on. One of `'var'`, `'obs'`, `'varm'`, `'obsm'`, or `'layers'`.
        column: The column in the selected table to filter on, or the layer name when `on` is `'layers'`.
        keep: The condition a value must satisfy to be kept. Layers support the numeric comparisons only.
        value: The value to compare against.
        key: Key to select the table from `.varm`/`.obsm` when `on` is `'varm'`/`'obsm'`.
        include_decoy: Whether the condition applies to decoy features. With `False`, every decoy is
            kept, whatever its value. Observation filters (`'obs'`, `'obsm'`) remove samples from
            targets and decoys alike and accept only `True`.

    Returns:
        `MuData` object with the modality filtered.
    """
    _require_key_for_matrix_source(on, key)
    if not include_decoy and on in {"obs", "obsm"}:
        raise ValueError(
            "include_decoy=False applies to variable and layer filters; an observation filter removes "
            "samples from targets and decoys alike."
        )

    mdata = copy_mudata(mdata)
    adata = get_anndata_mod(mdata, modality)

    if on == "layers":
        filtered_adata = _filter_cells_by_layer(adata, modality, column, keep, value, include_decoy)
    else:
        filtered_adata = _filter_axis_by_column(adata, modality, column, keep, value, on, key, include_decoy)
    is_condition_on_decoy_column = on == "var" and column == DECOY_COLUMN
    if not is_condition_on_decoy_column:  # removing the decoys on purpose needs no warning
        _warn_if_every_decoy_removed(adata, filtered_adata, modality)

    return replace_modality(mdata, modality, filtered_adata)


def _filter_axis_by_column(
    adata: AnnData, modality: str, column: str, keep, value, on: str, key: str | None, include_decoy: bool
) -> AnnData:
    """Remove the variables or observations failing a condition on a `.var`/`.obs`/`.varm`/`.obsm` column."""
    column_values = _get_filter_source_column(adata, modality, column, on, key)
    mask = _mask_boolean_filter(values_to_mask=column_values, keep=keep, value=value)

    if not _is_variable_filter(on):
        filtered_adata = adata[mask, :].copy()
        logger.info(
            "Filtered %s on %s '%s' %s %s: observations %d -> %d",
            modality, on, column, keep, value, adata.n_obs, filtered_adata.n_obs,
        )
        return filtered_adata

    if not include_decoy:
        mask = mask | ~_get_target_feature_mask(adata.var)
    filtered_adata = adata[:, mask].copy()
    logger.info(
        "Filtered %s on %s '%s' %s %s: variables %d -> %d (decoys %d -> %d)",
        modality, on, column, keep, value, adata.n_vars, filtered_adata.n_vars,
        _count_decoy_features(adata), _count_decoy_features(filtered_adata),
    )

    return filtered_adata


def _filter_cells_by_layer(
    adata: AnnData, modality: str, column: str, keep, value, include_decoy: bool
) -> AnnData:
    """Empty the cells failing a condition on one layer, then remove the variables left without a passing cell."""
    if keep in {"contains", "not_contains"}:
        raise ValueError(f"Layer filters compare numbers; '{keep}' is not supported when on is 'layers'.")
    if column not in adata.layers:
        raise ValueError(f"Layer '{column}' not found in {modality}.layers")

    # anndata lists `.X` among the layers under the key None; it is handled by name here.
    layer_names = [layer_name for layer_name in adata.layers.keys() if layer_name is not None]

    # The comparisons are elementwise, so the whole (observations x variables) frame is masked in one call.
    source_values = pd.DataFrame(adata.layers[column])
    # A copy: pandas 3 hands out a read-only view, and decoy cells may be marked as passing below.
    is_passing_cell = _mask_boolean_filter(values_to_mask=source_values, keep=keep, value=value).to_numpy(copy=True)
    if not include_decoy:
        is_passing_cell[:, ~_get_target_feature_mask(adata.var)] = True

    # A failing cell is an identification that is gone, so it is emptied wherever the cell has a value.
    adata.X = _empty_failing_cells(adata.X, is_passing_cell, f"{modality}.X")
    for layer_name in layer_names:
        adata.layers[layer_name] = _empty_failing_cells(
            adata.layers[layer_name], is_passing_cell, f"{modality}.layers['{layer_name}']"
        )
    filtered_adata = adata[:, is_passing_cell.any(axis=0)].copy()

    is_emptied_cell = source_values.notna().to_numpy() & ~is_passing_cell
    logger.info(
        "Filtered %s on layers['%s'] %s %s: emptied %d of %d cells, variables %d -> %d (decoys %d -> %d)",
        modality, column, keep, value,
        int(is_emptied_cell.sum()), int(source_values.notna().to_numpy().sum()),
        adata.n_vars, filtered_adata.n_vars, _count_decoy_features(adata), _count_decoy_features(filtered_adata),
    )

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


def _count_decoy_features(adata: AnnData) -> int:
    return int((~_get_target_feature_mask(adata.var)).sum())


def _warn_if_every_decoy_removed(adata: AnnData, filtered_adata: AnnData, modality: str) -> None:
    """Warn when a filter removes the last decoy, which leaves no decoy to calculate q-values from."""
    if _count_decoy_features(adata) > 0 and _count_decoy_features(filtered_adata) == 0:
        logger.warning(
            "The filter removed every decoy of %s, so no q-values can be calculated from them. Filter on "
            "identification scores before quantification, or pass include_decoy=False to keep the decoys.",
            modality,
        )
