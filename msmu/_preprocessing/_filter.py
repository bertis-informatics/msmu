from __future__ import annotations

from typing import Literal

import pandas as pd
from mudata import MuData

from .._utils._mudata import get_anndata_mod, get_mudata_mod_as_mutable
from .._core._provenance import log_provenance
from .._core._status import MuDataStatus
from ..logging_utils import get_logger

logger = get_logger(__name__)


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
    Adds a filter to the specified modality in the MuData object based on the given condition.

    Missing source values never pass, including for negated conditions. Matrix
    filters use names prefixed with ``varm[key].`` or ``obsm[key].`` (key repr)
    to distinguish their source from ordinary obs/var columns.

    Parameters:
        mdata: MuData object to which the filter will be added.
        modality: The modality within the MuData object to which the filter will be applied
        column: The column in the selected table to apply the filter on.
        keep: The condition to apply for filtering.
        value: The value to compare against for filtering.
        on: Target table to filter on. One of 'var', 'obs', 'varm', or 'obsm'.
        key: Key to select table from `.varm`/`.obsm` when `on` is 'varm'/'obsm'.
        name: Optional filter name for selection with [`apply_filter(columns=...)`][msmu.pp.apply_filter].
            Must be nonblank and contain no slash (HDF5 key restriction). An existing
            name can only be reused with the same recorded condition.

    Returns:
        MuData object with the added filter.
    """

    if name is not None and (not isinstance(name, str) or not name.strip() or "/" in name):
        raise ValueError("name must be a nonblank string without slashes")

    mdata = mdata.copy()
    mstatus = MuDataStatus(mdata)

    if on in {"varm", "obsm"} and key is None:
        raise ValueError("key must be provided when on is 'varm' or 'obsm'.")

    filter_name = f"{column}_{keep}_{value}"
    if on in {"varm", "obsm"}:
        filter_name = f"{on}[{key!r}].{filter_name}"
    if name is not None:
        filter_name = name
    adata = get_anndata_mod(mdata, modality)

    if on == "var":
        source_df = adata.var
        store_axis = "varm"
    elif on == "obs":
        source_df = adata.obs
        store_axis = "obsm"
    elif on == "varm":
        if key not in adata.varm:
            raise ValueError(f"Key {key} not found in {modality}.varm")
        source_df = pd.DataFrame(adata.varm[key], index=adata.var_names)
        store_axis = "varm"
    elif on == "obsm":
        if key not in adata.obsm:
            raise ValueError(f"Key {key} not found in {modality}.obsm")
        source_df = pd.DataFrame(adata.obsm[key], index=adata.obs_names)
        store_axis = "obsm"
    else:
        raise ValueError(f"Unknown filter target: {on}")

    if column not in source_df.columns:
        raise ValueError(f"Column '{column}' not found in {modality}.{on}")

    column_values = source_df[column]
    if not isinstance(column_values, pd.Series):
        raise ValueError(f"Column '{column}' must identify a single column in {modality}.{on}")

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

    mask = _mask_boolean_filter(series_to_mask=column_values, keep=keep, value=value)

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
    if store_axis == "varm" and mstatus.__getattribute__(modality).has_decoy:
        decoy_df = adata.uns["decoy"]
        if on == "var":
            decoy_mask = _mask_boolean_filter(series_to_mask=decoy_df[column], keep=keep, value=value)
        else:
            decoy_mask = mask.reindex(decoy_df.index).fillna(False)

        if "decoy_filter" not in adata.uns:
            adata.uns["decoy_filter"] = decoy_mask.to_frame(name=filter_name)
        else:
            adata.uns["decoy_filter"][filter_name] = decoy_mask

    if name is not None:
        adata.uns.setdefault("filter_conditions", {})[filter_name] = condition

    return mdata


def _mask_boolean_filter(series_to_mask: pd.Series, keep, value):
    # Missing source values never satisfy a condition, including negated conditions.
    values = series_to_mask
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
    Applies the filter to the specified modality in the MuData object.

    Parameters:
        mdata: MuData object to which the filter will be applied.
        modality: The modality within the MuData object to which the filter will be applied.
        on: Which axis to apply filters on. One of:
            - "var": apply only variable filters from `varm["filter"]`
            - "obs": apply only observation filters from `obsm["filter"]`
            - "all": apply both variable and observation filters
        columns: Optional list of filter column names to apply. When omitted, all
            available filter columns for the selected axis are applied. Missing
            requested columns raise ValueError rather than applying a partial set.

    Returns:
        MuData object with the filter applied.
    """
    if on not in {"all", "var", "obs"}:
        raise ValueError(f"Unknown filter axis: {on}")

    mdata = mdata.copy()
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

    get_mudata_mod_as_mutable(mdata)[modality] = filtered_adata

    return mdata.copy()
