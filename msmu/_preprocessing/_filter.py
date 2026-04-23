from __future__ import annotations

from typing import Literal

import pandas as pd
from mudata import MuData

from .._core._provenance import uns_logger
from .._core._status import MuDataStatus
from ..logging_utils import get_logger

logger = get_logger(__name__)


@uns_logger
def add_filter(
    mdata: MuData,
    modality: str,
    column: str,
    keep: Literal["eq", "ne", "lt", "le", "gt", "ge", "contains", "not_contains"],
    value: str | float | None,
    on: Literal["var", "obs", "varm", "obsm"] = "var",
    key: str | None = None,
) -> MuData:
    """
    Adds a filter to the specified modality in the MuData object based on the given condition.

    Parameters:
        mdata: MuData object to which the filter will be added.
        modality: The modality within the MuData object to which the filter will be applied
        column: The column in the selected table to apply the filter on.
        keep: The condition to apply for filtering.
        value: The value to compare against for filtering.
        on: Target table to filter on. One of 'var', 'obs', 'varm', or 'obsm'.
        key: Key to select table from `.varm`/`.obsm` when `on` is 'varm'/'obsm'.

    Returns:
        MuData object with the added filter.
    """

    mdata = mdata.copy()
    mstatus = MuDataStatus(mdata)

    if on in {"varm", "obsm"} and key is None:
        raise ValueError("key must be provided when on is 'varm' or 'obsm'.")

    filter_name = f"{column}_{keep}_{value}"
    adata = mdata.mod[modality]

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

    if "filter" not in mdata.mod[modality].uns:
        mdata.mod[modality].uns["filter"] = [filter_name]
    else:
        mdata.mod[modality].uns["filter"] = list(set([*mdata.mod[modality].uns["filter"]] + [filter_name]))

    # add filter for decoy (only supported for variable-level filters)
    if store_axis == "varm" and mstatus.__getattribute__(modality).has_decoy:
        decoy_df = mdata.mod[modality].uns["decoy"]
        if on == "var":
            decoy_mask = _mask_boolean_filter(series_to_mask=decoy_df[column], keep=keep, value=value)
        else:
            decoy_mask = mask.reindex(decoy_df.index).fillna(False)

        if "decoy_filter" not in mdata.mod[modality].uns:
            mdata.mod[modality].uns["decoy_filter"] = decoy_mask.to_frame(name=filter_name)
        else:
            mdata.mod[modality].uns["decoy_filter"][filter_name] = decoy_mask

    return mdata


def _mask_boolean_filter(series_to_mask: pd.Series, keep, value):
    if keep == "eq":
        return series_to_mask == value
    elif keep == "ne":
        return series_to_mask != value
    elif keep == "lt":
        return series_to_mask < value
    elif keep == "le":
        return series_to_mask <= value
    elif keep == "gt":
        return series_to_mask > value
    elif keep == "ge":
        return series_to_mask >= value
    elif keep == "contains":
        return series_to_mask.str.contains(str(value))
    elif keep == "not_contains":
        return ~series_to_mask.str.contains(str(value))
    else:
        raise ValueError(f"Unknown filter operator: {keep}")


@uns_logger
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
            available filter columns for the selected axis are applied.

    Returns:
        MuData object with the filter applied.
    """
    mdata = mdata.copy()
    mstatus = MuDataStatus(mdata)

    adata_to_filter = mdata.mod[modality]
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
                logger.warning("Filter columns not found in %s: %s", modality, missing_filter_columns)
            if not selected_filter_columns:
                raise ValueError(f"No matching filter columns found in {modality}.")

    filtered_adata = adata_to_filter[obs_mask, var_mask].copy()
    mdata.mod[modality] = filtered_adata

    if mstatus.__getattribute__(modality).has_decoy and var_filter_columns:
        decoy_df = adata_to_filter.uns["decoy"]
        if "decoy_filter" not in adata_to_filter.uns:
            raise ValueError("No decoy filter found in the modality's uns.")
        decoy_filter = adata_to_filter.uns["decoy_filter"]
        decoy_use_columns = [col for col in var_filter_columns if col in decoy_filter.columns]
        if not decoy_use_columns:
            raise ValueError("No matching decoy filter columns found in the modality's uns.")

        decoy_filtered_df = decoy_df[decoy_filter[decoy_use_columns].all(axis=1)].copy()
        decoy_filter = decoy_filter.loc[decoy_filtered_df.index, decoy_use_columns]

        mdata.mod[modality].uns["decoy"] = decoy_filtered_df
        mdata.mod[modality].uns["decoy_filter"] = decoy_filter

    return mdata.copy()
