from typing import Literal
import pandas as pd
from mudata import MuData

from .._utils import subset, uns_logger
from .._read_write._mdata_status import MuDataStatus


@uns_logger
def add_filter(
    mdata: MuData,
    modality: str,
    column: str,
    keep: Literal["eq", "ne", "lt", "le", "gt", "ge", "contains", "not_contains"],
    value: str | float | None,
) -> MuData:
    """
    Adds a filter to the specified modality in the MuData object based on the given condition.

    Parameters:
        mdata (MuData): MuData object to which the filter will be added.
        modality (str): The modality within the MuData object to which the filter will be applied
        column (str): The column in the modality's var DataFrame to apply the filter on.
        keep (Literal["eq", "ne", "lt", "le", "gt",
                            "ge", "contains", "not_contains"]): The condition to apply for filtering.
        value (str | float | None): The value to compare against for filtering.

    Returns:
        MuData: MuData object with the added filter.
    """

    mdata = mdata.copy()
    mstatus = MuDataStatus(mdata)

    filter_name = f"{column}_{keep}_{value}"

    var_df = mdata[modality].var
    mask = _mask_boolean_filter(series_to_mask=var_df[column], keep=keep, value=value)

    if "filter" not in mdata[modality].varm_keys():
        mdata[modality].varm["filter"] = mask.to_frame(name=filter_name)
    else:
        mdata[modality].varm["filter"][filter_name] = mask

    if "filter" not in mdata[modality].uns_keys():
        mdata[modality].uns["filter"] = [filter_name]
    else:
        mdata[modality].uns["filter"].append(filter_name)

    # add filter for decoy
    if mstatus.__getattribute__(modality).has_decoy:
        decoy_df = mdata[modality].uns["decoy"]
        decoy_mask = _mask_boolean_filter(series_to_mask=decoy_df[column], keep=keep, value=value)

        if "decoy_filter" not in mdata[modality].uns_keys():
            mdata[modality].uns["decoy_filter"] = decoy_mask.to_frame(name=filter_name)
        else:
            mdata[modality].uns["decoy_filter"][filter_name] = decoy_mask

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
) -> MuData:
    """
    Applies the filter to the specified modality in the MuData object.

    Parameters:
        mdata (MuData): MuData object to which the filter will be applied.
        modality (str): The modality within the MuData object to which the filter will be applied.

    Returns:
        MuData: MuData object with the filter applied.
    """
    mdata = mdata.copy()
    mstatus = MuDataStatus(mdata)

    adata_to_filter = mdata[modality]
    if "filter" not in adata_to_filter.varm_keys():
        raise ValueError("No filter found in the modality's varm.")

    filtered_adata = adata_to_filter[:, adata_to_filter.varm["filter"].all(axis=1)].copy()
    mdata.mod[modality] = filtered_adata

    if mstatus.__getattribute__(modality).has_decoy:
        decoy_df = adata_to_filter.uns["decoy"]
        if "decoy_filter" not in adata_to_filter.uns_keys():
            raise ValueError("No decoy filter found in the modality's uns.")
        decoy_filter = adata_to_filter.uns["decoy_filter"]
        decoy_filtered_df = decoy_df[decoy_filter.all(axis=1)].copy()
        decoy_filter = decoy_filter.loc[decoy_filtered_df.index]

        mdata[modality].uns["decoy"] = decoy_filtered_df
        mdata[modality].uns["decoy_filter"] = decoy_filter

    return mdata.copy()

    # def plot_filter_metric():
    ...
