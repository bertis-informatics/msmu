import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData


def subset(
    mdata: MuData,
    modality: str,
    cond_var: str = None,
    cond_obs: str = None,
) -> MuData:
    """
    Subset MuData object based on condition.

    Args:
        mdata (MuData): MuData object to subset.
        modality (str): Modality to subset.
        cond_var (str): Condition to subset variables.
        cond_obs (str): Condition to subset observations.

    Returns:
        mdata (MuData): Subsetted MuData object.
    """
    # Check inputs
    if (cond_obs is None) & (cond_var is None):
        print("No condition provided. Returning updated data.")
        return mdata

    # Prepare data
    mdata = mdata.copy()
    mdata.obs_names_make_unique() if cond_obs is not None else None
    mdata.var_names_make_unique() if cond_var is not None else None
    adata: AnnData = mdata[modality]

    # Subset
    if cond_obs is not None:
        adata = adata[adata.obs[cond_obs].index]
    if cond_var is not None:
        adata = adata[:, adata.var[cond_var].index]

    # Update mdata
    mdata.mod[modality] = adata
    mdata.update()

    return mdata


def split_tmt(
    mdata: MuData,
    map: dict[str, str] | pd.Series | pd.DataFrame,
) -> MuData:
    """
    Split TMT channels in a MuData object into separate modalities based on a mapping.

    Parameters
    ----------
    mdata : MuData
        The MuData object containing TMT data.
    map : dict[str, str] | pd.Series | pd.DataFrame
        A mapping of filenames to set names. If a DataFrame is provided, it should have two columns: the first for filenames and the second for set names.

    Returns
    -------
    MuData
        The modified MuData object with TMT channels split into separate modalities.
    """
    if isinstance(map, pd.Series):
        map = map.to_dict()
    elif isinstance(map, pd.DataFrame):
        if len(map.columns) != 2:
            raise ValueError("DataFrame must have exactly two columns.")
        map = map.set_index(map.columns[0])[map.columns[1]].to_dict()
    elif not isinstance(map, dict):
        raise ValueError("Map must be a dictionary, pandas Series, or DataFrame.")

    mdata["feature"].var["set"] = mdata["feature"].var["filename"].str.rsplit(".", n=1).str[0].map(map)

    df = mdata["feature"].to_df().T.copy()
    set_dfs = {}

    for set in mdata["feature"].var["set"].unique():
        set_index = mdata["feature"].var.index[mdata["feature"].var["set"] == set]
        set_df = df.loc[set_index]
        set_df.columns = set_df.columns + f"_{set}"
        set_dfs[set] = set_df

    set_df = pd.concat(set_dfs.values(), axis=1)
    set_df = set_df.loc[mdata["feature"].var.index]

    new_adata = AnnData(
        X=set_df.T,
        obs=pd.DataFrame(index=set_df.T.index),
        var=pd.DataFrame(index=set_df.T.columns),
    )
    new_adata.var = mdata["feature"].var.copy()
    new_adata.uns = mdata["feature"].uns.copy()

    new_mdata = MuData({"feature": new_adata})
    new_mdata.var = mdata.var.copy()
    new_mdata.uns = mdata.uns.copy()

    return new_mdata
