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
