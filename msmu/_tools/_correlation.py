"""
Module for correlation plots in MuData.
"""

import pandas as pd
from typing import Literal
from mudata import MuData

from .._utils import uns_logger
from .._utils.get import get_adata


@uns_logger
def corr(
    mdata: MuData,
    modality: str,
    layer: str | None = None,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
) -> pd.DataFrame:
    """
    Compute the correlation matrix for the specified modality in a MuData object.

    Parameters:
        mdata: MuData object containing the data.
        modality: The modality to compute correlations on.
        layer: Layer to use for quantification aggregation. If None, the default layer (.X) will be used. Defaults to None.
        method: Correlation method to use: "pearson", "spearman", or "kendall". Defaults to "pearson".

    Returns:
        DataFrame representing the correlation matrix.
    """
    mdata = mdata.copy()
    adata = get_adata(mdata, modality).copy()

    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in modality '{modality}'.")
        data = pd.DataFrame(adata.layers[layer], index=adata.obs_names, columns=adata.var_names)
    else:
        data = adata.to_df()

    corr_matrix = data.T.corr(method=method)

    mdata[modality].obsp["X_corr"] = corr_matrix.values

    return mdata
