"""
Module for correlation plots in MuData.
"""

from typing import Literal
from mudata import MuData

from .._utils._mudata import get_anndata_mod, get_mudata_mod_as_mutable
from .._core._blockdiag import to_dense_df
from .._core._provenance import uns_logger


@uns_logger
def corr(
    mdata: MuData,
    modality: str,
    layer: str | None = None,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
) -> MuData:
    """
    Compute the correlation matrix for the specified modality in a MuData object.

    Parameters:
        mdata: MuData object containing the data.
        modality: The modality to compute correlations on.
        layer: Layer to use for quantification aggregation. If None, the default layer (.X) will be used. Defaults to None.
        method: Correlation method to use: "pearson", "spearman", or "kendall". Defaults to "pearson".

    Returns:
        The input MuData (copied) with the correlation matrix stored in the modality's
        ``obsp["X_corr"]``.
    """
    mdata = mdata.copy()
    adata = get_anndata_mod(mdata, modality).copy()

    if layer is not None and layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in modality '{modality}'.")
    # to_dense_df restores absent cells as NaN for a sparse .X/layer (a plain DataFrame over
    # the raw sparse matrix crashes, and a densify would poison absent cells with 0).
    data = to_dense_df(adata, layer=layer)

    corr_matrix = data.T.corr(method=method)

    adata.obsp["X_corr"] = corr_matrix.values
    # Write the modified copy back into the returned mdata: previously obsp was set on a detached
    # .copy() while the untouched mdata was returned, so every corr() call silently discarded it.
    get_mudata_mod_as_mutable(mdata)[modality] = adata

    return mdata
