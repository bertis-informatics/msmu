import pandas as pd
from anndata import AnnData
from mudata import MuData

from .._utils._mudata import get_anndata_mod


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

    psm_adata = get_anndata_mod(mdata, "psm")
    psm_adata.var["set"] = psm_adata.var["filename"].str.rsplit(".", n=1).str[0].map(map)

    df = psm_adata.to_df().T.copy()
    set_dfs = {}

    for set_name in psm_adata.var["set"].unique():
        set_index = psm_adata.var.index[psm_adata.var["set"] == set_name]
        set_df = df.loc[set_index]
        set_df.columns = set_df.columns + f"_{set_name}"
        set_dfs[set_name] = set_df

    set_df = pd.concat(set_dfs.values(), axis=1)
    set_df = set_df.loc[psm_adata.var.index]

    new_adata = AnnData(
        X=set_df.T,
        obs=pd.DataFrame(index=set_df.T.index),
        var=pd.DataFrame(index=set_df.T.columns),
    )
    new_adata.var = psm_adata.var.copy()
    new_adata.uns = dict(psm_adata.uns)

    new_mdata = MuData({"psm": new_adata})
    new_mdata.var = mdata.var.copy()
    new_mdata.uns = dict(mdata.uns)

    return new_mdata


__all__ = ["split_tmt"]
