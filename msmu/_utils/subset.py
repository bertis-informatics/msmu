import pandas as pd
from anndata import AnnData
from mudata import MuData


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

    mdata.mod["psm"].var["set"] = mdata.mod["psm"].var["filename"].str.rsplit(".", n=1).str[0].map(map)

    df = mdata.mod["psm"].to_df().T.copy()
    set_dfs = {}

    for set in mdata.mod["psm"].var["set"].unique():
        set_index = mdata.mod["psm"].var.index[mdata.mod["psm"].var["set"] == set]
        set_df = df.loc[set_index]
        set_df.columns = set_df.columns + f"_{set}"
        set_dfs[set] = set_df

    set_df = pd.concat(set_dfs.values(), axis=1)
    set_df = set_df.loc[mdata.mod["psm"].var.index]

    new_adata = AnnData(
        X=set_df.T,
        obs=pd.DataFrame(index=set_df.T.index),
        var=pd.DataFrame(index=set_df.T.columns),
    )
    new_adata.var = mdata.mod["psm"].var.copy()
    new_adata.uns = mdata.mod["psm"].uns.copy()

    new_mdata = MuData({"psm": new_adata})
    new_mdata.var = mdata.var.copy()
    new_mdata.uns = mdata.uns.copy()

    return new_mdata
