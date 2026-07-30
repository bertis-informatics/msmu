import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from mudata import MuData

from .._core._blockdiag import dense_block
from .._utils._mudata import get_anndata_mod


def split_tmt(
    mdata: MuData,
    map: dict[str, str] | pd.Series | pd.DataFrame,
) -> MuData:
    """
    Split TMT channels in a MuData object into separate modalities based on a mapping.

    Splitting relabels the ``C`` reporter channels into ``C x n_set`` distinct ``channel_set``
    samples so that each biological sample is unambiguous. Because a given PSM is measured in
    only one set, the resulting PSM matrix is block-diagonal: every feature carries values in
    just its set's channels and is missing (NaN) in every other set's channels. Stored densely
    this is ``O(n_set^2)`` -- the feature count and the sample count both scale with the number
    of sets -- which is what makes many-plex studies exhaust memory. Only the block diagonal
    (``O(n_set)``) is stored, in a SciPy sparse ``.X``; the obs axis and every value are the
    same as a dense split would produce.

    Note: the structurally-missing (cross-set) cells are not stored, so
    ``mdata.mod["psm"].to_df()`` returns 0 (SciPy sparse convention) rather than NaN for those
    cells. Inspect the sparse matrix with :func:`msmu._core._blockdiag.dense_block`, which
    restores absent cells as NaN.

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
    set_labels = psm_adata.var["filename"].str.rsplit(".", n=1).str[0].map(map)
    # Fail loud on a filename with no set mapping: an unmapped filename would otherwise be
    # silently scattered into a phantom ``nan`` set, so raise here so a bad/incomplete map
    # cannot pass unnoticed.
    if set_labels.isna().any():
        unmapped = psm_adata.var["filename"].str.rsplit(".", n=1).str[0][set_labels.isna()].unique()
        raise ValueError(f"split_tmt: no set mapping for filename(s): {list(unmapped)[:5]}")
    psm_adata.var["set"] = set_labels

    channels = list(psm_adata.obs_names)
    set_names = list(pd.unique(set_labels))  # first-occurrence order (matches the legacy .unique())
    n_channels = len(channels)
    new_obs_names = [f"{channel}_{set_name}" for set_name in set_names for channel in channels]

    new_x = _build_block_diagonal_sparse(psm_adata.X, set_labels, set_names, n_channels)

    new_adata = AnnData(
        X=new_x,
        obs=pd.DataFrame(index=pd.Index(new_obs_names)),
        var=psm_adata.var.copy(),
    )
    new_adata.uns = dict(psm_adata.uns)

    new_mdata = MuData({"psm": new_adata})
    new_mdata.var = mdata.var.copy()
    new_mdata.uns = dict(mdata.uns)

    return new_mdata


def _build_block_diagonal_sparse(source_x, set_labels, set_names, n_channels) -> sp.csc_matrix:
    """Scatter each PSM's channel values into its set's block, storing observed cells only.

    Builds ``(n_channels * n_set, n_psm)`` directly as COO (no dense block-diagonal is ever
    materialised); NaN/absent cells are simply not stored.
    """
    # dense_block (not toarray) restores a sparse input's structurally-absent cells as NaN. toarray
    # fills them with 0, and np.isfinite(0) is True, so every absent cell would be stored as an
    # observed zero -- the exact corruption this block-diagonal representation exists to prevent.
    source = dense_block(source_x).astype(source_x.dtype) if sp.issparse(source_x) else np.asarray(source_x)
    n_psm = source.shape[1]
    set_code = set_labels.map({name: i for i, name in enumerate(set_names)}).to_numpy()

    rows, cols, vals = [], [], []
    psm_index = np.arange(n_psm)
    for channel in range(n_channels):
        channel_values = source[channel, :]
        row_of_channel = set_code * n_channels + channel
        observed = np.isfinite(channel_values)
        rows.append(row_of_channel[observed])
        cols.append(psm_index[observed])
        vals.append(channel_values[observed])

    coo = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_channels * len(set_names), n_psm),
        dtype=np.asarray(source).dtype,
    )
    return coo.tocsc()


__all__ = ["split_tmt"]
