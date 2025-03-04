import pandas as pd
import mudata as md
import plotly.graph_objects as go

from ._common import _draw_scatter
from ._template import DEFAULT_TEMPLATE
from ._utils import _get_traces, _set_color


DEFAULT_COLUMN = "obs"


def plot_pca(
    mdata: md.MuData,
    modality: str = "peptide",
    groupby: str = DEFAULT_COLUMN,
    colorby: str = None,
    template: str = DEFAULT_TEMPLATE,
    pcs: tuple[int, int] | list[int] = (1, 2),
    **kwargs,
) -> go.Figure:
    # Set titles
    title_text = "PCA"
    pcs, pc_columns = _get_pc_cols(mdata, modality, pcs)
    xaxis_title = f"{pc_columns[0]} ({mdata[modality].uns['pca']['variance_ratio'][pcs[0] - 1] * 100:.2f}%)"
    yaxis_title = f"{pc_columns[1]} ({mdata[modality].uns['pca']['variance_ratio'][pcs[1] - 1] * 100:.2f}%)"

    # Draw plot
    data = _prep_pca_data(mdata, modality, pc_columns)
    traces: list[dict] = _get_traces(data, pc_columns[0], pc_columns[1], groupby)

    fig: go.Figure = _draw_scatter(
        traces=traces,
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )

    # Update axis
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    # Set color
    if (colorby is not None) & (groupby == DEFAULT_COLUMN):
        fig = _set_color(fig, mdata, modality, colorby, template)

    return fig


def _get_pc_cols(
    mdata: md.MuData,
    modality: str,
    pcs: tuple[int, int],
) -> list[str]:
    # Check pcs length
    if len(pcs) != 2:
        raise ValueError("Only 2 PCs are allowed")

    # Check if pcs are integers
    if not all(isinstance(pc, int) for pc in pcs):
        raise ValueError("PCs must be integers")

    # Sort pcs
    if pcs[0] == pcs[1]:
        pcs[1] += 1
    elif pcs[0] > pcs[1]:
        pcs = (pcs[1], pcs[0])

    # Check if PCs exist
    if "X_pca" not in mdata[modality].obsm:
        raise ValueError(f"No PCA found in {modality}")

    # Get PC columns
    pc_columns = [f"PC_{pc}" for pc in pcs]

    if pc_columns[0] not in mdata[modality].obsm["X_pca"].columns:
        raise ValueError(f"{pc_columns[0]} not found in {modality}")
    if pc_columns[1] not in mdata[modality].obsm["X_pca"].columns:
        raise ValueError(f"{pc_columns[1]} not found in {modality}")

    return pcs, pc_columns


def _prep_pca_data(
    mdata: md.MuData,
    modality: str,
    pc_columns: list[str],
) -> pd.DataFrame:
    obs = mdata.obs.copy()
    obs[DEFAULT_COLUMN] = obs.index

    # Prepare data
    orig_df = mdata[modality].obsm["X_pca"][pc_columns].reset_index(names="_obs")
    join_df = orig_df.join(obs, on="_obs", how="left")

    return join_df
