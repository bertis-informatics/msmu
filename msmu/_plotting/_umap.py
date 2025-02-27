import pandas as pd
import mudata as md
import plotly.graph_objects as go

from ._common import _draw_scatter
from ._template import DEFAULT_TEMPLATE
from ._utils import _get_traces, _set_color

DEFAULT_COLUMN = "obs"


def plot_umap(
    mdata: md.MuData,
    modality: str = "psm",
    groupby: str = DEFAULT_COLUMN,
    colorby: str = None,
    template: str = DEFAULT_TEMPLATE,
    **kwargs,
) -> go.Figure:
    # Set titles
    title_text = "UMAP"
    xaxis_title = umap_columns[0]
    yaxis_title = umap_columns[1]

    # Draw plot
    umap_columns = _get_umap_cols(mdata, modality)
    data = _prep_umap_data(mdata, modality, umap_columns)
    traces: list[dict] = _get_traces(data, umap_columns[0], umap_columns[1], groupby)
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


def _get_umap_cols(
    mdata: md.MuData,
    modality: str,
) -> list[str]:
    # Check if UMAP exist
    if "X_umap" not in mdata[modality].obsm:
        raise ValueError(f"No UMAP found in {modality}")

    # Get UMAP columns
    umap_columns = [f"UMAP_{pc}" for pc in [1, 2]]

    if umap_columns[0] not in mdata[modality].obsm["X_umap"].columns:
        raise ValueError(f"{umap_columns[0]} not found in {modality}")
    if umap_columns[1] not in mdata[modality].obsm["X_umap"].columns:
        raise ValueError(f"{umap_columns[1]} not found in {modality}")

    return umap_columns


def _prep_umap_data(
    mdata: md.MuData,
    modality: str,
    umap_columns: list[str],
) -> pd.DataFrame:
    obs = mdata.obs.copy()
    obs[DEFAULT_COLUMN] = obs.index

    # Prepare data
    orig_df = mdata[modality].obsm["X_umap"][umap_columns].reset_index(names="_obs")
    join_df = orig_df.join(obs, on="_obs", how="left")

    return join_df
