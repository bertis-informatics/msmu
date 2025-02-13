import pandas as pd
import mudata as md
import plotly.graph_objects as go

from ._common import _draw_scatter
from ._template import DEFAULT_TEMPLATE
from ._utils import _get_2d_traces, _set_color


def plot_umap(
    mdata: md.MuData,
    modality: str = "psm",
    groupby: str = None,
    colorby: str = None,
    template: str = DEFAULT_TEMPLATE,
    **kwargs,
) -> go.Figure:
    # Prepare data
    umap_columns = _get_umap_cols(mdata, modality)
    data = _prep_umap_data(mdata, modality, groupby, umap_columns)

    # Get traceset
    traces: list[dict] = _get_2d_traces(
        data=data,
        x=umap_columns[0],
        y=umap_columns[1],
        name=groupby,
    )

    # Set titles
    title_text = "UMAP"
    xaxis_title = umap_columns[0]
    yaxis_title = umap_columns[1]

    # Draw plot
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
    if (colorby is not None) & (groupby is None):
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
    groupby: str,
    umap_columns: list[str],
) -> pd.DataFrame:
    # Get data
    data = mdata[modality].obsm["X_umap"][umap_columns]

    # Groupby
    if groupby is not None:
        data = data.join(mdata[modality].obs[groupby])

    return data
