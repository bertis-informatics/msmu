import mudata as md
import plotly.graph_objects as go

from .__pdata import PlotData
from .__ptypes import PlotScatter
from ._template import DEFAULT_TEMPLATE
from ._utils import _set_color

DEFAULT_COLUMN = "_obs_"


def plot_umap(
    mdata: md.MuData,
    modality: str = "psm",
    groupby: str = DEFAULT_COLUMN,
    colorby: str = None,
    template: str = DEFAULT_TEMPLATE,
    **kwargs,
) -> go.Figure:
    # Get required data
    umap_columns = _get_umap_cols(mdata, modality)

    # Set titles
    title_text = "UMAP"
    xaxis_title = umap_columns[0]
    yaxis_title = umap_columns[1]
    hovertemplate = "<b>%{meta}</b><br>" + xaxis_title + ": %{x}<br>" + yaxis_title + ": %{y}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, mods=[modality])
    plot = PlotScatter(
        data=data._prep_umap_data(modality, umap_columns),
        x=umap_columns[0],
        y=umap_columns[1],
        name=groupby,
        hovertemplate=hovertemplate,
    )
    fig = plot.figure(mode="markers")

    # Update axis
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
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
