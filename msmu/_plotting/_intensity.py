import pandas as pd
import mudata as md
import plotly.graph_objects as go

from ._common import _draw_histogram, _draw_density, _draw_box
from ._template import DEFAULT_TEMPLATE
from ._utils import _get_traces, _set_color


def plot_intensity(
    mdata: md.MuData,
    modality: str = "psm",
    groupby: str = None,
    colorby: str = None,
    plot: str = "hist",
    template: str = DEFAULT_TEMPLATE,
    binsize: float = 0.5,
    bandwidth: float = 0.5,
    **kwargs,
) -> go.Figure:
    # Prepare data
    data = _prep_intensity_data(mdata, modality, groupby)

    # Get traceset
    traces = _get_traces(data)

    # Set titles
    title_text = "Precursor intensity distribution"
    xaxis_title = "Intensity (log<sub>2</sub>)"
    yaxis_title = "Number of PSMs"

    # Draw plot
    if plot in ["hist", "histogram"]:
        fig: go.Figure = _draw_histogram(
            traces=traces,
            binsize=binsize,
            title_text=title_text,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
        )
    elif plot in ["density", "violin"]:
        fig: go.Figure = _draw_density(
            traces=traces,
            bandwidth=bandwidth,
            title_text=title_text,
            xaxis_title=xaxis_title,
        )
    elif plot == "box":
        fig: go.Figure = _draw_box(
            traces=traces,
            title_text=title_text,
            xaxis_title=xaxis_title,
        )
    else:
        raise ValueError(f"Unknown plot type: {plot}, choose from 'hist|histogram', 'density|violin', 'box'")

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    # Set color
    if (colorby is not None) & (groupby is None):
        fig = _set_color(fig, mdata, modality, colorby, template)

    return fig


def _prep_intensity_data(
    mdata: md.MuData,
    modality: str = "psm",
    groupby: str = None,
) -> pd.DataFrame:
    # Prepare data
    data = mdata[modality].to_df()
    if groupby is not None:
        data["_groupby"] = mdata[modality].obs[groupby]
        data = data.groupby("_groupby", observed=True).mean()

    return data
