import mudata as md
import plotly.graph_objects as go

from .__pdata import PlotData
from .__ptypes import PlotStackedBar
from .._utils import get_modality_dict


DEFAULT_COLUMN = "_obs_"


def plot_charge(
    mdata: md.MuData,
    level: str = None,
    groupby: str = DEFAULT_COLUMN,
    modality: str = None,
    **kwargs,
) -> go.Figure:
    # Set mods
    mods = list(get_modality_dict(mdata, level=level, modality=modality).keys())

    # Set titles
    title_text = "Number of PSMs by charge state"
    xaxis_title = "Samples"
    yaxis_title = "Number of PSMs"
    hovertemplate = "Charge: %{meta}<br>Number of PSMs: %{y:2,d}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, mods=mods)
    plot = PlotStackedBar(
        data=data._prep_charge_data(groupby, "charge"),
        x=groupby,
        y="count",
        name="charge",
        meta="charge",
        hovertemplate=hovertemplate,
    )
    fig = plot.figure()

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

    return fig
