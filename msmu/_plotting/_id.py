import mudata as md
import plotly.graph_objects as go

from .__pdata import PlotData
from .__ptypes import PlotBar
from ._template import DEFAULT_TEMPLATE
from ._utils import _set_color
from .._utils import get_modality_dict

DEFAULT_COLUMN = "_obs_"


def plot_id(
    mdata: md.MuData,
    level: str = None,
    modality: str = None,
    groupby: str = DEFAULT_COLUMN,
    colorby: str = None,
    template: str = DEFAULT_TEMPLATE,
    **kwargs,
) -> go.Figure:
    # Prepare data
    mods = list(get_modality_dict(mdata, level=level, modality=modality).keys())

    # Set titles
    title_text = "Number of PSMs"
    xaxis_title = "Samples"
    yaxis_title = "Number of PSMs"
    if groupby is not None:
        title_text = f"Average Number of PSMs by {groupby}"
        xaxis_title = groupby
        yaxis_title = "Average number of PSMs"
    hovertemplate = f"{xaxis_title}: %{{x}}<br>{yaxis_title}: %{{y:2,d}}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, mods=mods)
    plot = PlotBar(
        data=data._prep_id_data(groupby),
        x=groupby,
        y="_count",
        name=groupby,
        hovertemplate=hovertemplate,
        text="_count",
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

    # Set color
    if (colorby is not None) & (groupby == DEFAULT_COLUMN):
        fig = _set_color(fig, mdata, mods, colorby, template)

    return fig
