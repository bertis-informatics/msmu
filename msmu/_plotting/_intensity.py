import mudata as md
import plotly.graph_objects as go

from .__pdata import PlotData
from .__ptypes import PlotHistogram, PlotBox
from ._template import DEFAULT_TEMPLATE
from ._utils import _set_color
from .._utils import get_modality_dict

DEFAULT_COLUMN = "_obs_"


def plot_intensity(
    mdata: md.MuData,
    level: str = None,
    modality: str = None,
    groupby: str = DEFAULT_COLUMN,
    colorby: str = None,
    ptype: str = "hist",
    template: str = DEFAULT_TEMPLATE,
    bins: int = 50,
    **kwargs,
) -> go.Figure:
    # Set mods
    mods = list(get_modality_dict(mdata, level, modality).keys())

    # Set titles
    title_text = "Precursor intensity distribution"
    xaxis_title = "Intensity (log<sub>2</sub>)"
    yaxis_title = "Number of PSMs"

    # Draw plot
    if ptype in ["hist", "histogram"]:
        data = PlotData(mdata, mods=mods)
        bin_info = data._get_bin_info(data._get_data(), bins)
        hovertemplate = f"<b>%{{meta}}</b><br>{xaxis_title}: %{{x}} ± {round(bin_info['width'] / 2, 4)}<br>{yaxis_title}: %{{y:2,d}}<extra></extra>"
        plot = PlotHistogram(
            data=data._prep_intensity_data_hist(groupby, bins),
            x="center",
            y="count",
            name="name",
            hovertemplate=hovertemplate,
        )
        fig = plot.figure()
    elif ptype == "box":
        data = PlotData(mdata, mods=mods)
        plot = PlotBox(data=data._prep_intensity_data_box(groupby))
        fig = plot.figure()
    else:
        raise ValueError(f"Unknown plot type: {ptype}, choose from 'hist|histogram', 'box'")

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
