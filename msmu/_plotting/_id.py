import pandas as pd
import mudata as md
import plotly.graph_objects as go

from ._common import _draw_bar
from ._template import DEFAULT_TEMPLATE
from ._utils import _get_2d_traces, _set_color
from .._utils import get_modality_dict


def plot_id(
    mdata: md.MuData,
    level: str = None,
    modality: str = None,
    groupby: str = None,
    colorby: str = None,
    template: str = DEFAULT_TEMPLATE,
    **kwargs,
) -> go.Figure:
    # Prepare data
    mods = list(get_modality_dict(mdata, level=level, modality=modality).keys())
    data = _prep_id_data(mdata, groupby, mods=mods)

    # Get traceset
    traces = _get_2d_traces(data, x="idx", y="count")

    # Set titles
    title_text = "Number of PSMs"
    xaxis_title = "Samples"
    yaxis_title = "Number of PSMs"
    if groupby is not None:
        title_text = f"Average Number of PSMs by {groupby}"
        xaxis_title = groupby
        yaxis_title = "Average number of PSMs"

    # Draw plot
    fig: go.Figure = _draw_bar(
        traces=traces,
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    # Set color
    if (colorby is not None) & (groupby is None):
        fig = _set_color(fig, mdata, mods, colorby, template)

    return fig


def _prep_id_data(
    mdata: md.MuData,
    groupby: str = None,
    mods: list[str] = None,
) -> pd.DataFrame:
    # Prepare data
    data = pd.concat([mdata[mod].to_df() for mod in mods]).T.count()
    data = pd.DataFrame(data, columns=["count"])

    # Groupby
    if groupby is not None:
        obs = pd.concat([mdata[mod].obs for mod in mods])
        data = pd.concat([obs, data], axis=1).groupby(groupby, observed=True)["count"].mean()
        data = pd.DataFrame(data, columns=["count"])

    return data
