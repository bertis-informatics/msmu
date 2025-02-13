import pandas as pd
import mudata as md
import plotly.graph_objects as go

from ._common import _draw_bar
from ._template import DEFAULT_TEMPLATE
from ._utils import _get_2d_traces, _set_color


def plot_id(
    mdata: md.MuData,
    modality: str = "psm",
    groupby: str = None,
    colorby: str = None,
    template: str = DEFAULT_TEMPLATE,
    **kwargs,
) -> go.Figure:
    # Prepare data
    data = _prep_id_data(mdata, modality, groupby)

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
        fig = _set_color(fig, mdata, modality, colorby, template)

    return fig


def _prep_id_data(
    mdata: md.MuData,
    modality: str,
    groupby: str,
) -> pd.DataFrame:
    # Get data
    data = mdata[modality].to_df().T.count()
    data = pd.DataFrame(data, columns=["count"])

    # Groupby
    if groupby is not None:
        data = pd.concat([mdata[modality].obs, data], axis=1).groupby(groupby, observed=True).mean()

    return data
