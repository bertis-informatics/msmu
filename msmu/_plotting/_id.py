import pandas as pd
import mudata as md
import plotly.graph_objects as go

from ._common import _draw_bar
from ._template import DEFAULT_TEMPLATE
from ._utils import _get_traces, _set_color
from .._utils import get_modality_dict

DEFAULT_COLUMN = "obs"


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
    data = _prep_id_data(mdata, groupby, mods=mods)

    # Get traceset
    traces = _get_traces(data, groupby, "_count", groupby)

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
    if (colorby is not None) & (groupby == DEFAULT_COLUMN):
        fig = _set_color(fig, mdata, mods, colorby, template)

    return fig


def _prep_id_data(
    mdata: md.MuData,
    groupby: str,
    mods: list[str],
) -> pd.DataFrame:
    obs = mdata.obs.copy()
    obs[DEFAULT_COLUMN] = obs.index

    # Prepare data
    orig_df = pd.DataFrame(pd.concat([mdata[mod].to_df() for mod in mods]).T.count(), columns=["count"]).T
    melt_df = pd.melt(orig_df, var_name="_obs", value_name="_count").dropna()
    melt_df = melt_df.join(obs, on="_obs", how="left")

    prep_df = pd.DataFrame(melt_df.groupby(groupby, observed=True)["_count"].mean(), columns=["_count"]).T
    prep_df = prep_df.melt(var_name=groupby, value_name="_count").dropna()

    return prep_df
