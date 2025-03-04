import pandas as pd
import mudata as md
import plotly.graph_objects as go

from ._common import _draw_stacked_bar
from ._utils import _get_traces
from .._utils import get_modality_dict


DEFAULT_COLUMN = "obs"


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

    # Draw plot
    data = _prep_charge_data(mdata, mods, groupby, "charge")
    traces = _get_traces(data, groupby, "count", name="charge", meta="charge")
    fig: go.Figure = _draw_stacked_bar(
        traces=traces,
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_title_text="Charge",
    )

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    return fig


def _prep_charge_data(
    mdata: md.MuData,
    mods: list[str],
    groupby: str,
    name: str,
) -> pd.DataFrame:
    obs_df = mdata.obs.copy()
    obs_df[DEFAULT_COLUMN] = obs_df.index

    orig_df = pd.concat([mdata[mod].to_df() for mod in mods])
    orig_var_df = pd.concat([mdata[mod].var for mod in mods])

    merged_df = orig_df.notna().join(obs_df[groupby], how="left")
    merged_df = merged_df.groupby(groupby, observed=True).any()

    melt_df = merged_df.stack().reset_index()
    melt_df.columns = [groupby, "_var", "_exists"]

    prep_df = melt_df.merge(orig_var_df[[name]], left_on="_var", right_index=True)
    prep_df = prep_df[prep_df["_exists"] > 0]
    prep_df = prep_df.drop(["_var", "_exists"], axis=1)
    prep_df = prep_df.groupby(groupby, observed=True).value_counts().reset_index()

    return prep_df
