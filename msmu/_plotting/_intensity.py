import pandas as pd
import mudata as md
import plotly.graph_objects as go

from ._common import _draw_histogram, _draw_box
from ._template import DEFAULT_TEMPLATE
from ._utils import _get_traces, _set_color, _get_bin_info
from .._utils import get_modality_dict

DEFAULT_COLUMN = "obs"


def plot_intensity(
    mdata: md.MuData,
    level: str = None,
    modality: str = None,
    groupby: str = DEFAULT_COLUMN,
    colorby: str = None,
    plot: str = "hist",
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
    if plot in ["hist", "histogram"]:
        data, bin_info = _prep_intensity_data_hist(mdata, groupby, mods, bins)
        traces = _get_traces(data, "center", "count", "name")
        fig: go.Figure = _draw_histogram(
            traces=traces,
            title_text=title_text,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            bin_info=bin_info,
        )
    elif plot == "box":
        data = _prep_intensity_data_box(mdata, groupby, mods)
        traces = _get_traces(data)
        fig: go.Figure = _draw_box(
            traces=traces,
            title_text=title_text,
            xaxis_title=xaxis_title,
        )
    else:
        raise ValueError(f"Unknown plot type: {plot}, choose from 'hist|histogram', 'box'")

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    # Set color
    if (colorby is not None) & (groupby == DEFAULT_COLUMN):
        fig = _set_color(fig, mdata, mods, colorby, template)

    return fig


def _prep_intensity_data_hist(
    mdata: md.MuData,
    groupby: str,
    mods: list[str],
    bins: int,
) -> pd.DataFrame:
    obs = mdata.obs.copy()
    obs[DEFAULT_COLUMN] = obs.index

    data = pd.concat([mdata[mod].to_df() for mod in mods]).T
    data = pd.melt(data, var_name="_obs", value_name="_value").dropna()
    data = data.join(obs, on="_obs", how="left")

    bin_info = _get_bin_info(data["_value"], bins)

    data["bin"] = pd.cut(data["_value"], bins=bin_info["edges"], labels=bin_info["labels"], include_lowest=True)

    grouped = data.groupby([groupby, "bin"], observed=False).size().unstack(fill_value=0)
    grouped = grouped[grouped.sum(axis=1) > 0]

    bin_counts = grouped.values.flatten()
    bin_freqs = bin_counts / data.shape[0]
    bin_names = grouped.index.get_level_values(0).repeat(bins).tolist()

    # make dataframe
    prepped = pd.DataFrame(
        {
            "center": bin_info["centers"] * len(grouped),
            "label": bin_info["labels"] * len(grouped),
            "count": bin_counts,
            "frequency": bin_freqs,
            "name": bin_names,
        }
    )

    return prepped, bin_info


def _prep_intensity_data_box(
    mdata: md.MuData,
    groupby: str,
    mods: list[str],
) -> pd.DataFrame:
    obs = mdata.obs.copy()
    obs[DEFAULT_COLUMN] = obs.index

    # Prepare data
    orig_df = pd.concat([mdata[mod].to_df() for mod in mods]).T
    melt_df = pd.melt(orig_df, var_name="_obs", value_name="_value").dropna()
    join_df = melt_df.join(obs, on="_obs", how="left")

    prep_df = join_df[[groupby, "_value"]].groupby(groupby, observed=True).describe().droplevel(level=0, axis=1)

    return prep_df
