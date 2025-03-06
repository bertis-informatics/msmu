import numpy as np
import pandas as pd
import mudata as md
import plotly.graph_objects as go

from .__pdata import PlotData
from .__ptypes import PlotScatter
from .._utils import get_modality_dict


DEFAULT_COLUMN = "_obs_"


def plot_missingness(
    mdata: md.MuData,
    level: str = None,
    modality: str = None,
    **kwargs,
) -> go.Figure:
    # Set mods
    mods = list(get_modality_dict(mdata, level, modality).keys())

    # Set titles
    mod_names = (level or modality).capitalize()
    title_text = f"Missingness Inspection of {mod_names}"
    xaxis_title = "Missing value (%)"
    yaxis_title = f"Number of {mod_names} (%)"
    hovertemplate = f"Missing value ≤ %{{x:.2f}}%<br>{yaxis_title} : %{{y:.2f}}% (%{{meta}})<extra></extra>"

    # Draw plot
    data = PlotData(mdata, mods=mods)
    plot = PlotScatter(
        data=data._prep_missingness_data(),
        x="missingness",
        y="ratio",
        name="name",
        meta="count",
        hovertemplate=hovertemplate,
    )
    fig = plot.figure(mode="lines+markers", line=dict(shape="hv"))

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis_range=[-2.5, 102.5],
        xaxis_tickvals=[0, 20, 40, 60, 80, 100],
        yaxis_range=[-2.5, 102.5],
        yaxis_tickvals=[0, 20, 40, 60, 80, 100],
    )

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    return fig


def _prep_missingness_data(
    mdata: md.MuData,
    mods: list[str],
) -> pd.DataFrame:
    obs = mdata.obs.copy()
    obs[DEFAULT_COLUMN] = obs.index
    n_sample = obs.shape[0]

    # Prepare data
    orig_df = pd.concat([mdata[mod].to_df() for mod in mods])
    sum_list = orig_df.isna().sum(axis=0)

    count_list = sum_list.value_counts().sort_index().cumsum()
    count_list[np.int64(0)] = np.int64(0)
    count_list[n_sample] = np.int64(orig_df.shape[1])
    count_list = count_list.sort_index()

    prep_df = pd.DataFrame(count_list).reset_index(names="missingness")
    prep_df["ratio"] = prep_df["count"] / np.max(prep_df["count"]) * 100
    prep_df["missingness"] = prep_df["missingness"] / n_sample * 100
    prep_df["name"] = "Missingness"

    return prep_df
