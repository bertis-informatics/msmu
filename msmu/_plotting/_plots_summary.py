"""Summary plot facades for identification and intersection visualizations."""

from typing import Any

import mudata as md
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .._core._access import get_mdata
from ._pdata import PlotData
from ._ptypes import PlotBar
from ._template import DEFAULT_TEMPLATE
from ._utils import PlotContext, finalize_figure


def _build_upset_figure(
    data: tuple[pd.DataFrame, pd.Series],
    **layout_kwargs: Any,
) -> go.Figure:
    """Build an Upset figure from combination and item counts."""
    combination_counts, item_counts = data

    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.2, 0.8],
        column_widths=[0.2, 0.8],
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0,
        horizontal_spacing=0,
    )

    fig.add_trace(
        go.Bar(
            x=combination_counts["combination"].tolist(),
            y=combination_counts["count"].tolist(),
            text=combination_counts["count"].tolist(),
            textposition="auto",
            texttemplate="%{text:,d}",
            name="combination",
            showlegend=False,
            hovertemplate="Sets: %{x}<br>Count: %{y:,d}<extra></extra>",
            marker=dict(color="#1f77b4"),
        ),
        row=1,
        col=2,
    )

    set_names = item_counts.index.tolist()
    for _, row in combination_counts.iterrows():
        combination = row["combination"]
        for position, set_name in enumerate(set_names):
            fig.add_trace(
                go.Scatter(
                    x=[str(combination)],
                    y=[set_name],
                    mode="markers",
                    marker=dict(
                        color="#444444" if str(combination)[position] == "1" else "white",
                        size=10,
                        line=dict(color="#111111", width=2),
                    ),
                    showlegend=False,
                    hovertemplate="Sample: %{y}<extra></extra>",
                ),
                row=2,
                col=2,
            )

    fig.add_trace(
        go.Bar(
            x=item_counts.values.tolist(),
            y=set_names,
            text=item_counts.values.tolist(),
            textposition="auto",
            texttemplate="%{text:,d}",
            orientation="h",
            showlegend=False,
            hovertemplate="Sample: %{y}<br>Count: %{x:,d}<extra></extra>",
            marker=dict(color="#1f77b4"),
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(autorange="reversed", tickformat=",d", row=2, col=1)
    fig.update_xaxes(ticklen=0, showticklabels=False, row=1, col=2)
    fig.update_xaxes(ticklen=0, showticklabels=False, row=2, col=2)

    fig.update_yaxes(autorange="reversed", showticklabels=False, ticklen=0, side="right", row=2, col=1)
    fig.update_yaxes(side="right", tickformat=",d", showticklabels=True, row=1, col=2)
    fig.update_yaxes(side="right", showticklabels=True, row=2, col=2)

    fig.update_layout(**layout_kwargs)
    return fig


def plot_id(
    mdata: md.MuData,
    modality: str,
    layer: str | None = None,
    groupby: str | None = None,
    colorby: str | None = None,
    template: str = DEFAULT_TEMPLATE,
    obs_column: str | None = None,
    **kwargs: str,
) -> go.Figure:
    """Plots identification counts per modality grouped by observations."""
    context = PlotContext.grouped(
        mdata,
        modality,
        groupby=groupby,
        obs_column=obs_column,
        colorby=colorby,
        layer=layer,
        template=template,
    )
    if context.groupby is None:
        raise ValueError("plot_id requires a grouping column.")

    title_text = f"Number of {context.modality_label}s"
    xaxis_title = f"{context.groupby.capitalize()}"
    yaxis_title = f"Number of {context.modality_label}s"
    hovertemplate = f"{xaxis_title}: %{{x}}<br>{yaxis_title}: %{{y:,d}}<extra></extra>"

    data = PlotData(context.mdata, context.modality, layer=context.layer, obs_column=context.obs_column)
    plot = PlotBar(
        data=data.prep_id_bar(context.groupby, obs_column=context.obs_column),
        x=context.groupby,
        y="_count",
        name=context.groupby,
        hovertemplate=hovertemplate,
        text="_count",
    )
    fig = plot.figure()
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis_tickformat=",d",
        showlegend=True,
        legend=dict(title_text=f"{context.groupby.capitalize()}"),
    )
    fig.update_traces(texttemplate="%{y:,d}")

    return finalize_figure(fig, context=context, layout_kwargs=kwargs, apply_color=True)


def plot_upset(
    mdata: md.MuData,
    modality: str = "protein",
    layer: str | None = None,
    subset: str | None = None,
    subset_column: str | None = None,
    groupby: str | None = None,
    obs_column: str | None = None,
    **kwargs: str,
) -> go.Figure:
    """Draws an Upset plot showing feature intersections across observation groups."""
    subset_context = PlotContext.obs_only(mdata, modality, obs_column=subset_column, layer=layer)

    if subset is not None:
        mdata = get_mdata(mdata[mdata.obs[subset_context.obs_column] == subset].copy())

    context = PlotContext.grouped(mdata, modality, groupby=groupby, obs_column=obs_column, layer=layer)
    if context.groupby is None:
        raise ValueError("plot_upset requires a grouping column.")

    title_text = f"Intersection of Proteins among {context.groupby.capitalize()}"

    data = PlotData(context.mdata, context.modality, layer=context.layer, obs_column=context.obs_column)
    fig = _build_upset_figure(data.prep_id_upset(context.groupby, context.obs_column))
    fig.update_layout(title_text=title_text)

    return finalize_figure(fig, context=context, layout_kwargs=kwargs)
