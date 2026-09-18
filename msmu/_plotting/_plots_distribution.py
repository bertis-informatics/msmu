"""Distribution-oriented plotting functions."""

import mudata as md
import pandas as pd
import plotly.graph_objects as go

from ._pdata import PlotData
from ._ptypes import (
    PlotBox,
    PlotHeatmap,
    PlotHistogram,
    PlotScatter,
    PlotSimpleBox,
    PlotStackedBar,
    PlotViolin,
)
from ._template import DEFAULT_TEMPLATE
from ._utils import PlotContext, finalize_figure


def plot_intensity(
    mdata: md.MuData,
    modality: str,
    layer: str | None = None,
    groupby: str | None = None,
    colorby: str | None = None,
    ptype: str = "hist",
    template: str = DEFAULT_TEMPLATE,
    bins: int = 30,
    obs_column: str | None = None,
    **kwargs: str,
) -> go.Figure:
    """Visualize intensity distributions for a modality."""
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
        raise ValueError("plot_intensity requires a grouping column.")

    title_text = f"{context.modality_label} Intensity Distribution"
    data = PlotData(
        context.mdata,
        context.modality,
        layer=context.layer,
        obs_column=context.obs_column,
    )

    if ptype in ["hist", "histogram"]:
        xaxis_title = "Intensity (log<sub>2</sub>)"
        yaxis_title = f"Number of {context.modality_label}s"
        bin_info = data._get_bin_info(data._get_data(), bins)
        hovertemplate = f"<b>%{{meta}}</b><br>{xaxis_title}: %{{x}} ± {round(bin_info['width'] / 2, 4)}<br>{yaxis_title}: %{{y:2,d}}<extra></extra>"
        plot = PlotHistogram(
            data=data.prep_intensity_hist(context.groupby, context.obs_column, bin_info),
            x="center",
            y="count",
            name="name",
            hovertemplate=hovertemplate,
        )
        fig = plot.figure()
    elif ptype in ["box", "boxplot", "simple_box", "simplebox"]:
        xaxis_title = f"{context.groupby.capitalize()}"
        yaxis_title = "Intensity (log<sub>2</sub>)"
        plot = PlotSimpleBox(data=data.prep_intensity_simple_box(context.groupby, context.obs_column))
        fig = plot.figure()
    elif ptype in ["vln", "violin"]:
        xaxis_title = f"{context.groupby.capitalize()}"
        yaxis_title = "Intensity (log<sub>2</sub>)"
        plot = PlotViolin(
            data=data.prep_intensity_bar(context.groupby, context.obs_column),
            x=context.groupby,
            y="_value",
            name=context.groupby,
        )
        fig = plot.figure(
            spanmode="hard",
            points="suspectedoutliers",
            marker=dict(line=dict(outlierwidth=0)),
            box=dict(visible=True),
            meanline=dict(visible=True),
        )
    else:
        raise ValueError(f"Unknown plot type: {ptype}, choose from 'hist', 'box', 'vln'")

    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis_tickformat=",d",
        showlegend=True,
        legend=dict(title_text=f"{context.groupby.capitalize()}"),
    )

    return finalize_figure(fig, context=context, layout_kwargs=kwargs, apply_color=True)


def plot_missingness(
    mdata: md.MuData,
    modality: str,
    layer: str | None = None,
    obs_column: str | None = None,
    **kwargs: str,
) -> go.Figure:
    """Plot cumulative completeness percentages for a modality."""
    context = PlotContext.obs_only(mdata, modality, obs_column=obs_column, layer=layer)
    data = PlotData(
        context.mdata,
        context.modality,
        layer=context.layer,
        obs_column=context.obs_column,
    )

    title_text = f"{context.modality_label} Level"
    xaxis_title = "Data Completeness (%)"
    yaxis_title = f"Cumulative proportion of {context.modality_label} (%)"
    hovertemplate = f"Data Completeness ≤ %{{x:.2f}}%<br>{yaxis_title} : %{{y:.2f}}% (%{{meta}})<extra></extra>"

    plot = PlotScatter(
        data=data.prep_missingness_step(context.obs_column),
        x="missingness",
        y="ratio",
        name="name",
        meta="count",
        hovertemplate=hovertemplate,
    )
    fig = plot.figure(mode="lines+markers", line=dict(shape="hv"))
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis_range=[-2.5, 102.5],
        xaxis_tickvals=[0, 20, 40, 60, 80, 100],
        yaxis_range=[-2.5, 102.5],
        yaxis_tickvals=[0, 20, 40, 60, 80, 100],
    )

    return finalize_figure(fig, context=context, layout_kwargs=kwargs)


def plot_correlation(
    mdata: md.MuData,
    modality: str = "protein",
    groupby: str | None = None,
    obs_column: str | None = None,
    **kwargs: str,
) -> go.Figure:
    """Plot a lower-triangular Pearson correlation heatmap of grouped medians."""
    context = PlotContext.grouped(mdata, modality, groupby=groupby, obs_column=obs_column)
    if context.groupby is None:
        raise ValueError("plot_correlation requires a grouping column.")

    data = PlotData(
        context.mdata,
        context.modality,
        layer=context.layer,
        obs_column=context.obs_column,
    )
    plot = PlotHeatmap(
        data=data.prep_intensity_correlation(context.groupby, context.obs_column),
        hovertemplate="<b>%{x} / %{y}</b><br>Pearson's <i>r</i> : %{z:.4f}<extra></extra>",
    )
    fig = plot.figure()
    fig.update_traces(dict(colorbar_title_text="Pearson's <i>r</i>"))
    fig.update_layout(title_text="Correlation Heatmap")

    return finalize_figure(fig, context=context, layout_kwargs=kwargs)


def plot_var(
    mdata: md.MuData,
    modality: str = "psm",
    groupby: str | None = None,
    var_column: str | None = None,
    obs_column: str | None = None,
    ptype: str | None = None,
    bins: int = 30,
    **kwargs: str,
) -> go.Figure:
    """Plot variable annotations using stacked bars, box/violin plots, or histograms."""
    if var_column is None:
        raise ValueError("var_column must be specified.")

    context = PlotContext.grouped(mdata, modality, groupby=groupby, obs_column=obs_column)
    if context.groupby is None:
        raise ValueError("plot_var requires a grouping column.")

    modality_label = context.modality_label
    column_label = var_column.replace("_", " ").capitalize()

    if pd.api.types.is_numeric_dtype(mdata.mod[modality].var[var_column]):
        if len(mdata.mod[modality].var[var_column].unique()) > 20:
            ptype = ptype or "box"
        else:
            ptype = ptype or "stack"
    else:
        ptype = ptype or "stack"

    title_text = f"Number of {modality_label}s by {column_label}"
    xaxis_title = f"{context.groupby.capitalize()}"
    yaxis_title = f"Number of {modality_label}s"
    hovertemplate = f"{column_label}: %{{meta}}<br>Number of {modality_label}s: %{{y:2,d}}<extra></extra>"

    data = PlotData(
        context.mdata,
        context.modality,
        layer=context.layer,
        obs_column=context.obs_column,
    )
    if ptype in ["stack", "stackd", "stacked_bar"]:
        plot_data = data.prep_var_bar(context.groupby, var_column, context.obs_column)
        plot = PlotStackedBar(
            data=plot_data,
            x=context.groupby,
            y="count",
            name=var_column,
            meta=var_column,
            hovertemplate=hovertemplate,
        )
        fig = plot.figure()
    elif ptype in ["box"]:
        plot_data = data.prep_var_box(context.groupby, var_column, obs_column=context.obs_column)
        plot = PlotBox(
            data=plot_data,
            x=context.groupby,
            y=var_column,
            name=context.groupby,
        )
        fig = plot.figure(boxpoints="suspectedoutliers")
    elif ptype in ["simple_box", "simplebox"]:
        plot_data = data.prep_var_simple_box(context.groupby, var_column, context.obs_column)
        plot = PlotSimpleBox(data=plot_data)
        fig = plot.figure()
    elif ptype in ["vln", "violin"]:
        plot_data = data.prep_var_box(context.groupby, var_column, context.obs_column)
        plot = PlotViolin(
            data=plot_data,
            x=context.groupby,
            y=var_column,
            name=context.groupby,
        )
        fig = plot.figure(
            spanmode="hard",
            points="suspectedoutliers",
            marker=dict(line=dict(outlierwidth=0)),
            box=dict(visible=True),
            meanline=dict(visible=True),
        )
    elif ptype in ["hist", "histogram"]:
        bin_info = data._get_bin_info(data._get_var()[var_column], bins)
        plot_data = data.prep_var_hist(context.groupby, var_column, context.obs_column, bin_info)
        hovertemplate = f"<b>%{{meta}}</b><br>{column_label}: %{{x}} ± {round(bin_info['width'] / 2, 4)}<br>Number of {modality_label}s: %{{y:2,d}}<extra></extra>"
        plot = PlotHistogram(
            data=plot_data,
            x="center",
            y="count",
            name="name",
            hovertemplate=hovertemplate,
        )
        fig = plot.figure()
    else:
        raise ValueError(f"Unknown plot type: {ptype}, choose from 'stack', 'box', 'simplebox', 'vln', 'hist'")

    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend=dict(title_text=column_label),
    )

    return finalize_figure(fig, context=context, layout_kwargs=kwargs)
