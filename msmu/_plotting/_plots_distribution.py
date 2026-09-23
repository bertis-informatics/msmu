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
    """Plot intensity distributions for a modality.

    Parameters:
        mdata: MuData containing the selected modality and observation metadata.
        modality: Name of the modality to plot, such as `"protein"` or `"psm"`.
        layer: Quantification layer; `None` reads `.X`.
        groupby: Grouping column in observation metadata (or feature metadata where supported). `None` uses the resolved `obs_column`.
        colorby: Observation column used to assign sample colors when `groupby` resolves to `obs_column`; ignored for other groupings. `None` uses the template palette.
        ptype: `"hist"`/`"histogram"`, `"box"`/`"boxplot"`/`"simple_box"`/`"simplebox"`, or `"vln"`/`"violin"`.
        template: Registered Plotly template name, normally `"msmu"`.
        bins: Number of histogram bins; ignored by box and violin plots.
        obs_column: Sample identifier in `mdata.obs`. If omitted, uses `uns["plotting"]["default_obs_column"]`, then source name/source_name/sample/filename, then the observation index.
        kwargs: Additional Plotly layout options, for example `width=800` or `title_text="QC"`.

    Returns:
        Plotly `Figure`. Call `.show()` to display or `.write_html("plot.html")` to export.

    Notes:
        Axes assume log2 intensities. Apply [`log2_transform`][msmu.pp.log2_transform] beforehand when appropriate; this function does not log-transform values. Quantification and embeddings are unchanged. Resolving an existing sample-identifier column can convert that column in `mdata.obs` to categorical; pass a copy to preserve its dtype.

    Examples:
        ```python
        import msmu as mm
        fig = mm.pl.plot_intensity(mdata, modality="protein", ptype="box")
        fig.show()
        ```
    """
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
    """Plot the cumulative distribution of feature completeness.

    Parameters:
        mdata: MuData containing the selected modality and observation metadata.
        modality: Name of the modality to plot, such as `"protein"` or `"psm"`.
        layer: Quantification layer; `None` reads `.X`.
        obs_column: Sample identifier in `mdata.obs`. If omitted, uses `uns["plotting"]["default_obs_column"]`, then source name/source_name/sample/filename, then the observation index.
        kwargs: Additional Plotly layout options, for example `width=800` or `title_text="QC"`.

    Returns:
        Plotly `Figure`. Call `.show()` to display or `.write_html("plot.html")` to export.

    Notes:
        The x-axis is the percentage of samples with an observed value; the y-axis is the cumulative percentage of features at or below that completeness. Quantification and embeddings are unchanged. Resolving an existing sample-identifier column can convert that column in `mdata.obs` to categorical; pass a copy to preserve its dtype.

    Examples:
        ```python
        import msmu as mm
        fig = mm.pl.plot_missingness(mdata, modality="protein")
        fig.show()
        ```
    """
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
    """Plot a lower-triangular Pearson correlation heatmap of grouped medians.

    Parameters:
        mdata: MuData containing the selected modality and observation metadata.
        modality: Name of the modality to plot, such as `"protein"` or `"psm"`.
        groupby: Grouping column in observation metadata (or feature metadata where supported). `None` uses the resolved `obs_column`.
        obs_column: Sample identifier in `mdata.obs`. If omitted, uses `uns["plotting"]["default_obs_column"]`, then source name/source_name/sample/filename, then the observation index.
        kwargs: Additional Plotly layout options, for example `width=800` or `title_text="QC"`.

    Returns:
        Plotly `Figure`. Call `.show()` to display or `.write_html("plot.html")` to export.

    Notes:
        Computes correlations for the plot from `.X`; it does not consume results from [`corr`][msmu.tl.corr]. Quantification and embeddings are unchanged. Resolving an existing sample-identifier column can convert that column in `mdata.obs` to categorical; pass a copy to preserve its dtype.

    Examples:
        ```python
        import msmu as mm
        fig = mm.pl.plot_correlation(mdata, modality="protein")
        fig.show()
        ```
    """
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
    """Plot a feature annotation by group.

    Parameters:
        mdata: MuData containing the selected modality and observation metadata.
        modality: Name of the modality to plot, such as `"protein"` or `"psm"`.
        groupby: Grouping column in observation metadata (or feature metadata where supported). `None` uses the resolved `obs_column`.
        var_column: Required column in the selected modality `.var`, such as `"charge"` or `"purity"`.
        obs_column: Sample identifier in `mdata.obs`. If omitted, uses `uns["plotting"]["default_obs_column"]`, then source name/source_name/sample/filename, then the observation index.
        ptype: `"stack"` (aliases `"stackd"`, `"stacked_bar"`), `"box"`, `"simple_box"`/`"simplebox"`, `"vln"`/`"violin"`, or `"hist"`/`"histogram"`. `None` selects box for numeric columns with more than 20 distinct values, otherwise stacked bars.
        bins: Histogram bin count; ignored for other plot types.
        kwargs: Additional Plotly layout options, for example `width=800` or `title_text="QC"`.

    Returns:
        Plotly `Figure`. Call `.show()` to display or `.write_html("plot.html")` to export.

    Notes:
        The selected feature annotation must exist; numeric plot types require numeric values. Quantification and embeddings are unchanged. Resolving an existing sample-identifier column can convert that column in `mdata.obs` to categorical; pass a copy to preserve its dtype.

    Examples:
        ```python
        import msmu as mm
        fig = mm.pl.plot_var(mdata, modality="psm", var_column="charge")
        fig.show()
        ```
    """
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
