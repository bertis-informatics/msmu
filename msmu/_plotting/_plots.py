"""
Module providing various plotting functions for MuData objects using Plotly.
"""

import mudata as md
import pandas as pd
import plotly.graph_objects as go

from ._pdata import PlotData
from ._ptypes import (
    PlotBar,
    PlotBox,
    PlotHistogram,
    PlotScatter,
    PlotSimpleBox,
    PlotStackedBar,
    PlotViolin,
    PlotUpset,
    PlotHeatmap,
)
from ._template import DEFAULT_TEMPLATE
from ._utils import (
    apply_color_if_needed,
    apply_layout_overrides,
    format_modality,
    get_pc_cols,
    get_umap_cols,
    resolve_obs_column,
    resolve_plot_columns,
)
from .._utils.get import get_mdata


def plot_id(
    mdata: md.MuData,
    modality: str,
    groupby: str | None = None,
    colorby: str | None = None,
    template: str = DEFAULT_TEMPLATE,
    obs_column: str | None = None,
    **kwargs: str,
) -> go.Figure:
    """
    Plots identification counts per modality grouped by observations.

    Parameters:
        mdata: MuData object containing the modality to visualize.
        modality: Target modality (psm, peptide, protein, or site).
        groupby: Observation column used to group bars.
        colorby: Observation column used for coloring (when applicable).
        template: Plotly template for colorway.
        obs_column: Observation column used for labeling/group resolution.
        **kwargs: Additional layout options forwarded to Plotly.

    Returns:
        Bar chart of identification counts per group.
    """
    groupby, obs_column = resolve_plot_columns(mdata, groupby, obs_column)

    # Set titles
    title_text = f"Number of {format_modality(mdata, modality)}s"
    xaxis_title = f"{groupby.capitalize()}"
    yaxis_title = f"Number of {format_modality(mdata, modality)}s"
    hovertemplate = f"{xaxis_title}: %{{x}}<br>{yaxis_title}: %{{y:,d}}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, modality, obs_column=obs_column)
    plot = PlotBar(
        data=data.prep_id_bar(groupby, obs_column=obs_column),
        x=groupby,
        y="_count",
        name=groupby,
        hovertemplate=hovertemplate,
        text="_count",
    )

    fig = plot.figure()

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis_tickformat=",d",
        showlegend=True,
        legend=dict(title_text=f"{groupby.capitalize()}"),
    )

    # Update traces
    fig.update_traces(texttemplate="%{y:,d}")

    # Update layout with kwargs
    fig = apply_layout_overrides(fig, kwargs)

    # Set color
    fig = apply_color_if_needed(
        fig,
        mdata=mdata,
        modality=modality,
        groupby=groupby,
        colorby=colorby,
        obs_column=obs_column,
        template=template,
    )

    return fig


def plot_intensity(
    mdata: md.MuData,
    modality: str,
    groupby: str | None = None,
    colorby: str | None = None,
    ptype: str = "hist",
    template: str = DEFAULT_TEMPLATE,
    bins: int = 30,
    obs_column: str | None = None,
    **kwargs: str,
) -> go.Figure:
    """
    Visualizes intensity distributions for a modality using histograms, box, or violin plots.

    Parameters:
        mdata: MuData object containing the modality to visualize.
        modality: Target modality (psm, peptide, protein, or site).
        groupby: Observation column used to group traces.
        colorby: Observation column used for coloring (when applicable).
        ptype: Plot type: 'hist', 'box', or 'vln'.
        template: Plotly template for colorway.
        bins: Number of bins for histogram view.
        obs_column: Observation column used for labeling/group resolution.
        **kwargs: Additional layout options forwarded to Plotly.

    Returns:
        Intensity distribution figure.
    """
    groupby, obs_column = resolve_plot_columns(mdata, groupby, obs_column)

    # Set titles
    title_text = f"{format_modality(mdata, modality)} Intensity Distribution"

    # Draw plot
    data = PlotData(mdata, modality, obs_column=obs_column)
    if ptype in ["hist", "histogram"]:
        xaxis_title = "Intensity (log<sub>2</sub>)"
        yaxis_title = f"Number of {format_modality(mdata, modality)}s"
        bin_info = data._get_bin_info(data._get_data(), bins)
        hovertemplate = f"<b>%{{meta}}</b><br>{xaxis_title}: %{{x}} ± {round(bin_info['width'] / 2, 4)}<br>{yaxis_title}: %{{y:2,d}}<extra></extra>"
        plot = PlotHistogram(
            data=data.prep_intensity_hist(groupby, obs_column, bin_info),
            x="center",
            y="count",
            name="name",
            hovertemplate=hovertemplate,
        )
        fig = plot.figure()
    elif ptype in ["box", "boxplot", "simple_box", "simplebox"]:
        xaxis_title = f"{groupby.capitalize()}"
        yaxis_title = "Intensity (log<sub>2</sub>)"

        plot = PlotSimpleBox(data=data.prep_intensity_simple_box(groupby, obs_column))
        fig = plot.figure()
    elif ptype in ["vln", "violin"]:
        xaxis_title = f"{groupby.capitalize()}"
        yaxis_title = "Intensity (log<sub>2</sub>)"

        plot = PlotViolin(
            data=data.prep_intensity_bar(groupby, obs_column),
            x=groupby,
            y="_value",
            name=groupby,
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

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis_tickformat=",d",
        showlegend=True,
        legend=dict(title_text=f"{groupby.capitalize()}"),
    )

    # Update layout with kwargs
    fig = apply_layout_overrides(fig, kwargs)

    # Set color
    fig = apply_color_if_needed(
        fig,
        mdata=mdata,
        modality=modality,
        groupby=groupby,
        colorby=colorby,
        obs_column=obs_column,
        template=template,
    )

    return fig


def plot_missingness(
    mdata: md.MuData,
    modality: str,
    obs_column: str | None = None,
    **kwargs: str,
) -> go.Figure:
    """
    Plots cumulative data completeness percentages for a modality.

    Parameters:
        mdata: MuData object containing the modality to visualize.
        modality: Target modality (psm, peptide, protein, or site).
        obs_column: Observation column used to order samples.
        **kwargs: Additional layout options forwarded to Plotly.

    Returns:
        Step line plot of cumulative completeness.
    """
    obs_column = resolve_obs_column(mdata, obs_column)

    # Set titles
    title_text = f"{format_modality(mdata, modality)} Level"
    xaxis_title = "Data Completeness (%)"
    yaxis_title = f"Cumulative proportion of {format_modality(mdata, modality)} (%)"
    hovertemplate = f"Data Completeness ≤ %{{x:.2f}}%<br>{yaxis_title} : %{{y:.2f}}% (%{{meta}})<extra></extra>"

    # Draw plot
    data = PlotData(mdata, modality, obs_column=obs_column)
    plot = PlotScatter(
        data=data.prep_missingness_step(obs_column),
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
    fig = apply_layout_overrides(fig, kwargs)

    return fig


def plot_pca(
    mdata: md.MuData,
    modality: str = "protein",
    groupby: str | None = None,
    colorby: str | None = None,
    template: str = DEFAULT_TEMPLATE,
    pcs: tuple[int, int] | list[int] = (1, 2),
    obs_column: str | None = None,
    **kwargs: str,
) -> go.Figure:
    """
    Plots PCA scores for a modality colored/grouped by observation metadata.

    Parameters:
        mdata: MuData object containing PCA results.
        modality: Target modality; defaults to 'protein'.
        groupby: Observation column used to group traces.
        colorby: Observation column used for coloring (when applicable).
        template: Plotly template for colorway.
        pcs: Pair of principal component indices (1-based).
        obs_column: Observation column used for labeling/group resolution.
        **kwargs: Additional layout options forwarded to Plotly.

    Returns:
        Scatter plot of PCA scores.
    """
    groupby, obs_column = resolve_plot_columns(mdata, groupby, obs_column)

    # Get data
    pcs, pc_columns = get_pc_cols(mdata, modality, pcs)
    variances = mdata[modality].uns["pca"]["variance_ratio"]

    # Set titles
    title_text = "PCA"
    xaxis_title = f"{pc_columns[0]} ({variances[pcs[0] - 1] * 100:.2f}%)"
    yaxis_title = f"{pc_columns[1]} ({variances[pcs[1] - 1] * 100:.2f}%)"
    hovertemplate = f"<b>%{{meta}}</b><br>{xaxis_title}: %{{x}}<br>{yaxis_title}: %{{y}}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, modality, obs_column=obs_column)
    plot = PlotScatter(
        data=data.prep_pca_scatter(modality, groupby, pc_columns, obs_column),
        x=pc_columns[0],
        y=pc_columns[1],
        name=groupby,
        meta=obs_column,
        hovertemplate=hovertemplate,
    )
    fig = plot.figure(mode="markers", marker=dict(size=10))

    # Update axis
    fig.update_yaxes(  # type: ignore
        scaleanchor="x",
        scaleratio=1,
    )

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend=dict(
            title=f"{groupby.capitalize()}",
            orientation="h",
            xanchor="right",
            yanchor="bottom",
            x=1,
            y=1,
        ),
    )

    # Update layout with kwargs
    fig = apply_layout_overrides(fig, kwargs)

    # Set color
    fig = apply_color_if_needed(
        fig,
        mdata=mdata,
        modality=modality,
        groupby=groupby,
        colorby=colorby,
        obs_column=obs_column,
        template=template,
    )

    return fig


def plot_umap(
    mdata: md.MuData,
    modality: str = "protein",
    groupby: str | None = None,
    colorby: str | None = None,
    template: str = DEFAULT_TEMPLATE,
    obs_column: str | None = None,
    **kwargs: str,
) -> go.Figure:
    """
    Plots UMAP embeddings for a modality colored/grouped by observations.

    Parameters:
        mdata: MuData object containing UMAP embeddings.
        modality: Target modality; defaults to 'protein'.
        groupby: Observation column used to group traces.
        colorby: Observation column used for coloring (when applicable).
        template: Plotly template for colorway.
        obs_column: Observation column used for labeling/group resolution.
        **kwargs: Additional layout options forwarded to Plotly.

    Returns:
        Scatter plot of UMAP embeddings.
    """
    groupby, obs_column = resolve_plot_columns(mdata, groupby, obs_column)

    # Get required data
    umap_columns = get_umap_cols(mdata, modality)

    # Set titles
    title_text = "UMAP"
    xaxis_title = umap_columns[0]
    yaxis_title = umap_columns[1]
    hovertemplate = f"<b>%{{meta}}</b><br>{xaxis_title}: %{{x}}<br>{yaxis_title}: %{{y}}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, modality, obs_column=obs_column)
    plot = PlotScatter(
        data=data.prep_umap_scatter(modality, groupby, umap_columns, obs_column),
        x=umap_columns[0],
        y=umap_columns[1],
        name=groupby,
        meta=obs_column,
        hovertemplate=hovertemplate,
    )
    fig = plot.figure(mode="markers", marker=dict(size=10))

    # Update axis
    fig.update_yaxes(  # type: ignore
        scaleanchor="x",
        scaleratio=1,
    )

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend=dict(
            title=f"{groupby.capitalize()}",
            orientation="h",
            xanchor="right",
            yanchor="bottom",
            x=1,
            y=1,
        ),
    )
    # Update layout with kwargs
    fig = apply_layout_overrides(fig, kwargs)

    # Set color
    fig = apply_color_if_needed(
        fig,
        mdata=mdata,
        modality=modality,
        groupby=groupby,
        colorby=colorby,
        obs_column=obs_column,
        template=template,
    )

    return fig


def plot_upset(
    mdata: md.MuData,
    modality: str = "protein",
    subset: str | None = None,
    subset_column: str | None = None,
    groupby: str | None = None,
    obs_column: str | None = None,
    **kwargs: str,
) -> go.Figure:
    """
    Draws an Upset plot showing protein intersections across observation groups.

    Parameters:
        mdata: MuData object containing the modality to visualize.
        modality: Target modality; defaults to 'protein'.
        subset: Specific observation value to subset on; optional.
        subset_column: Observation column used for subsetting.
        groupby: Observation column used to define sets.
        obs_column: Observation column used for labeling/group resolution.
        **kwargs: Additional layout options forwarded to Plotly.

    Returns:
        Upset diagram of protein intersections.
    """
    subset_column = resolve_obs_column(mdata, subset_column)

    if subset is not None:
        mdata = get_mdata(mdata[mdata.obs[subset_column] == subset].copy())

    groupby, obs_column = resolve_plot_columns(mdata, groupby, obs_column)

    title_text = f"Intersection of Proteins among {groupby.capitalize()}"

    # Draw plot
    data = PlotData(mdata, modality, obs_column=obs_column)
    plot = PlotUpset(
        data=data.prep_id_upset(groupby, obs_column),
    )
    fig = plot.figure()

    # Update layout
    fig.update_layout(
        title_text=title_text,
    )

    # Update layout with kwargs
    fig = apply_layout_overrides(fig, kwargs)

    return fig


def plot_correlation(
    mdata: md.MuData,
    modality: str = "protein",
    groupby: str | None = None,
    obs_column: str | None = None,
    **kwargs: str,
) -> go.Figure:
    """
    Plots a lower-triangular Pearson correlation heatmap of grouped medians.

    Parameters:
        mdata: MuData object containing expression data.
        modality: Target modality; defaults to 'protein'.
        groupby: Observation column used to group and average values.
        obs_column: Observation column used for labeling/group resolution.
        **kwargs: Additional layout options forwarded to Plotly.

    Returns:
        Heatmap of pairwise correlations.
    """
    groupby, obs_column = resolve_plot_columns(mdata, groupby, obs_column)

    # Set titles
    title_text = "Correlation Heatmap"

    # Draw plot
    data = PlotData(mdata, modality, obs_column=obs_column)
    plot = PlotHeatmap(
        data=data.prep_intensity_correlation(groupby, obs_column),
        hovertemplate="<b>%{x} / %{y}</b><br>Pearson's <i>r</i> : %{z:.4f}<extra></extra>",
    )
    fig = plot.figure()

    fig.update_traces(
        dict(
            colorbar_title_text="Pearson's <i>r</i>",
        )
    )

    # Update layout
    fig.update_layout(
        title_text=title_text,
    )

    # Update layout with kwargs
    fig = apply_layout_overrides(fig, kwargs)

    return fig


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
    """
    Plots variable annotations using stacked bars, box/violin plots, or histograms.

    Parameters:
        mdata: MuData object containing the modality to visualize.
        modality: Target modality; defaults to 'psm'.
        groupby: Observation column used to group traces.
        var_column: Variable column to visualize.
        obs_column: Observation column used for labeling/group resolution.
        ptype: Plot type inferred for numeric/categorical data when None.
        bins: Number of bins for histogram view.
        **kwargs: Additional layout options forwarded to Plotly.

    Returns:
        Plot of variable annotation distributions.
    """
    if var_column is None:
        raise ValueError("var_column must be specified.")

    groupby, obs_column = resolve_plot_columns(mdata, groupby, obs_column)

    # Set labels
    modality_label = format_modality(mdata, modality)
    column_label = var_column.replace("_", " ").capitalize()

    if pd.api.types.is_numeric_dtype(mdata[modality].var[var_column]):
        if len(mdata[modality].var[var_column].unique()) > 20:
            ptype = ptype or "box"
        else:
            ptype = ptype or "stack"
    else:
        ptype = ptype or "stack"

    # Set titles
    title_text = f"Number of {modality_label}s by {column_label}"
    xaxis_title = f"{groupby.capitalize()}"
    yaxis_title = f"Number of {modality_label}s"
    hovertemplate = f"{column_label}: %{{meta}}<br>Number of {modality_label}s: %{{y:2,d}}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, modality, obs_column=obs_column)
    if ptype in ["stack", "stackd", "stacked_bar"]:
        plot_data = data.prep_var_bar(groupby, var_column, obs_column)
        plot = PlotStackedBar(
            data=plot_data,
            x=groupby,
            y="count",
            name=var_column,
            meta=var_column,
            hovertemplate=hovertemplate,
        )
        fig = plot.figure()
    elif ptype in ["box"]:
        plot_data = data.prep_var_box(groupby, var_column, obs_column=obs_column)
        plot = PlotBox(
            data=plot_data,
            x=groupby,
            y=var_column,
            name=groupby,
        )
        fig = plot.figure(
            boxpoints="suspectedoutliers",
        )
    elif ptype in ["simple_box", "simplebox"]:
        plot_data = data.prep_var_simple_box(groupby, var_column, obs_column)
        plot = PlotSimpleBox(data=plot_data)
        fig = plot.figure()
    elif ptype in ["vln", "violin"]:
        plot_data = data.prep_var_box(groupby, var_column, obs_column)
        plot = PlotViolin(
            data=plot_data,
            x=groupby,
            y=var_column,
            name=groupby,
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
        plot_data = data.prep_var_hist(groupby, var_column, obs_column, bin_info)
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

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis_tickformat=",d",
        legend=dict(title_text=column_label),
    )

    # Update layout with kwargs
    fig = apply_layout_overrides(fig, kwargs)

    return fig
