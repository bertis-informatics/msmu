import mudata as md
import plotly.graph_objects as go
import numpy as np

from .__pdata import PlotData
from .__ptypes import *
from ._template import DEFAULT_TEMPLATE
from ._utils import _DEFAULT_OBS_PRIORITY, _get_pc_cols, _get_umap_cols, _set_color, resolve_obs_column


def _resolve_plot_columns(
    mdata: md.MuData,
    groupby: str | None,
    obs_column: str | None,
) -> tuple[str, str]:
    resolved_obs = resolve_obs_column(mdata, obs_column)
    resolved_groupby = groupby or resolved_obs

    return resolved_groupby, resolved_obs


def _apply_layout_overrides(fig: go.Figure, layout_kwargs: dict) -> go.Figure:
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig


def _apply_color_if_needed(
    fig: go.Figure,
    *,
    mdata: md.MuData,
    modality: str,
    groupby: str,
    colorby: str | None,
    obs_column: str,
    template: str,
) -> go.Figure:
    if (colorby is not None) and (groupby == obs_column):
        return _set_color(fig, mdata, modality, colorby, obs_column, template)
    elif (colorby is not None) and (groupby != obs_column):
        print("[Warning] 'colorby' is only applicable when 'groupby' is not set. Ignoring 'colorby' parameter.")
    return fig


def format_modality(mdata: md.MuData, modality: str) -> str:
    if modality == "feature":
        if mdata["feature"].uns["search_engine"] == "Diann":
            return "Precursor"
        else:
            return "PSM"
    elif modality == "peptide":
        return "Peptide"
    elif modality == "protein":
        return "Protein"
    elif modality.endswith("_site"):
        return modality.capitalize()
    else:
        raise ValueError(f"Unknown modality: {modality}, choose from 'feature', 'peptide', 'protein', '[ptm]_site'")


def plot_charge(
    mdata: md.MuData,
    modality: str = "feature",
    groupby: str | None = None,
    obs_column: str | None = None,
    **kwargs,
) -> go.Figure:
    groupby, obs_column = _resolve_plot_columns(mdata, groupby, obs_column)

    # Set titles
    title_text = "Number of PSMs by Charge State"
    xaxis_title = f"{groupby.capitalize()}"
    yaxis_title = "Number of PSMs"
    hovertemplate = "Charge: %{meta}<br>Number of PSMs: %{y:2,d}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, modality=modality, obs_column=obs_column)
    plot = PlotStackedBar(
        data=data._prep_var_data(groupby, "charge", obs_column=obs_column),
        x=groupby,
        y="count",
        name="charge",
        meta="charge",
        hovertemplate=hovertemplate,
    )
    fig = plot.figure()

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis_tickformat=",d",
        legend=dict(title_text="Charge State"),
    )

    # Update layout with kwargs
    fig = _apply_layout_overrides(fig, kwargs)

    return fig


def plot_id(
    mdata: md.MuData,
    modality: str,
    groupby: str | None = None,
    colorby: str | None = None,
    template: str = DEFAULT_TEMPLATE,
    obs_column: str | None = None,
    **kwargs,
) -> go.Figure:
    groupby, obs_column = _resolve_plot_columns(mdata, groupby, obs_column)

    # Set titles
    title_text = f"Number of {format_modality(mdata, modality)}s"
    xaxis_title = f"{groupby.capitalize()}"
    yaxis_title = f"Number of {format_modality(mdata, modality)}s"
    hovertemplate = f"{xaxis_title}: %{{x}}<br>{yaxis_title}: %{{y:,d}}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, modality=modality, obs_column=obs_column)
    if groupby != "fraction":
        plot = PlotBar(
            data=data._prep_id_data(groupby, obs_column=obs_column),
            x=groupby,
            y="_count",
            name=groupby,
            hovertemplate=hovertemplate,
            text="_count",
        )
    else:
        plot_data = data._prep_var_data(groupby, "id_count", obs_column=obs_column)
        plot_data = plot_data.loc[(plot_data["fraction"] == plot_data["id_count"]),]
        plot = PlotBar(
            data=plot_data,
            x=groupby,
            y="count",
            name=groupby,
            hovertemplate=hovertemplate,
            text="count",
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
    fig = _apply_layout_overrides(fig, kwargs)

    # Set color
    fig = _apply_color_if_needed(
        fig,
        mdata=mdata,
        modality=modality,
        groupby=groupby,
        colorby=colorby,
        obs_column=obs_column,
        template=template,
    )

    return fig


def plot_id_fraction(
    mdata: md.MuData,
    modality: str,
    groupby: str = "filename",
    **kwargs,
) -> go.Figure:
    # Set titles
    title_text = f"Number of {format_modality(mdata, modality)}s by Fraction"
    # xaxis_title = f"Filenames"
    yaxis_title = f"Number of {format_modality(mdata, modality)}s"
    hovertemplate = f"<b>%{{x}}</b><br>{yaxis_title}: %{{y:2,d}}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, modality=modality)
    plot = PlotBar(
        data=data._prep_id_fraction_data(groupby),
        x=groupby,
        y="count",
        name="filename",
        hovertemplate=hovertemplate,
        text="count",
    )
    fig = plot.figure(
        texttemplate="%{text:,.0f}",
        textfont=dict(size=10),
    )

    # Update layout
    fig.update_layout(
        title_text=title_text,
        # xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis_tickformat=",d",
        showlegend=False,
        xaxis_showticklabels=False,
    )

    # Update layout with kwargs
    fig = _apply_layout_overrides(fig, kwargs)

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
    **kwargs,
) -> go.Figure:
    groupby, obs_column = _resolve_plot_columns(mdata, groupby, obs_column)

    # Set titles
    title_text = f"{format_modality(mdata, modality)} Intensity Distribution"

    # Draw plot
    data = PlotData(mdata, modality=modality, obs_column=obs_column)
    if ptype in ["hist", "histogram"]:
        xaxis_title = "Intensity (log<sub>2</sub>)"
        yaxis_title = f"Number of {format_modality(mdata, modality)}s"
        bin_info = data._get_bin_info(data._get_data(), bins)
        hovertemplate = f"<b>%{{meta}}</b><br>{xaxis_title}: %{{x}} ± {round(bin_info['width'] / 2, 4)}<br>{yaxis_title}: %{{y:2,d}}<extra></extra>"
        plot = PlotHistogram(
            data=data._prep_intensity_data_hist(groupby, obs_column=obs_column, bin_info=bin_info),
            x="center",
            y="count",
            name="name",
            hovertemplate=hovertemplate,
        )
        fig = plot.figure()

    elif ptype == "box":
        xaxis_title = f"{groupby.capitalize()}"
        yaxis_title = "Intensity (log<sub>2</sub>)"

        plot = PlotSimpleBox(data=data._prep_intensity_data_box(groupby, obs_column=obs_column))
        fig = plot.figure()
    elif ptype in ["vln", "violin"]:
        xaxis_title = f"{groupby.capitalize()}"
        yaxis_title = "Intensity (log<sub>2</sub>)"

        plot = PlotViolin(
            data=data._prep_intensity_data(groupby, obs_column=obs_column),
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
        raise ValueError(f"Unknown plot type: {ptype}, choose from 'hist|histogram', 'box'")

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
    fig = _apply_layout_overrides(fig, kwargs)

    # Set color
    fig = _apply_color_if_needed(
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
    **kwargs,
) -> go.Figure:
    obs_column = resolve_obs_column(mdata, obs_column)

    # Set titles
    title_text = f"{format_modality(mdata, modality)} Level"
    xaxis_title = "Data Completeness (%)"
    yaxis_title = f"Cumulative proportion of {format_modality(mdata, modality)} (%)"
    hovertemplate = f"Data Completeness ≤ %{{x:.2f}}%<br>{yaxis_title} : %{{y:.2f}}% (%{{meta}})<extra></extra>"

    # Draw plot
    data = PlotData(mdata, modality=modality, obs_column=obs_column)
    plot = PlotScatter(
        data=data._prep_missingness_data(obs_column=obs_column),
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
    fig = _apply_layout_overrides(fig, kwargs)

    return fig


def plot_pca(
    mdata: md.MuData,
    modality: str = "protein",
    groupby: str | None = None,
    colorby: str = None,
    template: str = DEFAULT_TEMPLATE,
    pcs: tuple[int, int] | list[int] = (1, 2),
    obs_column: str | None = None,
    **kwargs,
) -> go.Figure:
    groupby, obs_column = _resolve_plot_columns(mdata, groupby, obs_column)

    # Get data
    pcs, pc_columns = _get_pc_cols(mdata, modality, pcs)
    variances = mdata[modality].uns["pca"]["variance_ratio"]

    # Set titles
    title_text = "PCA"
    xaxis_title = f"{pc_columns[0]} ({variances[pcs[0] - 1] * 100:.2f}%)"
    yaxis_title = f"{pc_columns[1]} ({variances[pcs[1] - 1] * 100:.2f}%)"
    hovertemplate = f"<b>%{{meta}}</b><br>{xaxis_title}: %{{x}}<br>{yaxis_title}: %{{y}}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, modality=modality, obs_column=obs_column)
    plot = PlotScatter(
        data=data._prep_pca_data(modality, groupby, pc_columns, obs_column=obs_column),
        x=pc_columns[0],
        y=pc_columns[1],
        name=groupby,
        meta=obs_column,
        hovertemplate=hovertemplate,
    )
    fig = plot.figure(mode="markers", marker=dict(size=10))

    # Update axis
    fig.update_yaxes(
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
    fig = _apply_layout_overrides(fig, kwargs)

    # Set color
    fig = _apply_color_if_needed(
        fig,
        mdata=mdata,
        modality=modality,
        groupby=groupby,
        colorby=colorby,
        obs_column=obs_column,
        template=template,
    )

    return fig


def plot_purity(
    mdata: md.MuData,
    groupby: str = "filename",
    ptype: str = "hist",
    bins: int = 30,
    **kwargs,
) -> go.Figure:
    # Set mods
    modality = "feature"

    # Set titles
    title_text = "Precursor Isolation Purity Distribution"
    threshold = [v["filter_purity"] for k, v in mdata[modality].uns["filter"].items()][0]

    # Draw plot
    data = PlotData(mdata, modality=modality)
    if ptype in ["hist", "histogram"]:
        xaxis_title = "Precursor Isolation Purity"
        yaxis_title = "Number of PSMs"
        purity_data = data._prep_purity_data(groupby)
        bin_info = data._get_bin_info(purity_data["purity"], bins)
        hovertemplate = f"<b>%{{meta}}</b><br>{xaxis_title}: %{{x}} ± {round(bin_info['width'] / 2, 4)}<br>{yaxis_title}: %{{y:2,d}}<extra></extra>"
        plot = PlotHistogram(
            data=data._prep_purity_data_hist(purity_data, groupby, bin_info=bin_info),
            x="center",
            y="count",
            name="name",
            hovertemplate=hovertemplate,
        )
        fig = plot.figure()

        if not np.isnan(threshold):
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="red",
                line_width=1,
                annotation=dict(
                    text=f"Purity threshold : {threshold}",
                    yanchor="bottom",
                ),
            )

        fig.update_layout(
            yaxis_tickformat=",d",
        )

    elif ptype == "box":
        xaxis_title = "Raw Filenames"
        yaxis_title = "Precursor Isolation Purity"

        plot = PlotSimpleBox(data=data._prep_purity_data_box(groupby))
        fig = plot.figure()

        if not np.isnan(threshold):
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                line_width=1,
                annotation=dict(
                    text=f" Purity threshold : {threshold}",
                    xanchor="left",
                    yanchor="top",
                    x=0,
                ),
            )
    elif ptype in ["vln", "violin"]:
        xaxis_title = "Raw Filenames"
        yaxis_title = "Precursor Isolation Purity"

        plot = PlotViolin(
            data=data._prep_purity_data_vln(groupby),
            x=groupby,
            y="purity",
            name=groupby,
        )

        fig = plot.figure(
            spanmode="hard",
            points="suspectedoutliers",
            marker=dict(line=dict(outlierwidth=0)),
            box=dict(visible=True),
            meanline=dict(visible=True),
        )

        if not np.isnan(threshold):
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                line_width=1,
                annotation=dict(
                    text=f" Purity threshold : {threshold}",
                    xanchor="left",
                    yanchor="top",
                    x=0,
                ),
            )
    else:
        raise ValueError(f"Unknown plot type: {ptype}, choose from 'hist|histogram', 'box'")

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend=dict(title_text="Raw Filenames"),
    )

    # Update layout with kwargs
    fig = _apply_layout_overrides(fig, kwargs)

    return fig


def plot_umap(
    mdata: md.MuData,
    modality: str = "protein",
    groupby: str | None = None,
    colorby: str = None,
    template: str = DEFAULT_TEMPLATE,
    obs_column: str | None = None,
    **kwargs,
) -> go.Figure:
    groupby, obs_column = _resolve_plot_columns(mdata, groupby, obs_column)

    # Get required data
    umap_columns = _get_umap_cols(mdata, modality)

    # Set titles
    title_text = "UMAP"
    xaxis_title = umap_columns[0]
    yaxis_title = umap_columns[1]
    hovertemplate = f"<b>%{{meta}}</b><br>{xaxis_title}: %{{x}}<br>{yaxis_title}: %{{y}}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, modality=modality, obs_column=obs_column)
    plot = PlotScatter(
        data=data._prep_umap_data(modality, groupby, umap_columns, obs_column=obs_column),
        x=umap_columns[0],
        y=umap_columns[1],
        name=groupby,
        meta=obs_column,
        hovertemplate=hovertemplate,
    )
    fig = plot.figure(mode="markers", marker=dict(size=10))

    # Update axis
    fig.update_yaxes(
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
    fig = _apply_layout_overrides(fig, kwargs)

    # Set color
    fig = _apply_color_if_needed(
        fig,
        mdata=mdata,
        modality=modality,
        groupby=groupby,
        colorby=colorby,
        obs_column=obs_column,
        template=template,
    )

    return fig


def plot_peptide_length(
    mdata: md.MuData,
    groupby: str | None = None,
    colorby: str | None = None,
    template: str = DEFAULT_TEMPLATE,
    obs_column: str | None = None,
    ptype: str = "box",
    **kwargs,
) -> go.Figure:
    # Set mods
    modality: str = "peptide"
    groupby, obs_column = _resolve_plot_columns(mdata, groupby, obs_column)

    # Set titles
    title_text = "Peptide Length Distribution"
    xaxis_title = f"{groupby.capitalize()}"
    yaxis_title = "Length"

    # Draw plot
    if ptype in ["box", "boxplot"]:
        data = PlotData(mdata, modality=modality, obs_column=obs_column)
        plot = PlotSimpleBox(data=data._prep_peptide_length_data(groupby, obs_column=obs_column))
        fig = plot.figure()
    elif ptype in ["vln", "violin"]:
        data = PlotData(mdata, modality=modality, obs_column=obs_column)
        plot = PlotViolin(
            data=data._prep_peptide_length_data_vln(groupby, obs_column=obs_column),
            x=groupby,
            y="peptide_length",
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
        raise ValueError(f"Unknown plot type: {ptype}, choose from 'box|boxplot', 'vln|violin'")

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=True,
        legend=dict(title_text=f"{groupby.capitalize()}"),
    )

    # Update layout with kwargs
    fig = _apply_layout_overrides(fig, kwargs)

    # Set color
    fig = _apply_color_if_needed(
        fig,
        mdata=mdata,
        modality=modality,
        groupby=groupby,
        colorby=colorby,
        obs_column=obs_column,
        template=template,
    )

    return fig


def plot_missed_cleavage(
    mdata: md.MuData,
    modality: str = "feature",
    groupby: str | None = None,
    obs_column: str | None = None,
    **kwargs,
) -> go.Figure:
    groupby, obs_column = _resolve_plot_columns(mdata, groupby, obs_column)

    # Set titles
    title_text = "Number of PSMs by Missed Cleavages"
    xaxis_title = f"{groupby.capitalize()}"
    yaxis_title = "Number of PSMs"
    hovertemplate = "Missed Cleavages: %{meta}<br>Number of PSMs: %{y:2,d}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, modality=modality, obs_column=obs_column)
    plot = PlotStackedBar(
        data=data._prep_var_data(groupby, "missed_cleavages", obs_column=obs_column),
        x=groupby,
        y="count",
        name="missed_cleavages",
        meta="missed_cleavages",
        hovertemplate=hovertemplate,
    )
    fig = plot.figure()

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis_tickformat=",d",
        legend=dict(title_text="Missed Cleavages"),
    )

    # Update layout with kwargs
    fig = _apply_layout_overrides(fig, kwargs)

    return fig


def plot_upset(
    mdata: md.MuData,
    modality: str = "protein",
    subset: str = None,
    subset_column: str | None = None,
    groupby: str | None = None,
    obs_column: str | None = None,
    **kwargs,
):
    subset_column = resolve_obs_column(mdata, subset_column)

    if subset is not None:
        mdata = mdata[mdata.obs[subset_column] == subset].copy()

    groupby, obs_column = _resolve_plot_columns(mdata, groupby, obs_column)

    title_text = f"Intersection of Proteins among {groupby.capitalize()}"

    # Draw plot
    data = PlotData(mdata, modality=modality, obs_column=obs_column)
    plot = PlotUpset(
        data=data._prep_upset_data(groupby, obs_column=obs_column),
    )
    fig = plot.figure()

    # Update layout
    fig.update_layout(
        title_text=title_text,
    )

    # Update layout with kwargs
    fig = _apply_layout_overrides(fig, kwargs)

    return fig


def plot_correlation(
    mdata: md.MuData,
    groupby: str | None = None,
    modality: str = "protein",
    obs_column: str | None = None,
    **kwargs,
):
    groupby, obs_column = _resolve_plot_columns(mdata, groupby, obs_column)

    # Set titles
    title_text = "Correlation Heatmap"

    # Draw plot
    data = PlotData(mdata, modality=modality, obs_column=obs_column)
    plot = PlotHeatmap(
        data=data._prep_correlation_data(groupby, obs_column=obs_column),
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
    fig = _apply_layout_overrides(fig, kwargs)

    return fig


def plot_purity_metrics(
    mdata: md.MuData,
    mode: str = "ratio",
    **kwargs,
):
    # Set mods
    modality: str = "feature"

    # Set titles
    title_text = "Precursor Isolation Purity Metrics"
    # xaxis_title = "Fractions"
    yaxis_title = "Ratio of PSMs"

    # Draw plot
    hovertemplate = "<b>%{x}</b><br>Category: %{meta}<br>Ratio of PSMs: %{y:.2f}%<extra></extra>"
    data = PlotData(mdata, modality=modality)
    plot = PlotStackedBar(
        data=data._prep_purity_metrics_data(),
        x="filename",
        y=mode,
        name="purity_metrics",
        meta="purity_metrics",
        text=mode,
        hovertemplate=hovertemplate,
    )
    fig = plot.figure(
        texttemplate="%{text:.2f}",
        textfont=dict(size=10),
    )

    # Update layout
    fig.update_layout(
        title_text=title_text,
        # xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis_showticklabels=False,
        legend=dict(
            orientation="h",
            xanchor="right",
            yanchor="bottom",
            x=1,
            y=1,
        ),
    )

    # Update layout with kwargs
    fig = _apply_layout_overrides(fig, kwargs)

    return fig


def plot_tolerable_termini(
    mdata: md.MuData,
    modality: str = "feature",
    groupby: str | None = None,
    obs_column: str | None = None,
    **kwargs,
) -> go.Figure:
    groupby, obs_column = _resolve_plot_columns(mdata, groupby, obs_column)

    # Set titles
    title_text = "Number of PSMs by tolerable termini"
    xaxis_title = f"{groupby.capitalize()}"
    yaxis_title = "Number of PSMs"
    hovertemplate = "Tolerable termini: %{meta}<br>Number of PSMs: %{y:2,d}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, modality=modality, obs_column=obs_column)
    plot_data = data._prep_var_data(groupby, "semi_enzymatic", obs_column=obs_column)
    plot_data["semi_enzymatic"] = plot_data["semi_enzymatic"].map({0: "fully", 1: "semi"})
    plot = PlotStackedBar(
        data=plot_data,
        x=groupby,
        y="count",
        name="semi_enzymatic",
        meta="semi_enzymatic",
        hovertemplate=hovertemplate,
    )

    fig = plot.figure()

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis_tickformat=",d",
        legend=dict(title_text="Tolerable termini"),
    )

    # Update layout with kwargs
    fig = _apply_layout_overrides(fig, kwargs)

    return fig


def plot_var(
    mdata: md.MuData,
    modality: str = "feature",
    groupby: str | None = None,
    var_column: str = None,
    obs_column: str | None = None,
    ptype: str | None = None,
    bins: int = 30,
    **kwargs,
) -> go.Figure:
    groupby, obs_column = _resolve_plot_columns(mdata, groupby, obs_column)

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
    data = PlotData(mdata, modality=modality, obs_column=obs_column)
    if ptype in ["stack", "stackd", "stacked_bar"]:
        plot_data = data._prep_var_bar(groupby, var_column, obs_column=obs_column)
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
        plot_data = data._prep_var_box(groupby, var_column, obs_column=obs_column)
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
        plot_data = data._prep_var_simple_box(groupby, var_column, obs_column=obs_column)
        plot = PlotSimpleBox(data=plot_data)
        fig = plot.figure()
    elif ptype in ["vln", "violin"]:
        plot_data = data._prep_var_box(groupby, var_column, obs_column=obs_column)
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
        plot_data = data._prep_var_hist(groupby, var_column, obs_column=obs_column, bin_info=bin_info)
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
    fig = _apply_layout_overrides(fig, kwargs)

    return fig
