import mudata as md
import plotly.graph_objects as go

from .._utils import get_modality_dict
from .__pdata import PlotData
from .__ptypes import *
from ._template import DEFAULT_TEMPLATE
from ._utils import DEFAULT_COLUMN, _get_pc_cols, _get_umap_cols, _set_color


def format_modality(modality: str) -> str:
    if modality == "feature":
        return "Feature"
    elif modality == "peptide":
        return "Peptide"
    elif modality == "protein":
        return "Protein"
    else:
        raise ValueError(
            f"Unknown modality: {modality}, choose from 'feature', 'peptide', 'protein'"
        )


def plot_charge(
    mdata: md.MuData,
    modality: str = "feature",
    groupby: str = DEFAULT_COLUMN,
    obs_column: str = DEFAULT_COLUMN,
    **kwargs,
) -> go.Figure:
    # Set titles
    title_text = "Number of PSMs by Charge State"
    xaxis_title = f"{groupby.capitalize()}s"
    yaxis_title = "Number of PSMs"
    hovertemplate = "Charge: %{meta}<br>Number of PSMs: %{y:2,d}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, modality=modality)
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
        legend=dict(title_text="Charge State"),
    )

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    return fig


def plot_id(
    mdata: md.MuData,
    modality: str,
    groupby: str = DEFAULT_COLUMN,
    colorby: str | None = None,
    template: str = DEFAULT_TEMPLATE,
    obs_column: str = DEFAULT_COLUMN,
    **kwargs,
) -> go.Figure:
    if groupby not in [obs_column]:
        ytemplate = "%{y:,.1f}"
    else:
        ytemplate = "%{y:,d}"

    # Set titles
    title_text = f"Number of {format_modality(modality)}s"
    xaxis_title = f"{groupby.capitalize()}s"
    yaxis_title = f"Number of {format_modality(modality)}s"
    hovertemplate = (
        f"{xaxis_title}: %{{x}}<br>{yaxis_title}: {ytemplate}<extra></extra>"
    )

    # Draw plot
    data = PlotData(mdata, modality=modality)
    plot = PlotBar(
        data=data._prep_id_data(groupby, obs_column=obs_column),
        x=groupby,
        y="_count",
        name=groupby,
        hovertemplate=hovertemplate,
        text="_count",
    )
    fig = plot.figure()
    # Update traces
    fig.update_traces(texttemplate=ytemplate)

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=True,
        legend=dict(title_text=f"{groupby.capitalize()}s"),
    )

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    # Set color
    if (colorby is not None) & (groupby in [obs_column]):
        fig = _set_color(fig, mdata, modality, colorby, template)

    return fig


def plot_id_fraction(
    mdata: md.MuData,
    modality: str,
    groupby: str = "filename",
    **kwargs,
) -> go.Figure:
    # Set titles
    title_text = f"Number of {format_modality(modality)}s by Fraction"
    # xaxis_title = f"Filenames"
    yaxis_title = f"Number of {format_modality(modality)}s"
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
        showlegend=False,
        xaxis_showticklabels=False,
    )

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    return fig


def plot_intensity(
    mdata: md.MuData,
    modality: str,
    groupby: str = DEFAULT_COLUMN,
    colorby: str | None = None,
    ptype: str = "hist",
    template: str = DEFAULT_TEMPLATE,
    bins: int = 30,
    obs_column: str = DEFAULT_COLUMN,
    **kwargs,
) -> go.Figure:
    # Set titles
    title_text = f"{format_modality(modality)} Intensity Distribution"

    # Draw plot
    if ptype in ["hist", "histogram"]:
        xaxis_title = "Intensity (log<sub>2</sub>)"
        yaxis_title = f"Number of {format_modality(modality)}s"

        data = PlotData(mdata, modality=modality)
        bin_info = data._get_bin_info(data._get_data(), bins)
        hovertemplate = f"<b>%{{meta}}</b><br>{xaxis_title}: %{{x}} ± {round(bin_info['width'] / 2, 4)}<br>{yaxis_title}: %{{y:2,d}}<extra></extra>"
        plot = PlotHistogram(
            data=data._prep_intensity_data_hist(groupby, bins, obs_column=obs_column),
            x="center",
            y="count",
            name="name",
            hovertemplate=hovertemplate,
        )
        fig = plot.figure()

    elif ptype == "box":
        xaxis_title = f"{groupby.capitalize()}s"
        yaxis_title = "Intensity (log<sub>2</sub>)"

        data = PlotData(mdata, modality=modality)
        plot = PlotBox(data=data._prep_intensity_data_box(groupby))
        fig = plot.figure()

    else:
        raise ValueError(
            f"Unknown plot type: {ptype}, choose from 'hist|histogram', 'box'"
        )

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=True,
        legend=dict(title_text=f"{groupby.capitalize()}s"),
    )

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    # Set color
    if (colorby is not None) & (groupby in [obs_column]):
        fig = _set_color(fig, mdata, modality, colorby, template)

    return fig


def plot_missingness(
    mdata: md.MuData,
    modality: str,
    obs_column: str = DEFAULT_COLUMN,
    **kwargs,
) -> go.Figure:
    # Set titles
    title_text = f"{format_modality(modality)} Level"
    xaxis_title = "Data Completeness (%)"
    yaxis_title = f"Cumulative proportion of {format_modality(modality)} (%)"
    hovertemplate = f"Data Completeness ≤ %{{x:.2f}}%<br>{yaxis_title} : %{{y:.2f}}% (%{{meta}})<extra></extra>"

    # Draw plot
    data = PlotData(mdata, modality=modality)
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
    fig.update_layout(
        **kwargs,
    )

    return fig


def plot_pca(
    mdata: md.MuData,
    modality: str = "protein",
    groupby: str = DEFAULT_COLUMN,
    colorby: str = None,
    template: str = DEFAULT_TEMPLATE,
    pcs: tuple[int, int] | list[int] = (1, 2),
    obs_column: str = DEFAULT_COLUMN,
    **kwargs,
) -> go.Figure:
    # Get data
    pcs, pc_columns = _get_pc_cols(mdata, modality, pcs)
    variances = mdata[modality].uns["pca"]["variance_ratio"]

    # Set titles
    title_text = "PCA"
    xaxis_title = f"{pc_columns[0]} ({variances[pcs[0] - 1] * 100:.2f}%)"
    yaxis_title = f"{pc_columns[1]} ({variances[pcs[1] - 1] * 100:.2f}%)"
    hovertemplate = (
        "<b>%{meta}</b><br>"
        + xaxis_title
        + ": %{x}<br>"
        + yaxis_title
        + ": %{y}<extra></extra>"
    )

    # Draw plot
    data = PlotData(mdata, modality=modality)
    plot = PlotScatter(
        data=data._prep_pca_data(modality, groupby, pc_columns, obs_column=obs_column),
        x=pc_columns[0],
        y=pc_columns[1],
        name=groupby,
        meta=obs_column,
        hovertemplate=hovertemplate,
    )
    fig = plot.figure(mode="markers")

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
            orientation="h",
            xanchor="right",
            yanchor="bottom",
            x=1,
            y=1,
        ),
    )

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    # Set color
    if (colorby is not None) & (groupby in [obs_column]):
        fig = _set_color(fig, mdata, modality, colorby, template)

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
    threshold = mdata[modality].uns["filter"]["filter_purity"]

    # Draw plot
    if ptype in ["hist", "histogram"]:
        xaxis_title = "Precursor Isolation Purity"
        yaxis_title = "Number of PSMs"

        data = PlotData(mdata, modality=modality)
        data._prep_purity_data(groupby)
        bin_info = data._get_bin_info(data.X["purity"], bins)
        hovertemplate = f"<b>%{{meta}}</b><br>{xaxis_title}: %{{x}} ± {round(bin_info['width'] / 2, 4)}<br>{yaxis_title}: %{{y:2,d}}<extra></extra>"
        plot = PlotHistogram(
            data=data._prep_purity_data_hist(groupby, bins),
            x="center",
            y="count",
            name="name",
            hovertemplate=hovertemplate,
        )
        fig = plot.figure()

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
    elif ptype == "box":
        xaxis_title = "Raw Filenames"
        yaxis_title = "Precursor Isolation Purity"

        data = PlotData(mdata, modality=modality)
        plot = PlotBox(data=data._prep_purity_data_box(groupby))
        fig = plot.figure()

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
        raise ValueError(
            f"Unknown plot type: {ptype}, choose from 'hist|histogram', 'box'"
        )

    # Add threshold line

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend=dict(title_text="Raw Filenames"),
    )

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    return fig


def plot_umap(
    mdata: md.MuData,
    modality: str = "protein",
    groupby: str = DEFAULT_COLUMN,
    colorby: str = None,
    template: str = DEFAULT_TEMPLATE,
    obs_column: str = DEFAULT_COLUMN,
    **kwargs,
) -> go.Figure:
    # Get required data
    umap_columns = _get_umap_cols(mdata, modality)

    # Set titles
    title_text = "UMAP"
    xaxis_title = umap_columns[0]
    yaxis_title = umap_columns[1]
    hovertemplate = (
        "<b>%{meta}</b><br>"
        + xaxis_title
        + ": %{x}<br>"
        + yaxis_title
        + ": %{y}<extra></extra>"
    )

    # Draw plot
    data = PlotData(mdata, modality=modality)
    plot = PlotScatter(
        data=data._prep_umap_data(
            modality, groupby, umap_columns, obs_column=obs_column
        ),
        x=umap_columns[0],
        y=umap_columns[1],
        name=groupby,
        meta=obs_column,
        hovertemplate=hovertemplate,
    )
    fig = plot.figure(mode="markers")

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
            orientation="h",
            xanchor="right",
            yanchor="bottom",
            x=1,
            y=1,
        ),
    )
    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    # Set color
    if (colorby is not None) & (groupby in [obs_column]):
        fig = _set_color(fig, mdata, modality, colorby, template)

    return fig


def plot_peptide_length(
    mdata: md.MuData,
    groupby: str = DEFAULT_COLUMN,
    colorby: str = None,
    template: str = DEFAULT_TEMPLATE,
    obs_column: str = DEFAULT_COLUMN,
    **kwargs,
) -> go.Figure:
    # Set mods
    modality: str = "peptide"

    # Set titles
    title_text = "Peptide Length Distribution"
    xaxis_title = f"{groupby.capitalize()}s"
    yaxis_title = "Length"

    # Draw plot
    data = PlotData(mdata, modality=modality)
    plot = PlotBox(data=data._prep_peptide_length_data(groupby, obs_column=obs_column))
    fig = plot.figure()

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=True,
        legend=dict(title_text=f"{groupby.capitalize()}s"),
    )

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    # Set color
    if (colorby is not None) & (groupby in [obs_column]):
        fig = _set_color(fig, mdata, modality, colorby, template)

    return fig


def plot_missed_cleavage(
    mdata: md.MuData,
    modality: str = "psm",
    groupby: str = DEFAULT_COLUMN,
    obs_column: str = DEFAULT_COLUMN,
    **kwargs,
) -> go.Figure:
    # Set titles
    title_text = "Number of PSMs by Missed Cleavages"
    xaxis_title = f"{groupby.capitalize()}s"
    yaxis_title = "Number of PSMs"
    hovertemplate = (
        "Missed Cleavages: %{meta}<br>Number of PSMs: %{y:2,d}<extra></extra>"
    )

    # Draw plot
    data = PlotData(mdata, modality=modality)
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
        legend=dict(title_text="Missed Cleavages"),
    )

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    return fig


def plot_upset(
    mdata: md.MuData,
    modality: str = "protein",
    obs_column: str = DEFAULT_COLUMN,
    **kwargs,
):
    # Set titles
    title_text = "Intersection of Proteins among Samples"

    # Draw plot
    data = PlotData(mdata, modality=modality)
    plot = PlotUpset(
        data=data._prep_upset_data(obs_column=obs_column),
    )
    fig = plot.figure()

    # Update layout
    fig.update_layout(
        title_text=title_text,
    )

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    return fig


def plot_correlation(
    mdata: md.MuData,
    groupby: str = DEFAULT_COLUMN,
    modality: str = "protein",
    obs_column: str = DEFAULT_COLUMN,
    **kwargs,
):

    # Set titles
    title_text = "Correlation Heatmap"

    # Draw plot
    data = PlotData(mdata, modality=modality)
    plot = PlotHeatmap(
        data=data._prep_correlation_data(obs_column=obs_column, groupby=groupby),
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
    fig.update_layout(
        **kwargs,
    )

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
    hovertemplate = (
        "<b>%{x}</b><br>Category: %{meta}<br>Ratio of PSMs: %{y:.2f}%<extra></extra>"
    )
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
    fig.update_layout(
        **kwargs,
    )

    return fig
