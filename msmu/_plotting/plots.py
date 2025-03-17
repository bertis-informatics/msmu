import json
import mudata as md
import plotly.graph_objects as go

from .__pdata import PlotData
from .__ptypes import *
from ._template import DEFAULT_TEMPLATE
from ._utils import _set_color, _get_pc_cols, _get_umap_cols

from .._utils import get_modality_dict


DEFAULT_COLUMN = "_obs_"


def save_figure_data(fig: go.Figure, path: str) -> None:
    data = fig.to_dict()
    data.pop("layout")
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {path}")


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
    hovertemplate = "Charge: %{meta}<br>Number of PSMs: %{y:2,d}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, mods=mods)
    plot = PlotStackedBar(
        data=data._prep_charge_data(groupby, "charge"),
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
    )

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    return fig


def plot_id(
    mdata: md.MuData,
    level: str = None,
    modality: str = None,
    groupby: str = DEFAULT_COLUMN,
    colorby: str = None,
    template: str = DEFAULT_TEMPLATE,
    **kwargs,
) -> go.Figure:
    # Set mods
    mods = list(get_modality_dict(mdata, level=level, modality=modality).keys())

    # Set titles
    title_text = "Number of PSMs"
    xaxis_title = "Samples"
    yaxis_title = "Number of PSMs"
    if groupby is not None:
        title_text = f"Average Number of PSMs by {groupby}"
        xaxis_title = groupby
        yaxis_title = "Average number of PSMs"
    hovertemplate = f"{xaxis_title}: %{{x}}<br>{yaxis_title}: %{{y:2,d}}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, mods=mods)
    plot = PlotBar(
        data=data._prep_id_data(groupby),
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
    )

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    # Set color
    if (colorby is not None) & (groupby == DEFAULT_COLUMN):
        fig = _set_color(fig, mdata, mods, colorby, template)

    return fig


def plot_intensity(
    mdata: md.MuData,
    level: str = None,
    modality: str = None,
    groupby: str = DEFAULT_COLUMN,
    colorby: str = None,
    ptype: str = "hist",
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
    if ptype in ["hist", "histogram"]:
        data = PlotData(mdata, mods=mods)
        bin_info = data._get_bin_info(data._get_data(), bins)
        hovertemplate = f"<b>%{{meta}}</b><br>{xaxis_title}: %{{x}} ± {round(bin_info['width'] / 2, 4)}<br>{yaxis_title}: %{{y:2,d}}<extra></extra>"
        plot = PlotHistogram(
            data=data._prep_intensity_data_hist(groupby, bins),
            x="center",
            y="count",
            name="name",
            hovertemplate=hovertemplate,
        )
        fig = plot.figure()
    elif ptype == "box":
        data = PlotData(mdata, mods=mods)
        plot = PlotBox(data=data._prep_intensity_data_box(groupby))
        fig = plot.figure()
    else:
        raise ValueError(f"Unknown plot type: {ptype}, choose from 'hist|histogram', 'box'")

    # Update layout
    fig.update_layout(
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


def plot_pca(
    mdata: md.MuData,
    modality: str = "peptide",
    groupby: str = DEFAULT_COLUMN,
    colorby: str = None,
    template: str = DEFAULT_TEMPLATE,
    pcs: tuple[int, int] | list[int] = (1, 2),
    **kwargs,
) -> go.Figure:
    # Get data
    pcs, pc_columns = _get_pc_cols(mdata, modality, pcs)
    variances = mdata[modality].uns["pca"]["variance_ratio"]

    # Set titles
    title_text = "PCA"
    xaxis_title = f"{pc_columns[0]} ({variances[pcs[0] - 1] * 100:.2f}%)"
    yaxis_title = f"{pc_columns[1]} ({variances[pcs[1] - 1] * 100:.2f}%)"
    hovertemplate = "<b>%{meta}</b><br>" + xaxis_title + ": %{x}<br>" + yaxis_title + ": %{y}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, mods=[modality])
    plot = PlotScatter(
        data=data._prep_pca_data(modality, pc_columns),
        x=pc_columns[0],
        y=pc_columns[1],
        name=groupby,
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
    if (colorby is not None) & (groupby == DEFAULT_COLUMN):
        fig = _set_color(fig, mdata, modality, colorby, template)

    return fig


def plot_purity(
    mdata: md.MuData,
    level: str = None,
    modality: str = None,
    groupby: str = "filename",
    ptype: str = "hist",
    bins: int = 50,
    **kwargs,
) -> go.Figure:
    # Set mods
    mods = list(get_modality_dict(mdata, level=level, modality=modality).keys())

    # Set titles
    title_text = "Precursor purity distribution"
    xaxis_title = "Precursor purity"
    yaxis_title = "Number of PSMs"

    # Draw plot
    if ptype in ["hist", "histogram"]:
        data = PlotData(mdata, mods=mods)
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
    elif ptype == "box":
        data = PlotData(mdata, mods=mods)
        plot = PlotBox(data=data._prep_purity_data_box(groupby))
        fig = plot.figure()
    else:
        raise ValueError(f"Unknown plot type: {ptype}, choose from 'hist|histogram', 'box'")

    # Add threshold line
    threshold = mdata[mods[0]].uns["filter"]["filter_purity"]
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

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    return fig


def plot_umap(
    mdata: md.MuData,
    modality: str = "psm",
    groupby: str = DEFAULT_COLUMN,
    colorby: str = None,
    template: str = DEFAULT_TEMPLATE,
    **kwargs,
) -> go.Figure:
    # Get required data
    umap_columns = _get_umap_cols(mdata, modality)

    # Set titles
    title_text = "UMAP"
    xaxis_title = umap_columns[0]
    yaxis_title = umap_columns[1]
    hovertemplate = "<b>%{meta}</b><br>" + xaxis_title + ": %{x}<br>" + yaxis_title + ": %{y}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, mods=[modality])
    plot = PlotScatter(
        data=data._prep_umap_data(modality, umap_columns),
        x=umap_columns[0],
        y=umap_columns[1],
        name=groupby,
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
    if (colorby is not None) & (groupby == DEFAULT_COLUMN):
        fig = _set_color(fig, mdata, modality, colorby, template)

    return fig


def plot_peptide_length(
    mdata: md.MuData,
    level: str = "peptide",
    modality: str = None,
    groupby: str = DEFAULT_COLUMN,
    colorby: str = None,
    template: str = DEFAULT_TEMPLATE,
    **kwargs,
) -> go.Figure:
    # Set mods
    mods = list(get_modality_dict(mdata, level=level, modality=modality).keys())

    # Set titles
    title_text = "Number of peptides by length"
    xaxis_title = "Length"
    yaxis_title = "Number of peptides"
    hovertemplate = "Sample: %{meta}<br>Peptide Length: %{x}<br>Number of Peptides: %{y:2,d}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, mods=mods)
    plot = PlotBar(
        data=data._prep_peptide_length_data(groupby),
        x="peptide_length",
        y="count",
        name=groupby,
        hovertemplate=hovertemplate,
    )
    fig = plot.figure()

    # Update layout
    fig.update_layout(
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


def plot_missed_cleavage(
    mdata: md.MuData,
    level: str = "psm",
    groupby: str = DEFAULT_COLUMN,
    modality: str = None,
    **kwargs,
) -> go.Figure:
    # Set mods
    mods = list(get_modality_dict(mdata, level=level, modality=modality).keys())

    # Set titles
    title_text = "Number of PSMs by missed cleavages"
    xaxis_title = "Samples"
    yaxis_title = "Number of PSMs"
    hovertemplate = "Missed Cleavages: %{meta}<br>Number of PSMs: %{y:2,d}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, mods=mods)
    plot = PlotStackedBar(
        data=data._prep_charge_data(groupby, "missed_cleavages"),
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
    )

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    return fig


def plot_upset(
    mdata: md.MuData,
    level: str = "protein",
    modality: str = None,
    **kwargs,
):
    # Set mods
    mods = list(get_modality_dict(mdata, level, modality).keys())

    # Set titles
    title_text = "Upset Plot"

    # Draw plot
    data = PlotData(mdata, mods=mods)
    plot = PlotUpset(
        data=data._prep_upset_data(),
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
    level: str = "protein",
    modality: str = None,
    **kwargs,
):
    # Set mods
    mods = list(get_modality_dict(mdata, level, modality).keys())

    # Set titles
    title_text = "Correlation Plot"

    # Draw plot
    data = PlotData(mdata, mods=mods)
    plot = PlotHeatmap(data=data._prep_correlation_data())
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
