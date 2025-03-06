import numpy as np
import pandas as pd
import mudata as md
import plotly.graph_objects as go

from .__pdata import PlotData
from .__ptypes import PlotHistogram, PlotBox
from .._utils import get_modality_dict


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


def plot_purity_metrics(
    mdata: md.MuData,
    modality: str = "psm",
    groupby: str = None,
    normalize: bool = True,
    **kwargs,
) -> go.Figure:
    # Prepare dataset
    dataset, data_colname = _prep_purity_metrics(
        data_var=mdata[modality].var,
        data_varm=mdata[modality].varm,
        groupby=groupby,
        normalize=normalize,
    )

    # Initialize figure
    fig = go.Figure()

    # Add traces
    fig: go.Figure = _add_purity_metrics_trace(
        fig=fig,
        dataset=dataset,
        data_colname=data_colname,
    )

    # Update layout
    fig.update_layout(
        width=800,
        height=400,
        barmode="stack",
        title_text="Precursor purity metrics",
        xaxis_title=data_colname.capitalize(),
        xaxis_tickformat=".2f",
        legend_traceorder="normal",
    )

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    return fig


def _prep_purity_metrics(
    data_var: pd.DataFrame,
    data_varm: pd.DataFrame,
    groupby: str = None,
    normalize: bool = True,
):
    data = pd.concat([data_var, data_varm["filter"]["filter_purity"]], axis=1)

    # Define conditions and choices for purity_result
    conditions = [
        data["filter_purity"] == True,
        (data["filter_purity"] == False) & (data["purity"] >= 0),
        data["purity"] == -1,
        data["purity"] == -2,
    ]
    choices = [
        "High purity",
        "Low purity",
        "No isotope peak",
        "No isolation peak",
    ]
    data["purity_result"] = np.select(condlist=conditions, choicelist=choices, default="Unknown")

    # Treat groupby
    if groupby:
        dataset = (
            data[[groupby, "purity_result"]]
            .groupby(groupby, observed=False)
            .value_counts(normalize=normalize)
            .reset_index()
        )
        dataset["groupby"] = dataset[groupby].astype(str)
    else:
        dataset = data["purity_result"].value_counts(normalize=normalize).reset_index()
        dataset["groupby"] = "MuData"

    # Column name for data
    if normalize:
        data_colname = "proportion"
    else:
        data_colname = "count"

    return dataset, data_colname


def _add_purity_metrics_trace(
    fig: go.Figure,
    dataset: pd.DataFrame,
    data_colname: str = "purity_result",
) -> go.Figure:
    for purity_category in dataset["purity_result"].unique():
        fig.add_trace(
            go.Bar(
                x=dataset.loc[dataset["purity_result"] == purity_category, data_colname],
                y=dataset.loc[dataset["purity_result"] == purity_category, "groupby"],
                name=purity_category,
                hovertemplate=f"{purity_category}: %{{x:.2f}}<extra></extra>",
                orientation="h",
            )
        )

    fig.update_yaxes(autorange="reversed")

    return fig
