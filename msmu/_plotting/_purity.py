import numpy as np
import pandas as pd
import mudata as md
import plotly.graph_objects as go

from ._common import _draw_histogram, _draw_density, _draw_box
from ._utils import _get_traces


def plot_purity(
    mdata: md.MuData,
    modality: str = "psm",
    groupby: str = None,
    plot: str = "hist",
    binsize: float = 0.01,
    bandwidth: float = 0.01,
    **kwargs,
) -> go.Figure:
    # Prepare dataset
    data = _prep_purity_data(mdata, modality, groupby)

    # Get traceset
    traces = _get_traces(data)

    # Set titles
    title_text = "Precursor purity distribution"
    xaxis_title = "Precursor purity"
    yaxis_title = "Number of PSMs"

    # Draw plot
    if plot in ["hist", "histogram"]:
        fig: go.Figure = _draw_histogram(
            traces=traces,
            binsize=binsize,
            title_text=title_text,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
        )
    elif plot in ["density", "violin"]:
        fig: go.Figure = _draw_density(
            traces=traces,
            bandwidth=bandwidth,
            title_text=title_text,
            xaxis_title=xaxis_title,
        )
    elif plot == "box":
        fig: go.Figure = _draw_box(
            traces=traces,
            title_text=title_text,
            xaxis_title=xaxis_title,
        )
    else:
        raise ValueError(f"Unknown plot type: {plot}, choose from 'hist|histogram', 'density|violin', 'box'")

    # Add threshold line
    threshold = mdata[modality].uns["filter"]["filter_purity"]
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

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    return fig


def _prep_purity_data(
    mdata: md.MuData,
    modality: str = "psm",
    groupby: str = None,
) -> pd.DataFrame:
    data_var = mdata[modality].var

    # Treat groupby
    if groupby:
        data = data_var.pivot_table(index=data_var.index, columns=groupby, values="purity", observed=True)
    else:
        data = data_var[["purity"]]

    # Filter out zero purity values less than 0
    data = data[data >= 0]

    return data.T


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
