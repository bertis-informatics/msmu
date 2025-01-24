import numpy as np
import pandas as pd
import mudata as md
import plotly.graph_objects as go


def plot_purity(
    mdata: md.MuData,
    modality: str = "psm",
    groupby: str = None,
    binsize: float = 0.01,
    plot: str = "hist",
    **kwargs,
) -> go.Figure:
    # Prepare dataset
    dataset = _prep_purity(
        data_var=mdata[modality].var,
        groupby=groupby,
    )

    if plot == "hist":
        fig = draw_purity_hist(dataset=dataset, binsize=binsize)
    elif plot in ["dist", "density", "violin"]:
        fig = draw_purity_density(dataset=dataset)
    elif plot == "box":
        fig = draw_purity_box(dataset=dataset)
    else:
        raise ValueError(f"Unknown plot type: {plot}, choose from 'hist', 'dist|density|violin', 'box'")

    # Add threshold line
    threshold = mdata[modality].uns["filter"]["purity"]
    fig.add_vline(
        x=threshold["min"],
        line_dash="dash",
        line_color="red",
        line_width=1,
        annotation_text=" > Purity threshold",
    )

    # Update layout
    fig.update_layout(
        width=800,
        height=400,
        title_text="Precursor purity distribution",
        xaxis_title="Precursor purity",
        xaxis_tickformat=".2f",
    )

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    return fig


def _prep_purity(
    data_var: pd.DataFrame,
    groupby: str = None,
) -> dict:

    # Treat groupby
    if groupby:
        categories = data_var[groupby].cat.categories
        dataset = {c: data_var.loc[data_var[groupby] == c, "purity"] for c in categories}
    else:
        dataset = {"MuData": data_var["purity"]}

    # Filter out zero purity values less than 0
    dataset = {k: v[v >= 0] for k, v in dataset.items()}

    return dataset


def draw_purity_hist(
    dataset: dict,
    binsize: float = 0.04,
) -> go.Figure:
    fig = go.Figure()
    for category, data in dataset.items():
        fig.add_trace(
            go.Histogram(
                x=data,
                xbins=dict(size=binsize),
                name=category,
                hovertemplate="Purity: %{x}<br>Count: %{y:.0f}<extra></extra>",
            )
        )

    fig.update_layout(
        yaxis_title="Number of PSMs",
        yaxis_tickformat=".0f",
    )

    return fig


def draw_purity_density(
    dataset: dict,
) -> go.Figure:
    fig = go.Figure()
    for category, data in dataset.items():
        fig.add_trace(
            go.Violin(
                x=data,
                name=category,
                showlegend=False,
                width=1.8,
                points=False,
                hoveron="points",
                orientation="h",
                side="positive",
                line=dict(width=1),
                bandwidth=0.01,
                spanmode="hard",
                box=dict(
                    visible=True,
                    fillcolor="white",
                    line=dict(width=1),
                ),
            )
        )

    return fig


def draw_purity_box(
    dataset: dict,
) -> go.Figure:
    fig = go.Figure()
    for category, data in dataset.items():
        fig.add_trace(
            go.Box(
                x=data,
                name=category,
                boxpoints=False,
                hoverinfo="x",
                orientation="h",
                showlegend=False,
            )
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
    data = pd.concat([data_var, data_varm["filter"]["high_purity"]], axis=1)

    # Define conditions and choices for purity_result
    conditions = [
        data["high_purity"] == True,
        (data["high_purity"] == False) & (data["purity"] >= 0),
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
