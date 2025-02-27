import numpy as np
import pandas as pd
import mudata as md
import plotly.graph_objects as go

from ._common import _draw_histogram, _draw_box
from ._utils import _get_traces, _get_bin_info
from .._utils import get_modality_dict


def plot_purity(
    mdata: md.MuData,
    level: str = None,
    modality: str = None,
    groupby: str = "filename",
    plot: str = "hist",
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
    if plot in ["hist", "histogram"]:
        data, bin_info = _prep_purity_data_hist(mdata, groupby, mods, bins)
        traces = _get_traces(data, "center", "count", "name")
        fig: go.Figure = _draw_histogram(
            traces=traces,
            title_text=title_text,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            bin_info=bin_info,
        )
    elif plot == "box":
        data = _prep_purity_data_box(mdata, groupby, mods)
        traces = _get_traces(data)
        fig: go.Figure = _draw_box(
            traces=traces,
            title_text=title_text,
            xaxis_title=xaxis_title,
        )
    else:
        raise ValueError(f"Unknown plot type: {plot}, choose from 'hist|histogram', 'box'")

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

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    return fig


def _prep_purity_data_hist(
    mdata: md.MuData,
    groupby: str = None,
    mods: list[str] = None,
    bins: int = 100,
) -> pd.DataFrame:
    # Prepare data
    data = pd.concat([mdata[mod].var for mod in mods])
    if groupby is not None:
        data = data[[groupby, "purity"]]
    else:
        data = data[["purity"]]
    data = data[data["purity"] >= 0]

    # get bin data
    bin_info = _get_bin_info(data["purity"], bins)

    # Treat groupby
    data["bin"] = pd.cut(data["purity"], bins=bin_info["edges"], labels=bin_info["labels"], include_lowest=True)
    if groupby is not None:
        grouped = data.groupby([groupby, "bin"], observed=True).size().unstack(fill_value=0)
        bin_counts = grouped.values.flatten()
        bin_frequencies = bin_counts / data.shape[0]
        bin_names = grouped.index.get_level_values(0).repeat(bins).tolist()

        # make dataframe
        prepped = pd.DataFrame(
            {
                "center": bin_info["centers"] * len(grouped),
                "label": bin_info["labels"] * len(grouped),
                "count": bin_counts,
                "frequency": bin_frequencies,
                "name": bin_names,
            }
        )
    else:
        bin_counts = data["bin"].value_counts(sort=False).values
        bin_frequencies = bin_counts / data.shape[0]

        # make dataframe
        prepped = pd.DataFrame(
            {
                "center": bin_info["centers"],
                "label": bin_info["labels"],
                "count": bin_counts,
                "frequency": bin_frequencies,
                "name": "Purity",
            }
        )
    return prepped, bin_info


def _prep_purity_data_box(
    mdata: md.MuData,
    groupby: str,
    mods: list[str],
) -> pd.DataFrame:
    if groupby not in mdata[mods[0]].var.columns:
        raise ValueError(f"{groupby} not in var columns")

    # Prepare data
    orig_df = pd.concat([mdata[mod].var for mod in mods])[[groupby, "purity"]]
    orig_df[["purity"]] = orig_df[["purity"]][orig_df[["purity"]] >= 0]

    prep_df = orig_df.groupby(groupby, observed=True).describe().droplevel(0, axis=1)

    return prep_df


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
