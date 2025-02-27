import numpy as np
import pandas as pd
import mudata as md
import plotly.io as pio
import plotly.graph_objects as go


def _get_traces(
    data: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
    name: str | None = None,
    meta: str | None = None,
) -> list[dict]:
    if y is None:
        return _get_1d_traces(data)
    else:
        return _get_2d_traces(data, x, y, name, meta=meta)


def _get_1d_traces(
    data: pd.DataFrame,
) -> list[dict]:
    return [
        {
            "x": row,
            "name": idx,
            "text": idx,
        }
        for idx, row in data.iterrows()
    ]


def _get_2d_traces(
    data: pd.DataFrame,
    x: str,
    y: str,
    name: str = "_idx",
    meta: str = None,
) -> list[dict]:
    data = data.copy()
    data["_idx"] = data.index

    # Group by the specified name and create traces
    grouped = data.groupby(name, observed=True)
    return [
        dict(
            x=group[x].values,
            y=group[y].values,
            name=n,
            meta=group[meta].values if meta is not None else n,
        )
        for n, group in grouped
    ]


def _set_color(
    fig: go.Figure,
    mdata: md.MuData,
    mods: list[str],
    colorby: str,
    template: str = None,
):
    # Get categories
    categories = pd.concat([mdata[modality].obs[colorby] for modality in mods])

    # Get colors
    colors = pio.templates[template].layout["colorway"]

    colormap_dict = {val: colors[i % len(colors)] for i, val in enumerate(categories.unique())}
    colormap = categories.map(colormap_dict)

    # Update figure
    for i, trace in enumerate(fig.data):
        if hasattr(trace, "marker"):
            trace.marker.color = colormap[trace.name]
        if hasattr(trace, "line"):
            trace.line.color = colormap[trace.name]

    return fig


def _merge_traces(
    traces: list[dict],
    options: dict,
) -> list[dict]:
    merged_traces = []
    for trace in traces:
        merged_traces.append({**trace, **options})

    return merged_traces


def _get_bin_info(data: pd.DataFrame, bins: int) -> dict:
    # get bin data
    min_value = np.min(data)
    max_value = np.max(data)
    data_range = max_value - min_value
    bin_width = data_range / bins
    bin_edges = [min_value + bin_width * i for i in range(bins + 1)]
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(bins)]
    bin_labels = [f"{bin_edges[i]} - {bin_edges[i + 1]}" for i in range(bins)]

    return {
        "width": bin_width,
        "edges": bin_edges,
        "centers": bin_centers,
        "labels": bin_labels,
    }
