import pandas as pd
import mudata as md
import plotly.io as pio
import plotly.graph_objects as go


def _get_traces(data: pd.DataFrame) -> list[dict]:
    traces: list[dict] = []

    # Get traces
    for idx, row in data.iterrows():
        traces.append(dict(x=row, name=idx))

    return traces


def _get_2d_traces(data: pd.DataFrame, colname: str) -> list[dict]:
    traces: list[dict] = []

    # Get traces
    for idx, row in data.iterrows():
        traces.append(dict(x=[idx], y=[row[colname]], name=idx))

    return traces


def _set_color(
    fig: go.Figure,
    mdata: md.MuData,
    modality: str,
    colorby: str,
    template: str = None,
):
    # Get categories
    categories = mdata[modality].obs[colorby]

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
