import plotly.graph_objects as go

from ._utils import _merge_traces


def _draw_histogram(
    traces: list[dict],
    title_text: str,
    xaxis_title: str,
    yaxis_title: str,
    bin_info: float,
) -> go.Figure:
    # Create figure
    fig = go.Figure()

    # Set options
    hovertemplate = f"<b>%{{meta}}</b><br>{xaxis_title}: %{{x}} ± {round(bin_info['width'] / 2, 4)}<br>{yaxis_title}: %{{y:2,d}}<extra></extra>"
    options = {
        "hovertemplate": hovertemplate,
        "text": None,
    }
    traces = _merge_traces(traces, options)

    # Add traces
    fig.add_traces([go.Bar(**trace) for trace in traces])

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_title_text=xaxis_title,
    )

    return fig


def _draw_box(
    traces: list[dict],
    title_text: str,
    xaxis_title: str,
) -> go.Figure:
    # Create figure
    fig = go.Figure()

    # Add traces
    fig.add_traces(
        [
            go.Box(
                y=[trace["name"]],
                lowerfence=[trace["x"]["min"]],
                q1=[trace["x"]["25%"]],
                median=[trace["x"]["50%"]],
                q3=[trace["x"]["75%"]],
                upperfence=[trace["x"]["max"]],
                boxpoints=False,
                hoverinfo="x",
                orientation="h",
                showlegend=False,
                name=trace["name"],
            )
            for trace in traces
        ]
    )

    # Update axes
    fig.update_yaxes(autorange="reversed")

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
    )

    return fig


def _draw_bar(
    traces: list[dict],
    title_text: str,
    xaxis_title: str,
    yaxis_title: str,
) -> go.Figure:
    # Create figure
    fig = go.Figure()

    # Set options
    hovertemplate = f"{xaxis_title}: %{{x}}<br>{yaxis_title}: %{{y:2,d}}<extra></extra>"
    options = {
        "hovertemplate": hovertemplate,
        "texttemplate": "%{y:,d}",
    }
    traces = _merge_traces(traces, options)

    # Add traces
    fig.add_traces([go.Bar(**trace) for trace in traces])

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_title_text=xaxis_title,
    )

    return fig


def _draw_scatter(
    traces: list[dict],
    title_text: str,
    xaxis_title: str,
    yaxis_title: str,
    mode: str = "markers",
    hovertemplate: str = None,
    **kwargs,
) -> go.Figure:
    # Create figure
    fig = go.Figure()

    # Set hovertemplate
    if hovertemplate is None:
        hovertemplate = "<b>%{meta}</b><br>" + xaxis_title + ": %{x}<br>" + yaxis_title + ": %{y}<extra></extra>"
    options = {
        "hovertemplate": hovertemplate,
        "mode": mode,
        **kwargs,
    }
    traces = _merge_traces(traces, options)

    # Add traces
    fig.add_traces([go.Scatter(**trace) for trace in traces])

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )

    return fig
