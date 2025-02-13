import plotly.graph_objects as go


def _draw_histogram(
    traces: list[dict],
    title_text: str,
    xaxis_title: str,
    yaxis_title: str,
    binsize: float = 0.1,
) -> go.Figure:
    # Create figure
    fig = go.Figure()

    # Add traces
    hovertemplate = xaxis_title + ": %{x}<br>" + yaxis_title + ": %{y:,d}"
    for trace in traces:
        fig.add_trace(
            go.Histogram(
                **trace,
                xbins=dict(size=binsize),
                hovertemplate=hovertemplate,
            )
        )

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )

    return fig


def _draw_density(
    traces: list[dict],
    title_text: str,
    xaxis_title: str,
    bandwidth: float = 0.1,
) -> go.Figure:
    # Create figure
    fig = go.Figure()

    # Add traces
    for trace in traces:
        fig.add_trace(
            go.Violin(
                **trace,
                bandwidth=bandwidth,
                showlegend=False,
                width=1.8,
                points=False,
                hoveron="points",
                orientation="h",
                side="negative",
                spanmode="hard",
                box=dict(
                    visible=True,
                    fillcolor="white",
                    line=dict(width=1),
                ),
            )
        )

    # Update axes
    fig.update_yaxes(autorange="reversed")

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
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
    for trace in traces:
        fig.add_trace(
            go.Box(
                **trace,
                boxpoints=False,
                hoverinfo="x",
                orientation="h",
                showlegend=False,
            )
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

    # Add traces
    hovertemplate = xaxis_title + ": %{x}<br>" + yaxis_title + ": %{y:2,d}<extra></extra>"
    for trace in traces:
        fig.add_trace(
            go.Bar(
                **trace,
                hovertemplate=hovertemplate,
                texttemplate="%{y:,d}",
            )
        )

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
) -> go.Figure:
    # Create figure
    fig = go.Figure()

    # Add traces
    for trace in traces:
        # idx = trace.pop("idx")
        hovertemplate = "<b>%{text}</b><br>" + xaxis_title + ": %{x}<br>" + yaxis_title + ": %{y}<extra></extra>"
        fig.add_trace(
            go.Scatter(
                **trace,
                hovertemplate=hovertemplate,
                mode="markers",
            )
        )

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )

    return fig
