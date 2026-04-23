"""
Module defining various plot types using Plotly.
"""

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from ._trace import Trace, TraceHeatmap, TracePie


class PlotTypes:
    trace_builder_cls = Trace
    trace_type: type[Any] = go.Trace

    def __init__(
        self,
        data: pd.DataFrame,
        x: str | None = None,
        y: str | None = None,
        name: str | None = None,
        meta: str | None = None,
        text: str | None = None,
        hovertemplate: str | None = None,
    ) -> None:
        """
        Sets up trace options and defaults for a plot type.

        Parameters:
            data: Prepared plotting data.
            x: Column mapped to x-axis.
            y: Column mapped to y-axis.
            name: Column defining trace grouping.
            meta: Column supplying hover metadata.
            text: Column for text labels.
            hovertemplate: Optional Plotly hovertemplate.
        """
        # Initial setup
        self.data = data
        self.x = x
        self.y = y
        self.name = name
        self.meta = meta
        self.text = text

        self.base_options: dict[str, Any] = dict(hovertemplate=hovertemplate)

    def figure(self, **kwargs: Any) -> go.Figure:
        """
        Builds and returns a Plotly figure for this plot type.

        Parameters:
            **kwargs: Additional trace options.

        Returns:
            Completed Plotly figure.
        """
        self.fig = go.Figure()
        trace_options = {**self.base_options, **kwargs}
        self.fig.add_traces(self.build_traces(trace_options))
        self.fig.update_layout(**self.default_layout())

        return self.fig

    def build_traces(self, trace_options: dict[str, Any]) -> list[Any]:
        """Instantiate Plotly trace objects from the prepared trace specs."""
        traces = self.trace_builder_cls(
            data=self.data,
            x=self.x,
            y=self.y,
            name=self.name,
            meta=self.meta,
            text=self.text,
        )
        traces.merge_trace_options(**trace_options)
        return [self.trace_type(**trace) for trace in traces()]

    def default_layout(self) -> dict[str, Any]:
        """Return layout defaults for the concrete plot type."""
        return {}


class PlotBar(PlotTypes):
    """
    Plot type for bar charts.
    """

    trace_type = go.Bar


class PlotSimpleBox(PlotTypes):
    """
    Plot type for box plots. Simplified version using go.Box with pre-calculated metrics.
    """

    def build_traces(self, trace_options: dict[str, Any]) -> list[Any]:
        return [
            go.Box(
                x=[idx],
                q1=[row["25%"]],
                median=[row["50%"]],
                q3=[row["75%"]],
                lowerfence=[row["min"]],
                upperfence=[row["max"]],
                boxpoints=False,
                name=idx,
            )
            for idx, row in self.data.iterrows()
        ]


class PlotBox(PlotTypes):
    """
    Plot type for box plots.
    """

    trace_type = go.Box

    def default_layout(self) -> dict[str, Any]:
        return dict(xaxis=dict(showticklabels=False))


class PlotViolin(PlotTypes):
    """
    Plot type for violin plots.
    """

    trace_type = go.Violin

    def default_layout(self) -> dict[str, Any]:
        return dict(xaxis=dict(showticklabels=False))


class PlotHistogram(PlotTypes):
    """
    Plot type for histogram plots.
    """

    trace_type = go.Bar


class PlotScatter(PlotTypes):
    """
    Plot type for scatter plots.
    """

    trace_type = go.Scatter


class PlotStackedBar(PlotTypes):
    """
    Plot type for stacked bar plots.
    """

    trace_type = go.Bar

    def default_layout(self) -> dict[str, Any]:
        return dict(legend=dict(traceorder="normal"), barmode="stack")


class PlotHeatmap(PlotTypes):
    """
    Plot type for heatmap plots.
    """

    trace_builder_cls = TraceHeatmap
    trace_type = go.Heatmap

    def build_traces(self, trace_options: dict[str, Any]) -> list[Any]:
        traces = TraceHeatmap(data=self.data)
        traces.merge_trace_options(**trace_options)
        return [self.trace_type(**trace) for trace in traces()]

    def default_layout(self) -> dict[str, Any]:
        return dict(yaxis=dict(autorange="reversed"))


class PlotPie(PlotTypes):
    """
    Plot type for pie charts.
    """

    trace_builder_cls = TracePie
    trace_type = go.Pie

    def build_traces(self, trace_options: dict[str, Any]) -> list[Any]:
        traces = TracePie(data=self.data)
        traces.merge_trace_options(**trace_options)
        return [self.trace_type(**trace) for trace in traces()]

    def figure(self, **kwargs: Any) -> go.Figure:
        fig = super().figure(**kwargs)
        self.fig.update_traces(hoverinfo="label+percent+name", textinfo="percent", textposition="inside")
        return fig
