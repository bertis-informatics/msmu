"""
Module defining various plot types using Plotly.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any

from ._trace import Trace, TraceHeatmap, TracePie


class PlotTypes:
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

        self.fig = go.Figure()
        self.ptype: type[go.Trace] = go.Trace
        self.options: dict[str, Any] = dict(hovertemplate=hovertemplate)
        self.layouts: dict[str, Any] = {}

    def figure(
        self,
        ptype: type[go.Trace] | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """
        Builds and returns a Plotly figure of the specified type.

        Parameters:
            ptype: Plotly trace constructor (e.g., `go.Bar`).
            **kwargs: Additional trace options.

        Returns:
            Completed Plotly figure.
        """
        self.ptype = ptype
        self.options.update(**kwargs)
        self.trace()
        self.layout(**self.layouts)

        return self.fig

    def trace(self) -> None:
        """
        Generates and adds traces to the figure using stored options.
        """
        traces = Trace(data=self.data, x=self.x, y=self.y, name=self.name, meta=self.meta, text=self.text)
        traces.merge_trace_options(**self.options)
        self.fig.add_traces([self.ptype(**trace) for trace in traces()])

    def layout(self, **kwargs: Any) -> None:
        """
        Applies layout updates to the figure.

        Parameters:
            **kwargs: Layout keyword arguments passed to Plotly.
        """
        self.fig.update_layout(**kwargs)


class PlotBar(PlotTypes):
    """
    Plot type for bar charts.
    """

    def figure(self, ptype=None, **kwargs) -> go.Figure:
        return super().figure(go.Bar, **kwargs)


class PlotSimpleBox(PlotTypes):
    """
    Plot type for box plots. Simplified version using go.Box with pre-calculated metrics.
    """

    def figure(self, ptype=None, **kwargs) -> go.Figure:
        self.trace()
        self.layout(**self.layouts)

        return self.fig

    def trace(self) -> None:
        self.fig.add_traces(
            [
                go.Box(
                    x=[idx],
                    # "min": [row["min"]],
                    lowerfence=[max(row["25%"] - 1.5 * (row["75%"] - row["25%"]), row["min"])],
                    q1=[row["25%"]],
                    median=[row["50%"]],
                    q3=[row["75%"]],
                    upperfence=[min(row["75%"] + 1.5 * (row["75%"] - row["25%"]), row["max"])],
                    # "max": [row["max"]],
                    name=idx,
                    boxpoints=False,
                    hoverinfo="y",
                )
                for idx, row in self.data.iterrows()
            ]
        )
        self.fig.add_traces(
            [
                go.Scatter(
                    x=[idx] * 2,
                    y=[row["max"], row["min"]],
                    mode="markers",
                    marker=dict(size=4),
                    name=idx,
                )
                for idx, row in self.data.iterrows()
            ]
        )


class PlotBox(PlotTypes):
    """
    Plot type for box plots.
    """

    def figure(self, ptype=go.Box, **kwargs) -> go.Figure:
        self.layouts.update(dict(xaxis=dict(showticklabels=False)))
        return super().figure(ptype, **kwargs)


class PlotViolin(PlotTypes):
    """
    Plot type for violin plots.
    """

    def figure(self, ptype=go.Violin, **kwargs) -> go.Figure:
        self.layouts.update(dict(xaxis=dict(showticklabels=False)))
        return super().figure(ptype, **kwargs)


class PlotHistogram(PlotTypes):
    """
    Plot type for histogram plots.
    """

    def figure(self, ptype=go.Bar, **kwargs) -> go.Figure:
        return super().figure(ptype, **kwargs)


class PlotScatter(PlotTypes):
    """
    Plot type for scatter plots.
    """

    def figure(self, ptype=go.Scatter, **kwargs) -> go.Figure:
        return super().figure(ptype, **kwargs)


class PlotStackedBar(PlotTypes):
    """
    Plot type for stacked bar plots.
    """

    def figure(self, ptype=go.Bar, **kwargs) -> go.Figure:
        self.layouts.update(dict(legend=dict(traceorder="normal"), barmode="stack"))
        return super().figure(ptype, **kwargs)


class PlotHeatmap(PlotTypes):
    """
    Plot type for heatmap plots.
    """

    def figure(self, ptype=go.Heatmap, **kwargs) -> go.Figure:
        self.layouts.update(dict(yaxis=dict(autorange="reversed")))
        return super().figure(ptype, **kwargs)

    def trace(self) -> None:
        traces = TraceHeatmap(data=self.data)
        traces.merge_trace_options(**self.options)
        self.fig.add_traces([self.ptype(**trace) for trace in traces()])


class PlotUpset(PlotTypes):
    def __init__(
        self,
        data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """
        Plot type for Upset diagrams.

        Parameters:
            data: Combination counts and item counts.
        """
        self.combination_counts, self.item_counts = data
        super().__init__(data)  # type: ignore

        self.fig = make_subplots(
            rows=2,
            cols=2,
            row_heights=[0.2, 0.8],
            column_widths=[0.2, 0.8],
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0,
            horizontal_spacing=0,
        )

    def figure(
        self,
        ptype: type[go.Trace] | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        self.trace()
        self.layout(**self.layouts)

        return self.fig

    def trace(self) -> None:
        self.fig.add_trace(
            go.Bar(
                x=self.combination_counts["combination"].tolist(),
                y=self.combination_counts["count"].tolist(),
                text=self.combination_counts["count"].tolist(),
                textposition="auto",
                texttemplate="%{text:,d}",
                name="combination",
                showlegend=False,
                hovertemplate=f"Sets: %{{x}}<br>Count: %{{y:,d}}<extra></extra>",
                marker=dict(color="#1f77b4"),
            ),
            row=1,
            col=2,
        )

        sets = self.item_counts.index

        # Add dots for each set in the combination
        for i, row in self.combination_counts.iterrows():
            combination = row["combination"]
            for j, set_name in enumerate(sets):
                self.fig.add_trace(
                    go.Scatter(
                        x=[f"{combination}"],
                        y=[set_name],
                        mode="markers",
                        marker=dict(
                            color="#444444" if combination[j] == "1" else "white",
                            size=10,
                            line=dict(color="#111111", width=2),
                        ),
                        showlegend=False,
                        hovertemplate=f"Sample: %{{y}}<extra></extra>",
                    ),
                    row=2,
                    col=2,
                )

        self.fig.add_trace(
            go.Bar(
                x=self.item_counts.values.tolist(),
                y=self.item_counts.index.tolist(),
                text=self.item_counts.values.tolist(),
                textposition="auto",
                texttemplate="%{text:,d}",
                orientation="h",
                showlegend=False,
                hovertemplate=f"Sample: %{{y}}<br>Count: %{{x:,d}}<extra></extra>",
                marker=dict(color="#1f77b4"),
            ),
            row=2,
            col=1,
        )

    def layout(self, **kwargs: bool) -> None:
        self.fig.update_xaxes(autorange="reversed", tickformat=",d", row=2, col=1)  # type: ignore
        self.fig.update_xaxes(ticklen=0, showticklabels=False, row=1, col=2)  # type: ignore
        self.fig.update_xaxes(ticklen=0, showticklabels=False, row=2, col=2)  # type: ignore

        self.fig.update_yaxes(autorange="reversed", showticklabels=False, ticklen=0, side="right", row=2, col=1)  # type: ignore
        self.fig.update_yaxes(side="right", tickformat=",d", showticklabels=True, row=1, col=2)  # type: ignore
        self.fig.update_yaxes(side="right", showticklabels=True, row=2, col=2)  # type: ignore

        # self.fig.update_yaxes(yaxis_tickformat=",d", row=1, col=1)
        # self.fig.update_layout(yaxis_tickformat=",d", row=2, col=1)

        self.fig.update_layout(**kwargs)


class PlotPie(PlotTypes):
    """
    Plot type for pie charts.
    """

    def figure(self, ptype=go.Pie, **kwargs) -> go.Figure:
        return super().figure(ptype, **kwargs)

    def trace(self) -> None:
        traces = TracePie(data=self.data)
        self.fig.add_traces([self.ptype(**trace) for trace in traces()])
        self.fig.update_traces(hoverinfo="label+percent+name", textinfo="percent", textposition="inside")
