import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .__trace import *


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
    ):
        # Initial setup
        self.data = data
        self.x = x
        self.y = y
        self.name = name
        self.meta = meta
        self.text = text

        self.fig = go.Figure()
        self.ptype = None
        self.options = dict(hovertemplate=hovertemplate)
        self.layouts = {}

    def figure(self, ptype, **kwargs):
        self.ptype = ptype
        self.options.update(**kwargs)
        self.trace()
        self.layout(**self.layouts)

        return self.fig

    def trace(self):
        traces = Trace(data=self.data, x=self.x, y=self.y, name=self.name, meta=self.meta, text=self.text)
        traces.merge_trace_options(**self.options)
        self.fig.add_traces([self.ptype(**trace) for trace in traces()])

    def layout(self, **kwargs):
        self.fig.update_layout(**kwargs)


class PlotBar(PlotTypes):
    def figure(self, **kwargs):
        return super().figure(go.Bar, **kwargs)


class PlotBox(PlotTypes):
    def figure(self, **kwargs):
        return super().figure(go.Box, **kwargs)

    def trace(self):
        traces = TraceDescribed(data=self.data)
        self.fig.add_traces([self.ptype(**trace) for trace in traces()])
        self.fig.update_traces(boxpoints=False, orientation="h", hoverinfo="x", showlegend=False)


class PlotHistogram(PlotTypes):
    def figure(self, **kwargs):
        return super().figure(go.Bar, **kwargs)


class PlotScatter(PlotTypes):
    def figure(self, **kwargs):
        return super().figure(go.Scatter, **kwargs)


class PlotStackedBar(PlotTypes):
    def figure(self, **kwargs):
        self.layouts.update(dict(legend=dict(traceorder="normal"), barmode="stack"))
        return super().figure(go.Bar, **kwargs)


class PlotHeatmap(PlotTypes):
    def __init__(
        self,
        data: pd.DataFrame,
    ):
        super().__init__(data)

    def figure(self, **kwargs):
        self.layouts.update(dict(yaxis=dict(autorange="reversed"))),
        return super().figure(go.Heatmap, **kwargs)

    def trace(self):
        traces = TraceHistogram(data=self.data)
        self.fig.add_traces([self.ptype(**trace) for trace in traces()])


class PlotUpset(PlotTypes):
    def __init__(
        self,
        data: tuple[pd.DataFrame, pd.DataFrame],
    ):
        self.combination_counts, self.item_counts = data
        super().__init__(data)

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

    def figure(self):
        self.trace()
        self.layout(**self.layouts)

        return self.fig

    def trace(self):
        self.fig.add_trace(
            go.Bar(
                x=self.combination_counts["combination"],
                y=self.combination_counts["count"],
                text=self.combination_counts["count"],
                textposition="auto",
                name="combination",
                showlegend=False,
                hovertemplate=f"Sets: %{{x}}<br>Count: %{{y}}<extra></extra>",
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
                        x=[combination],
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
                x=self.item_counts,
                y=self.item_counts.index,
                text=self.item_counts,
                textposition="auto",
                orientation="h",
                showlegend=False,
                hovertemplate=f"Sample: %{{y}}<br>Count: %{{x}}<extra></extra>",
                marker=dict(color="#1f77b4"),
            ),
            row=2,
            col=1,
        )

    def layout(self, **kwargs):
        self.fig.update_xaxes(autorange="reversed", row=2, col=1)
        self.fig.update_xaxes(ticklen=0, showticklabels=False, row=1, col=2)
        self.fig.update_xaxes(ticklen=0, showticklabels=False, row=2, col=2)

        self.fig.update_yaxes(autorange="reversed", showticklabels=False, ticklen=0, side="right", row=2, col=1)
        self.fig.update_yaxes(side="right", showticklabels=True, row=1, col=2)
        self.fig.update_yaxes(side="right", showticklabels=True, row=2, col=2)

        self.fig.update_layout(**kwargs)
