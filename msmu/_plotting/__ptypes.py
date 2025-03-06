import pandas as pd
import plotly.graph_objects as go

from .__trace import Trace, TraceDescribed


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
