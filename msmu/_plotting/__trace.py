import pandas as pd


class Trace:
    def __init__(
        self,
        data: pd.DataFrame,
        x: str | None = None,
        y: str | None = None,
        name: str | None = None,
        meta: str | None = None,
        text: str | None = None,
    ):
        self.data = data.copy()
        self.x = x
        self.y = y
        self.name = name
        self.meta = meta
        self.text = text

        self.data["_idx"] = self.data.index
        self.traces = self._get_traces()

    def __call__(self):
        return self.traces

    def __repr__(self):
        return f"Trace(x={self.x}, y={self.y}, name={self.name}, meta={self.meta})"

    def _get_traces(self):
        grouped = self.data.groupby(self.name, observed=True)
        return [
            {
                "x": group[self.x].values.tolist(),
                "y": group[self.y].values.tolist(),
                "name": name,
                "meta": group[self.meta].values.tolist() if self.meta is not None else name,
                "text": group[self.text].values.tolist() if self.text is not None else None,
            }
            for name, group in grouped
        ]

    def merge_trace_options(
        self,
        **kwargs,
    ) -> list[dict]:
        self.traces = [{**trace, **kwargs} for trace in self.traces]
        return self.traces


class TraceDescribed(Trace):
    def __init__(
        self,
        data: pd.DataFrame,
    ):
        super().__init__(data)

    def _get_traces(self):
        return [
            {
                "x": [idx],
                "lowerfence": [row["min"]],
                "q1": [row["25%"]],
                "median": [row["50%"]],
                "q3": [row["75%"]],
                "upperfence": [row["max"]],
                "name": idx,
            }
            for idx, row in self.data.iterrows()
        ]


class TraceHeatmap(Trace):
    def __init__(
        self,
        data: pd.DataFrame,
    ):
        self.data = data.copy()
        self.traces = self._get_traces()

    def _get_traces(self):
        if len(self.data.index) < 20:
            texttemplate = "%{z:.2f}"
        else:
            texttemplate = None

        return [
            {
                "x": self.data.columns.tolist(),
                "y": self.data.index.tolist(),
                "z": self.data.values.tolist(),
                "zmin": -1,
                "zmax": 1,
                "hoverongaps": False,
                "texttemplate": texttemplate,
            }
        ]


class TracePie(Trace):
    def __init__(
        self,
        data: pd.DataFrame,
    ):
        self.data = data.copy()
        self.traces = self._get_traces()

    def _get_traces(self):
        return [
            {
                "labels": self.data.index.tolist(),
                "values": self.data.values.tolist(),
            }
        ]
