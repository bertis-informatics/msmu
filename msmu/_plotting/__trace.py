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
                "x": group[self.x].values,
                "y": group[self.y].values,
                "name": name,
                "meta": group[self.meta].values if self.meta is not None else name,
                "text": group[self.text].values if self.text is not None else None,
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
                "y": [idx],
                "lowerfence": [row["min"]],
                "q1": [row["25%"]],
                "median": [row["50%"]],
                "q3": [row["75%"]],
                "upperfence": [row["max"]],
            }
            for idx, row in self.data.iterrows()
        ]


class TraceHistogram(Trace):
    def __init__(
        self,
        data: pd.DataFrame,
    ):
        self.data = data.copy()
        self.traces = self._get_traces()

    def _get_traces(self):
        return [
            {
                "x": self.data.columns,
                "y": self.data.index,
                "z": self.data.values,
            }
        ]
