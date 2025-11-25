"""
Module defining trace classes for Plotly visualizations.
"""

import pandas as pd
from typing import Any


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
        """
        Builds Plotly-compatible trace dictionaries from a DataFrame.

        Parameters:
            data (pd.DataFrame): DataFrame containing plotting columns.
            x (str | None): Column mapped to x-axis.
            y (str | None): Column mapped to y-axis.
            name (str | None): Column used to split traces.
            meta (str | None): Column used for hover metadata.
            text (str | None): Column used for text labels.
        """
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
        """
        Groups the data and constructs trace dictionaries.

        Returns:
            list[dict]: Plotly trace keyword dictionaries.
        """
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
        **kwargs: Any,
    ) -> list[dict]:
        """
        Merges shared trace options into all trace dictionaries.

        Parameters:
            **kwargs: Arbitrary Plotly trace options to merge.

        Returns:
            list[dict]: Updated trace dictionaries.
        """
        self.traces = [{**trace, **kwargs} for trace in self.traces]
        return self.traces


class TraceDescribed(Trace):
    def __init__(
        self,
        data: pd.DataFrame,
    ):
        """
        Base class for traces built from descriptive statistics.

        Parameters:
            data (pd.DataFrame): Summary statistics for traces.
        """
        super().__init__(data)

    def _get_traces(self):
        """
        Constructs box/whisker descriptive traces from summary statistics.

        Returns:
            list[dict]: Trace definitions for Plotly box plots.
        """
        return [
            {
                "x": [idx],
                # "min": [row["min"]],
                "lowerfence": [max(row["25%"] - 1.5 * (row["75%"] - row["25%"]), row["min"])],
                "q1": [row["25%"]],
                "median": [row["50%"]],
                "q3": [row["75%"]],
                "upperfence": [min(row["75%"] + 1.5 * (row["75%"] - row["25%"]), row["max"])],
                # "max": [row["max"]],
                "name": idx,
            }
            for idx, row in self.data.iterrows()
        ]


class TraceHeatmap(Trace):
    def __init__(
        self,
        data: pd.DataFrame,
    ):
        """
        Generates heatmap trace definitions from a DataFrame.

        Parameters:
            data (pd.DataFrame): DataFrame containing heatmap values.
        """
        self.data = data.copy()
        self.traces = self._get_traces()

    def _get_traces(self):
        """
        Generates a single heatmap trace with optional text labels.

        Returns:
            list[dict]: Heatmap trace definition.
        """
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
        """
        Builds pie chart trace definitions from categorical counts.

        Parameters:
            data (pd.DataFrame): DataFrame containing pie chart values.
        """
        self.data = data.copy()
        self.traces = self._get_traces()

    def _get_traces(self):
        """
        Builds pie chart trace definitions from categorical counts.

        Returns:
            list[dict]: Pie trace definition.
        """
        return [
            {
                "labels": self.data.index.tolist(),
                "values": self.data.values.tolist(),
            }
        ]
