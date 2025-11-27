from dataclasses import dataclass
from typing import Literal
import logging

import numpy as np
import pandas as pd

from .._plotting._ptypes import PlotScatter
import plotly.graph_objects as go


logger = logging.getLogger(__name__)


class Dea:
    """
    Class to perform Differential Expression Analysis (DEA) using permutation tests.
    This class is used to compare two groups of data (control and experimental) and
    calculate statistics such as median differences, fold changes, and p-values.
    """

    def __init__(self):
        self._de_available: bool = True

    def validate_inputs(
        self,
        ctrl_arr: np.ndarray,
        expr_arr: np.ndarray,
    ) -> None:
        if not isinstance(ctrl_arr, np.ndarray) or not isinstance(expr_arr, np.ndarray):
            logger.error("Control and experimental arrays must be numpy arrays.")
            raise
        if ctrl_arr.shape[1] == 0 or expr_arr.shape[1] == 0:
            logger.error("Control and experimental arrays must have at least one sample (column).")
            raise
        if ctrl_arr.shape[0] < 2 or expr_arr.shape[0] < 2:
            logger.warning("Control and experimental arrays must have at least two samples each.")
            logger.warning(
                "Any statistics will not be performed. Results will only contain Fold Changes and Pct Expressions."
            )

            self.de_available = False

    def get_insufficient_feature_indices(self): ...

    @property
    def de_available(self) -> bool:
        """
        Check if DEA is available based on the number of samples in control and experimental groups.
        Returns True if DEA is available, False otherwise.
        """
        return self._de_available

    @de_available.setter
    def de_available(self, value: bool) -> None:
        """
        Set the availability of DEA.
        Parameters:
            value: True if DEA is available, False otherwise.
        """
        if not isinstance(value, bool):
            raise TypeError("de_available must be a boolean value.")
        self._de_available = value


@dataclass
class StatTestResult:
    """
    Data class to store results from statistical tests in DEA.

    Attributes:
        stat_method: The statistical method used.
        ctrl: Name of the control group.
        expr: Name of the experimental group.
        features: List or array of feature names.
        median_ctrl: Median values for the control group.
        median_expr: Median values for the experimental group.
        pct_ctrl: Percentage of non-zero values in the control group.
        pct_expr: Percentage of non-zero values in the experimental group.
        log2fc: Log2 fold change between experimental and control groups.
        p_value: P-values from the statistical test.
        q_value: Adjusted p-values (q-values) after multiple testing correction.

    Methods:
        to_df: Convert the results to a pandas DataFrame.
        plot_volcano: Plot a volcano plot of the DEA results.
    """

    stat_method: str
    ctrl: str | None
    expr: str | None = None
    features: pd.Index | np.ndarray | None = None
    median_ctrl: np.ndarray | None = None
    median_expr: np.ndarray | None = None
    pct_ctrl: np.ndarray | None = None
    pct_expr: np.ndarray | None = None
    log2fc: np.ndarray | None = None
    p_value: np.ndarray | None = None
    q_value: np.ndarray | None = None

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "features": self.features,
                "median_ctrl": self.median_ctrl,
                "median_expr": self.median_expr,
                "pct_ctrl": self.pct_ctrl,
                "pct_expr": self.pct_expr,
                "log2fc": self.log2fc,
                "p_value": self.p_value,
                "q_value": self.q_value,
            }
        )

    def plot_volcano(
        self, log2fc_threshold: float | None = None, pval_threshold: float = 0.05, label_top: int | None = None
    ) -> go.Figure:
        """
        Plots a volcano plot for the DEA results.

        Parameters:
            log2fc_threshold: Log2 fold change threshold for significance. If None, uses the 5th percentile fold change from permutation test results.
            pval_threshold: p-value threshold for significance.
            label_top: Number of top significant features to label on the plot. If None, no labels are added.

        Returns:
            Plotly Figure object containing the volcano plot.
        """

        df = self.to_df().copy()

        if log2fc_threshold == None:
            if self.fc_pct_5:
                log2fc_threshold = self.fc_pct_5
            else:
                logger.error("log2fc_threshold not provided. set log2fc_threshold or run permutation test")
                raise

        df["logp"] = -np.log10(df["p_value"])
        up_cond = df["log2fc"] > log2fc_threshold
        down_cond = df["log2fc"] < -log2fc_threshold
        sig_cond = df["p_value"] < pval_threshold

        df.loc[:, "de"] = "nonDE"
        df.loc[up_cond & sig_cond, "de"] = "UP"
        df.loc[down_cond & sig_cond, "de"] = "DOWN"

        up_count = len(df.loc[df["de"] == "UP",])
        down_count = len(df.loc[df["de"] == "DOWN",])

        p = PlotScatter(
            data=df,
            x="log2fc",
            y="logp",
            name="de",
            meta="features",
            text="p_value",
            hovertemplate="<b>%{meta}</b><br>Log<sub>2</sub>FC: %{x}<br>p-value: %{text}",
        )

        f = p.figure(mode="markers")

        f.update_xaxes(title="log<sub>2</sub>FC")
        f.update_yaxes(title="-log<sub>10</sub>p")

        f.update_traces(marker=dict(color="#E15759"), selector=dict(name="UP"))
        f.update_traces(marker=dict(color="#4E79A7"), selector=dict(name="DOWN"))
        f.update_traces(marker=dict(color="#BAB0AC"), selector=dict(name="nonDE"))

        f.update_traces(
            marker=dict(
                size=4,
            )
        )

        f.update_layout(
            title=f"{self.ctrl} vs. {self.expr}",
            width=600,
            height=500,
        )

        f.add_hline(
            y=-np.log10(pval_threshold),
            line=dict(color="grey", dash="dot", width=1),
        )
        f.add_vline(
            x=log2fc_threshold,
            line=dict(color="grey", dash="dot", width=1),
        )
        f.add_vline(
            x=-log2fc_threshold,
            line=dict(color="grey", dash="dot", width=1),
        )

        f.add_annotation(
            x=float(df["log2fc"].min()),
            y=float(df["logp"].min()),
            text=f"{self.ctrl} ({down_count})",
            showarrow=False,
        )
        f.add_annotation(
            x=float(df["log2fc"].max()), y=float(df["logp"].min()), text=f"{self.expr} ({up_count})", showarrow=False
        )

        if label_top is not None:
            up_top = df.loc[(df["de"] == "UP"), :].sort_values("log2fc").tail(label_top)
            down_top = df.loc[(df["de"] == "DOWN"), :].sort_values("log2fc").head(label_top)

            concated_tops = pd.concat([up_top, down_top])

            for _, row in concated_tops.iterrows():
                f.add_annotation(x=row["log2fc"], y=row["logp"], text=row["features"], arrowhead=0, arrowwidth=1)

        return f


@dataclass
class PermTestResult(StatTestResult):
    """
    Data class to store results from permutation tests in DEA.

    Attributes:
        Inherits all attributes from StatTestResult.
        permutation_method: The permutation method used ("exact" or "randomised").
        n_permutations: Number of permutations performed.
        fc_pct_1: Fold change at the 1st percentile.
        fc_pct_5: Fold change at the 5th percentile.
    """

    permutation_method: Literal["exact", "randomised"] | None = None
    n_permutations: int | None = None
    fc_pct_1: float | None = None
    fc_pct_5: float | None = None
