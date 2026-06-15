from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..logging_utils import get_logger

logger = get_logger(__name__)


class DeaValidator:
    def __init__(self, ctrl_arr, expr_arr, min_pct) -> None:
        self.min_pct = min_pct
        self.ctrl_arr = ctrl_arr
        self.expr_arr = expr_arr

        self._min_sample_size_availability: bool = True
        self._sufficient_feature_indices: np.ndarray = np.array([])

        self.validate_inputs()
        self.validate_sample_size()
        self.get_sufficient_feature_indices()

    def validate_inputs(self) -> None:
        if not isinstance(self.ctrl_arr, np.ndarray) or not isinstance(self.expr_arr, np.ndarray):
            logger.error("Control and experimental arrays must be numpy arrays.")
            raise TypeError("Control and experimental arrays must be numpy arrays.")
        if self.ctrl_arr.shape[1] == 0 or self.expr_arr.shape[1] == 0:
            logger.error("Control and experimental arrays must have at least one sample (column).")
            raise ValueError("Control and experimental arrays must have at least one sample (column).")

    def validate_sample_size(self) -> None:
        if self.ctrl_arr.shape[1] < 2 or self.expr_arr.shape[1] < 2:
            logger.debug("Control and experimental arrays have fewer than two samples in at least one group.")
            self._min_sample_size_availability = False

    def get_sufficient_feature_indices(self) -> None:
        ctrl_sample_cutoff = self.ctrl_arr.shape[0] * self.min_pct
        expr_sample_cutoff = self.expr_arr.shape[0] * self.min_pct
        sufficient_ctrl_indices = np.sum(~np.isnan(self.ctrl_arr), axis=0) >= ctrl_sample_cutoff
        sufficient_expr_indices = np.sum(~np.isnan(self.expr_arr), axis=0) >= expr_sample_cutoff

        sufficient_indices = sufficient_ctrl_indices & sufficient_expr_indices

        self._sufficient_feature_indices = sufficient_indices

    @property
    def min_sample_size_availability(self) -> bool:
        return self._min_sample_size_availability

    @property
    def sufficient_feature_indices(self) -> np.ndarray:
        return self._sufficient_feature_indices


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
    p_value: np.ndarray | None = None
    q_value: np.ndarray | None = None


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


@dataclass
class DeaResult:
    stat_method: str
    ctrl: str | None
    expr: str | None = None
    features: pd.Index | np.ndarray | None = None
    repr_ctrl: np.ndarray | None = None
    repr_expr: np.ndarray | None = None
    pct_ctrl: np.ndarray | None = None
    pct_expr: np.ndarray | None = None
    log2fc: np.ndarray | None = None

    def __init__(self, test_result: PermTestResult | StatTestResult) -> None:
        for field in test_result.__dataclass_fields__:
            setattr(self, field, getattr(test_result, field))

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "features": self.features,
                "repr_ctrl": self.repr_ctrl,
                "repr_expr": self.repr_expr,
                "pct_ctrl": self.pct_ctrl,
                "pct_expr": self.pct_expr,
                "log2fc": self.log2fc,
                "p_value": self.p_value,
                "q_value": self.q_value,
            }
        )

    def plot_volcano(
        self,
        log2fc_threshold: float | None = None,
        pval_threshold: float = 0.05,
        label_top: int | None = None,
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

        if log2fc_threshold is None:
            if self.fc_pct_5:
                log2fc_threshold = self.fc_pct_5
            else:
                logger.error("log2fc_threshold not provided. set log2fc_threshold or run permutation test")
                raise

        from .. import pl

        return pl.plot_volcano(
            self.to_df(),
            ctrl=self.ctrl,
            expr=self.expr,
            log2fc_threshold=log2fc_threshold,
            pval_threshold=pval_threshold,
            label_top=label_top,
        )
