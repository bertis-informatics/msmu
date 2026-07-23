from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..logging_utils import get_logger

logger = get_logger(__name__)


def sufficient_feature_mask(
    cell_blocks: list[np.ndarray],
    min_pct: float,
    require_residual_df: bool = False,
) -> np.ndarray:
    """Boolean mask over features to keep, by per-cell non-missing coverage (single min_pct rule).

    A feature is kept when every cell has at least ``max(1, min_pct * cell_size)`` non-missing
    observations. With ``require_residual_df`` the total non-missing across cells must also exceed
    the number of cells, so a per-feature linear fit has residual degrees of freedom >= 1 — needed
    by limma; the permutation path, which fits no model, leaves it off. Both DE engines route
    their min_pct filtering through this one function.

    Args:
        cell_blocks: one ``(n_features x n_cell_samples)`` array per design cell / group.
        min_pct: minimum non-missing fraction required in every cell.
        require_residual_df: also require total non-missing across cells > number of cells.

    Returns:
        Boolean array of length n_features, True for features to keep.
    """
    n_features = cell_blocks[0].shape[0]
    n_cells = len(cell_blocks)
    total_non_missing = np.zeros(n_features, dtype=float)
    keep = np.ones(n_features, dtype=bool)

    for cell_block in cell_blocks:
        non_missing = np.sum(~np.isnan(cell_block), axis=1)
        total_non_missing += non_missing
        required = max(1.0, min_pct * cell_block.shape[1])
        keep &= non_missing >= required

    if require_residual_df:
        keep &= total_non_missing > n_cells

    return keep


class DeaValidator:
    """Pre-flight feature/sample validation for one DE comparison, expressed over design cells.

    Given one ``(n_features x n_cell_samples)`` block per comparison cell, it produces the
    usable-feature mask (via the shared :func:`sufficient_feature_mask`) and whether every cell
    has enough samples to test. Both DE engines build their own cells and route through this one
    validator, so the single ``min_pct`` rule stays in one place:

    * the permutation path passes the two groups with ``require_residual_df=False`` (it fits no
      model), and
    * limma passes its design cells with ``require_residual_df=True`` (a per-feature linear fit
      needs at least one residual degree of freedom).
    """

    MIN_SAMPLES_PER_CELL = 2  # a two-sample comparison needs at least this many samples per cell

    def __init__(self, cell_blocks: list[np.ndarray], min_pct: float, require_residual_df: bool) -> None:
        self.feature_mask: np.ndarray = sufficient_feature_mask(cell_blocks, min_pct, require_residual_df)
        self.has_enough_samples: bool = all(
            cell.shape[1] >= self.MIN_SAMPLES_PER_CELL for cell in cell_blocks
        )


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
        statistic: Per-feature test statistic. Its meaning depends on stat_method:
            Welch/Student t, Wilcoxon rank-sum, or limma moderated t. For the
            permutation path it is the observed parametric statistic (the value
            ranked against the empirical null), while p_value stays empirical.
        p_value: P-values from the statistical test.
        q_value: Adjusted p-values (q-values) after multiple testing correction.

    Methods:
        to_df: Convert the results to a pandas DataFrame.
        plot_volcano: Plot a volcano plot of the DEA results.
    """

    stat_method: str
    statistic: np.ndarray | None = None
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
    contrast_label: str | None = None
    # Fold-change guidance-line thresholds. Declared here so the attribute always exists
    # regardless of the engine: the permutation path copies these from PermTestResult, while
    # limma / simple results have them filled in by run_de's engine-independent computation.
    fc_pct_1: float | None = None
    fc_pct_5: float | None = None

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
                "statistic": self.statistic,
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
            if self.fc_pct_5 is not None:
                log2fc_threshold = self.fc_pct_5
            else:
                message = (
                    "log2fc_threshold is None and no fold-change guidance line (fc_pct_5) is "
                    "available; pass log2fc_threshold explicitly."
                )
                logger.error(message)
                raise ValueError(message)

        from .. import pl

        return pl.plot_volcano(
            self.to_df(),
            ctrl=self.ctrl,
            expr=self.expr,
            log2fc_threshold=log2fc_threshold,
            pval_threshold=pval_threshold,
            label_top=label_top,
        )
