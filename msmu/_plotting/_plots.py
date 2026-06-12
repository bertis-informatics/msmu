"""Backward-compatible plotting facade re-exporting family-specific plot modules."""

from ._plots_distribution import (
    plot_correlation,
    plot_intensity,
    plot_missingness,
    plot_var,
)
from ._plots_embedding import plot_pca, plot_umap
from ._plots_summary import plot_id, plot_upset

__all__ = [
    "plot_correlation",
    "plot_id",
    "plot_intensity",
    "plot_missingness",
    "plot_pca",
    "plot_umap",
    "plot_upset",
    "plot_var",
]
