"""
Module for plotting utilities and functions in msmu.
"""

from ._plots import (
    plot_correlation,
    plot_id,
    plot_intensity,
    plot_missingness,
    plot_pca,
    plot_umap,
    plot_upset,
    plot_var,
)
from ._template import set_templates

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
