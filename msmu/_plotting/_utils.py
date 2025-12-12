"""
Utility functions for plotting with MuData and Plotly.
"""

import mudata as md
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from typing import TypedDict

_FALLBACK_COLUMN = "__obs_idx__"
_DEFAULT_OBS_PRIORITY = ("sample", "filename", _FALLBACK_COLUMN)


class BinInfo(TypedDict):
    """
    Encapsulates histogram bin metadata.

    Attributes:
        width: Width of each bin.
        edges: Edges of the bins.
        centers: Centers of the bins.
        labels: Labels for each bin.
    """

    width: float
    edges: list[float]
    centers: list[float]
    labels: list[str]


def resolve_obs_column(
    mdata: md.MuData,
    requested: str | None = None,
) -> str:
    """
    Determines which observation column to use for grouping/plotting.

    Parameters:
        mdata: MuData object containing observation metadata.
        requested: Specific observation column to prioritize when available.

    Returns:
        Name of the categorical observation column added/resolved in `mdata.obs`.
    """
    # Allow MuData to specify a default preference via uns
    plotting_defaults = mdata.uns.get("plotting", {}) if hasattr(mdata, "uns") else {}
    preferred = plotting_defaults.get("default_obs_column")

    candidates: list[str] = []
    for name in (requested, preferred, *_DEFAULT_OBS_PRIORITY):
        if name and name not in candidates:
            candidates.append(name)

    for name in candidates:
        if name in mdata.obs.columns:
            return ensure_obs_categorical(mdata, name)
        elif (name == requested) or (name == preferred):
            print(f"[INFO] Requested obs column '{name}' not found in observations.")

    # Create a stable fallback using obs index
    fallback_name = requested or preferred or _FALLBACK_COLUMN
    print(f"[INFO] Using fallback obs column '{fallback_name}' created from index.")
    if fallback_name in mdata.obs.columns:
        return ensure_obs_categorical(mdata, fallback_name)

    fallback_values = pd.Index(mdata.obs.index).map(str)
    mdata.obs[fallback_name] = pd.Categorical(fallback_values, categories=pd.unique(fallback_values))
    return ensure_obs_categorical(mdata, fallback_name)


def resolve_plot_columns(
    mdata: md.MuData,
    groupby: str | None,
    obs_column: str | None,
) -> tuple[str, str]:
    """
    Resolves grouping and observation columns with sensible defaults.

    Parameters:
        mdata: MuData object holding observation metadata.
        groupby: Requested grouping column; defaults to `obs_column` when None.
        obs_column: Requested observation column; resolved via `resolve_obs_column`.

    Returns:
        Resolved `(groupby, obs_column)` pair.
    """
    resolved_obs = resolve_obs_column(mdata, obs_column)
    resolved_groupby = groupby or resolved_obs

    return resolved_groupby, resolved_obs


def ensure_obs_categorical(mdata: md.MuData, column: str) -> str:
    """
    Casts the observation column to a pandas categorical type if needed.

    Parameters:
        mdata: MuData object containing observation metadata.
        column: Observation column name to cast.

    Returns:
        The validated categorical column name.
    """
    if column not in mdata.obs.columns:
        raise KeyError(f"Column '{column}' not found in observations.")
    if not isinstance(mdata.obs[column].dtype, pd.CategoricalDtype):
        mdata.obs[column] = pd.Categorical(mdata.obs[column], categories=pd.unique(mdata.obs[column]))
    return column


def format_modality(mdata: md.MuData, modality: str) -> str:
    """
    Formats modality keys into human-readable labels.

    Parameters:
        mdata: MuData object containing modality metadata.
        modality: Modality key such as 'psm', 'peptide', or 'protein'.

    Returns:
        Display-ready modality label.
    """
    if modality == "psm":
        if mdata["psm"].uns["search_engine"] == "Diann":
            return "Precursor"
        else:
            return "PSM"
    elif modality == "peptide":
        return "Peptide"
    elif modality == "protein":
        return "Protein"
    elif modality.endswith("_site"):
        return modality.capitalize()
    else:
        raise ValueError(f"Unknown modality: {modality}, choose from 'psm', 'peptide', 'protein', '[ptm]_site'")


def set_color(
    fig: go.Figure,
    mdata: md.MuData,
    modality: str,
    colorby: str,
    groupby_column: str,
    template: str | None = None,
) -> go.Figure:
    """
    Applies consistent colors to traces based on a categorical observation column.

    Parameters:
        fig: Plotly figure whose traces will be recolored.
        mdata: MuData object providing observations and metadata.
        modality: Modality key for accessing the appropriate AnnData object.
        colorby: Observation column used to map groups to colors.
        groupby_column: Observation column used to group traces.
        template: Optional Plotly template name for extracting the colorway.

    Returns:
        Figure with trace colors and ordering updated.
    """
    groupby_column = resolve_obs_column(mdata, groupby_column)

    # Ensure color column exists and is categorical
    if colorby not in mdata.obs.columns:
        raise KeyError(f"Column '{colorby}' not found in observations.")
    color_series = mdata.obs[colorby].copy()

    if not isinstance(color_series.dtype, pd.CategoricalDtype):
        color_series = color_series.astype("category")
        mdata.obs[colorby] = color_series
    else:
        mdata.obs[colorby] = color_series.cat.remove_unused_categories()

    group_series = mdata.obs[groupby_column].copy()
    if not isinstance(group_series.dtype, pd.CategoricalDtype):
        group_series = group_series.astype("category")
        mdata.obs[groupby_column] = group_series
    else:
        mdata.obs[groupby_column] = group_series.cat.remove_unused_categories()

    # Get categories
    categories = color_series.cat.categories

    # Get colors
    template_key = template if template in pio.templates else pio.templates.default
    if isinstance(template_key, (list, tuple)):
        template_key = template_key[0]
    if template_key not in pio.templates:
        template_key = "plotly"
    colors = (
        pio.templates[template_key].layout["colorway"]
        if "colorway" in pio.templates[template_key].layout
        else pio.templates["plotly"].layout["colorway"]
    )

    colormap_dict = {val: colors[i % len(colors)] for i, val in enumerate(categories)}
    group_to_category: dict[str, str] = {}
    group_to_color: dict[str, str] = {}
    group_values = group_series.to_numpy(dtype=object)
    color_values = color_series.to_numpy(dtype=object)
    for group_value, category_value in zip(group_values, color_values):
        if pd.isna(group_value) or pd.isna(category_value):
            continue
        if group_value not in group_to_category:
            group_to_category[group_value] = category_value
            group_to_color[group_value] = colormap_dict[category_value]

    # Update figure
    for i, trace in enumerate(fig.data):
        trace_name = getattr(trace, "name", "")
        color_value = group_to_color.get(trace_name)
        if hasattr(trace, "marker"):
            trace.marker.color = color_value  # type: ignore
        if hasattr(trace, "line"):
            trace.line.color = color_value  # type: ignore

    order_dict = {value: index for index, value in enumerate(categories)}
    fig.data = tuple(
        sorted(
            fig.data,
            key=lambda trace: order_dict.get(
                group_to_category.get(getattr(trace, "name", "")),
                float("inf"),
            ),
        )
    )

    return fig


def merge_traces(
    traces: list[dict],
    options: dict,
) -> list[dict]:
    """
    Merges a list of trace dictionaries with a common set of options.

    Parameters:
        traces: Trace dictionaries to merge.
        options: Shared options applied to every trace.

    Returns:
        Updated trace definitions with merged options.
    """
    merged_traces = []
    for trace in traces:
        merged_traces.append({**trace, **options})

    return merged_traces


def get_bin_info(data: pd.DataFrame, bins: int) -> BinInfo:
    """
    Computes histogram bin metadata for the provided numeric data.

    Parameters:
        data: DataFrame or Series containing numeric values.
        bins: Number of bins to generate.

    Returns:
        BinInfo: Encapsulated bin width, edges, centers, and labels.
    """
    min_value = float(np.min(data))
    max_value = float(np.max(data))
    data_range = max_value - min_value
    bin_width = data_range / bins if bins > 0 else 0.0
    bin_edges = [min_value + bin_width * i for i in range(bins + 1)]
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(bins)]
    bin_labels = [f"{bin_edges[i]} - {bin_edges[i + 1]}" for i in range(bins)]

    return BinInfo(
        width=bin_width,
        edges=bin_edges,
        centers=bin_centers,
        labels=bin_labels,
    )


def apply_color_if_needed(
    fig: go.Figure,
    *,
    mdata: md.MuData,
    modality: str,
    groupby: str,
    colorby: str | None,
    obs_column: str,
    template: str,
) -> go.Figure:
    """
    Applies trace colors when grouping and coloring on the same observation column.

    Parameters:
        fig: Figure whose traces may be recolored.
        mdata: MuData object providing observation metadata.
        modality: Modality key for the AnnData object.
        groupby: Observation column used for grouping traces.
        colorby: Observation column used for color mapping.
        obs_column: Resolved observation column.
        template: Plotly template name for colorway selection.

    Returns:
        Plotly figure with color applied when applicable.
    """
    if (colorby is not None) and (groupby == obs_column):
        return set_color(fig, mdata, modality, colorby, obs_column, template)
    elif (colorby is not None) and (groupby != obs_column):
        print("[Warning] 'colorby' is only applicable when 'groupby' is not set. Ignoring 'colorby' parameter.")
    return fig


def apply_layout_overrides(fig: go.Figure, layout_kwargs: dict) -> go.Figure:
    """
    Applies optional layout keyword arguments to a figure.

    Parameters:
        fig: Figure to update.
        layout_kwargs: Layout options to apply.

    Returns:
        Updated figure.
    """
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig


def get_pc_cols(
    mdata: md.MuData,
    modality: str,
    pcs: tuple[int, int] | list[int],
) -> tuple[tuple[int, int], list[str]]:
    """
    Validates requested principal components and returns column names.

    Parameters:
        mdata: MuData object containing PCA results in `obsm["X_pca"]`.
        modality: Modality key for accessing the appropriate AnnData object.
        pcs: Pair of principal component indices (1-based).

    Returns:
        Validated PC indices and their column names.
    """
    # Check pcs length
    if len(pcs) != 2:
        raise ValueError("Only 2 PCs are allowed")

    # Check if pcs are integers
    if not all(isinstance(pc, int) for pc in pcs):
        raise ValueError("PCs must be integers")

    # Sort pcs
    pcs = (pcs[0], pcs[1])
    if pcs[0] == pcs[1]:
        raise ValueError("PCs must be different")
    elif pcs[0] > pcs[1]:
        pcs = (pcs[1], pcs[0])

    # Check if PCs exist
    if "X_pca" not in mdata[modality].obsm:
        raise ValueError(f"No PCA found in {modality}")

    # Get PC columns
    pc_columns = [f"PC_{pc}" for pc in pcs]

    if pc_columns[0] not in mdata[modality].obsm["X_pca"].columns:  # type: ignore
        raise ValueError(f"{pc_columns[0]} not found in {modality}")
    if pc_columns[1] not in mdata[modality].obsm["X_pca"].columns:  # type: ignore
        raise ValueError(f"{pc_columns[1]} not found in {modality}")

    return pcs, pc_columns


def get_umap_cols(
    mdata: md.MuData,
    modality: str,
) -> list[str]:
    """
    Validates UMAP embeddings and returns expected column names.

    Parameters:
        mdata: MuData object containing UMAP embeddings in `obsm["X_umap"]`.
        modality: Modality key for accessing the appropriate AnnData object.

    Returns:
        List of UMAP column names used for plotting.
    """
    # Check if UMAP exist
    if "X_umap" not in mdata[modality].obsm:
        raise ValueError(f"No UMAP found in {modality}")

    # Get UMAP columns
    umap_columns = [f"UMAP_{pc}" for pc in [1, 2]]

    if umap_columns[0] not in mdata[modality].obsm["X_umap"].columns:  # type: ignore
        raise ValueError(f"{umap_columns[0]} not found in {modality}")
    if umap_columns[1] not in mdata[modality].obsm["X_umap"].columns:  # type: ignore
        raise ValueError(f"{umap_columns[1]} not found in {modality}")

    return umap_columns
