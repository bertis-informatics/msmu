import mudata as md
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pandas.api.types import is_categorical_dtype

_FALLBACK_COLUMN = "__obs_idx__"
_DEFAULT_OBS_PRIORITY = ("sample", "filename", _FALLBACK_COLUMN)


def resolve_obs_column(
    mdata: md.MuData,
    requested: str | None = None,
) -> str:
    """
    Determine a usable observation column for grouping/plotting.
    Falls back through a priority list and finally creates a categorical
    column from the obs index when nothing suitable exists.
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
            return _ensure_obs_categorical(mdata, name)
        elif (name == requested) or (name == preferred):
            print(f"[INFO] Requested obs column '{name}' not found in observations.")

    # Create a stable fallback using obs index
    fallback_name = requested or preferred or _FALLBACK_COLUMN
    print(f"[INFO] Using fallback obs column '{fallback_name}' created from index.")
    if fallback_name in mdata.obs.columns:
        return _ensure_obs_categorical(mdata, fallback_name)

    fallback_values = pd.Index(mdata.obs.index).map(str)
    mdata.obs[fallback_name] = pd.Categorical(fallback_values, categories=pd.unique(fallback_values))
    return _ensure_obs_categorical(mdata, fallback_name)


def _ensure_obs_categorical(mdata: md.MuData, column: str) -> str:
    """Cast the obs column to categorical in-place if needed."""
    if column not in mdata.obs.columns:
        raise KeyError(f"Column '{column}' not found in observations.")
    if not is_categorical_dtype(mdata.obs[column]):
        mdata.obs[column] = pd.Categorical(mdata.obs[column], categories=pd.unique(mdata.obs[column]))
    return column


def _set_color(
    fig: go.Figure,
    mdata: md.MuData,
    modality: str,
    colorby: str,
    groupby_column: str,
    template: str = None,
):
    groupby_column = resolve_obs_column(mdata, groupby_column)

    # Ensure color column exists and is categorical
    if colorby not in mdata.obs.columns:
        raise KeyError(f"Column '{colorby}' not found in observations.")
    color_series = mdata.obs[colorby].copy()

    if not is_categorical_dtype(color_series):
        color_series = color_series.astype("category")
        mdata.obs[colorby] = color_series
    else:
        mdata.obs[colorby] = color_series.cat.remove_unused_categories()

    group_series = mdata.obs[groupby_column].copy()
    if not is_categorical_dtype(group_series):
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
    for group_value, category_value in zip(group_series, color_series):
        if pd.isna(group_value) or pd.isna(category_value):
            continue
        if group_value not in group_to_category:
            group_to_category[group_value] = category_value
            group_to_color[group_value] = colormap_dict[category_value]

    # Update figure
    for i, trace in enumerate(fig.data):
        trace_name = getattr(trace, "name", None)
        color_value = group_to_color.get(trace_name)
        if hasattr(trace, "marker"):
            trace.marker.color = color_value
        if hasattr(trace, "line"):
            trace.line.color = color_value

    order_dict = {value: index for index, value in enumerate(categories)}
    fig.data = tuple(
        sorted(
            fig.data,
            key=lambda trace: order_dict.get(
                group_to_category.get(getattr(trace, "name", None)),
                float("inf"),
            ),
        )
    )

    return fig


def _merge_traces(
    traces: list[dict],
    options: dict,
) -> list[dict]:
    merged_traces = []
    for trace in traces:
        merged_traces.append({**trace, **options})

    return merged_traces


def _get_bin_info(data: pd.DataFrame, bins: int) -> dict:
    # get bin data
    min_value = np.min(data)
    max_value = np.max(data)
    data_range = max_value - min_value
    bin_width = data_range / bins
    bin_edges = [min_value + bin_width * i for i in range(bins + 1)]
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(bins)]
    bin_labels = [f"{bin_edges[i]} - {bin_edges[i + 1]}" for i in range(bins)]

    return {
        "width": bin_width,
        "edges": bin_edges,
        "centers": bin_centers,
        "labels": bin_labels,
    }


def _get_pc_cols(
    mdata: md.MuData,
    modality: str,
    pcs: tuple[int, int],
) -> list[str]:
    # Check pcs length
    if len(pcs) != 2:
        raise ValueError("Only 2 PCs are allowed")

    # Check if pcs are integers
    if not all(isinstance(pc, int) for pc in pcs):
        raise ValueError("PCs must be integers")

    # Sort pcs
    if pcs[0] == pcs[1]:
        pcs[1] += 1
    elif pcs[0] > pcs[1]:
        pcs = (pcs[1], pcs[0])

    # Check if PCs exist
    if "X_pca" not in mdata[modality].obsm:
        raise ValueError(f"No PCA found in {modality}")

    # Get PC columns
    pc_columns = [f"PC_{pc}" for pc in pcs]

    if pc_columns[0] not in mdata[modality].obsm["X_pca"].columns:
        raise ValueError(f"{pc_columns[0]} not found in {modality}")
    if pc_columns[1] not in mdata[modality].obsm["X_pca"].columns:
        raise ValueError(f"{pc_columns[1]} not found in {modality}")

    return pcs, pc_columns


def _get_umap_cols(
    mdata: md.MuData,
    modality: str,
) -> list[str]:
    # Check if UMAP exist
    if "X_umap" not in mdata[modality].obsm:
        raise ValueError(f"No UMAP found in {modality}")

    # Get UMAP columns
    umap_columns = [f"UMAP_{pc}" for pc in [1, 2]]

    if umap_columns[0] not in mdata[modality].obsm["X_umap"].columns:
        raise ValueError(f"{umap_columns[0]} not found in {modality}")
    if umap_columns[1] not in mdata[modality].obsm["X_umap"].columns:
        raise ValueError(f"{umap_columns[1]} not found in {modality}")

    return umap_columns
