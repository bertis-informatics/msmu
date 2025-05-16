import numpy as np
import pandas as pd
import mudata as md
import plotly.io as pio
import plotly.graph_objects as go


DEFAULT_COLUMN = "sample"


def _set_color(
    fig: go.Figure,
    mdata: md.MuData,
    mods: list[str],
    colorby: str,
    template: str = None,
):
    # Get categories
    categories = pd.concat([mdata[modality].obs[colorby] for modality in mods])

    # Get colors
    colors = pio.templates[template].layout["colorway"]

    colormap_dict = {val: colors[i % len(colors)] for i, val in enumerate(categories.unique())}
    colormap = categories.map(colormap_dict)

    # Update figure
    for i, trace in enumerate(fig.data):
        if hasattr(trace, "marker"):
            trace.marker.color = colormap[trace.name]
        if hasattr(trace, "line"):
            trace.line.color = colormap[trace.name]

    order_dict = {value: index for index, value in enumerate(mdata.obs[colorby].unique())}
    fig.data = tuple(
        sorted(
            fig.data,
            key=lambda trace: order_dict.get(
                mdata.obs.loc[mdata.obs[DEFAULT_COLUMN] == trace.name][colorby].values[0], float("inf")
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
