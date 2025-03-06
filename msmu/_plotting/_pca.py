import mudata as md
import plotly.graph_objects as go

from .__pdata import PlotData
from .__ptypes import PlotScatter
from ._template import DEFAULT_TEMPLATE
from ._utils import _set_color


DEFAULT_COLUMN = "_obs_"


def plot_pca(
    mdata: md.MuData,
    modality: str = "peptide",
    groupby: str = DEFAULT_COLUMN,
    colorby: str = None,
    template: str = DEFAULT_TEMPLATE,
    pcs: tuple[int, int] | list[int] = (1, 2),
    **kwargs,
) -> go.Figure:
    # Get required data
    pcs, pc_columns = _get_pc_cols(mdata, modality, pcs)
    variances = mdata[modality].uns["pca"]["variance_ratio"]

    # Set titles
    title_text = "PCA"
    xaxis_title = f"{pc_columns[0]} ({variances[pcs[0] - 1] * 100:.2f}%)"
    yaxis_title = f"{pc_columns[1]} ({variances[pcs[1] - 1] * 100:.2f}%)"
    hovertemplate = "<b>%{meta}</b><br>" + xaxis_title + ": %{x}<br>" + yaxis_title + ": %{y}<extra></extra>"

    # Draw plot
    data = PlotData(mdata, mods=[modality])
    plot = PlotScatter(
        data=data._prep_pca_data(modality, pc_columns),
        x=pc_columns[0],
        y=pc_columns[1],
        name=groupby,
        hovertemplate=hovertemplate,
    )
    fig = plot.figure(mode="markers")

    # Update axis
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    # Update layout
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )

    # Update layout with kwargs
    fig.update_layout(
        **kwargs,
    )

    # Set color
    if (colorby is not None) & (groupby == DEFAULT_COLUMN):
        fig = _set_color(fig, mdata, modality, colorby, template)

    return fig


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
