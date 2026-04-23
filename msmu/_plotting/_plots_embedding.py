"""Embedding-oriented plotting functions."""

import mudata as md
import pandas as pd
import plotly.graph_objects as go

from ._pdata import PlotData
from ._ptypes import PlotScatter
from ._template import DEFAULT_TEMPLATE
from ._utils import PlotContext, finalize_figure, get_pc_cols, get_umap_cols


def _build_embedding_plot(
    *,
    context: PlotContext,
    coordinates: pd.DataFrame,
    x: str,
    y: str,
    title_text: str,
    xaxis_title: str,
    yaxis_title: str,
    **kwargs: str,
) -> go.Figure:
    """Render a shared scatter plot for embedding visualizations."""
    if context.groupby is None:
        raise ValueError("Embedding plots require a grouping column.")

    hovertemplate = f"<b>%{{meta}}</b><br>{xaxis_title}: %{{x}}<br>{yaxis_title}: %{{y}}<extra></extra>"
    plot = PlotScatter(
        data=coordinates,
        x=x,
        y=y,
        name=context.groupby,
        meta=context.obs_column,
        hovertemplate=hovertemplate,
    )
    fig = plot.figure(mode="markers", marker=dict(size=10))
    fig.update_yaxes(  # type: ignore
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend=dict(
            title=f"{context.groupby.capitalize()}",
            orientation="h",
            xanchor="right",
            yanchor="bottom",
            x=1,
            y=1,
        ),
    )

    return finalize_figure(fig, context=context, layout_kwargs=kwargs, apply_color=True)


def plot_pca(
    mdata: md.MuData,
    modality: str = "protein",
    groupby: str | None = None,
    colorby: str | None = None,
    template: str = DEFAULT_TEMPLATE,
    pcs: tuple[int, int] | list[int] = (1, 2),
    obs_column: str | None = None,
    key: str = "X_pca",
    **kwargs: str,
) -> go.Figure:
    """Plot PCA scores for a modality."""
    context = PlotContext.grouped(
        mdata,
        modality,
        groupby=groupby,
        obs_column=obs_column,
        colorby=colorby,
        template=template,
    )
    if context.groupby is None:
        raise ValueError("plot_pca requires a grouping column.")

    pcs, pc_columns = get_pc_cols(mdata, modality, pcs, key=key)
    if key not in mdata.mod[modality].uns:
        raise ValueError(f"Key {key} not found in .uns at {modality}")
    variances = mdata.mod[modality].uns[key]["variance_ratio"]

    data = PlotData(context.mdata, context.modality, layer=context.layer, obs_column=context.obs_column)
    coordinates = data.prep_embedding_scatter(
        modality,
        context.groupby,
        pc_columns,
        context.obs_column,
        key=key,
    )
    return _build_embedding_plot(
        context=context,
        coordinates=coordinates,
        x=pc_columns[0],
        y=pc_columns[1],
        title_text="PCA",
        xaxis_title=f"{pc_columns[0]} ({variances[pcs[0] - 1] * 100:.2f}%)",
        yaxis_title=f"{pc_columns[1]} ({variances[pcs[1] - 1] * 100:.2f}%)",
        **kwargs,
    )


def plot_umap(
    mdata: md.MuData,
    modality: str = "protein",
    groupby: str | None = None,
    colorby: str | None = None,
    template: str = DEFAULT_TEMPLATE,
    obs_column: str | None = None,
    key: str = "X_umap",
    **kwargs: str,
) -> go.Figure:
    """Plot UMAP embeddings for a modality."""
    context = PlotContext.grouped(
        mdata,
        modality,
        groupby=groupby,
        obs_column=obs_column,
        colorby=colorby,
        template=template,
    )
    if context.groupby is None:
        raise ValueError("plot_umap requires a grouping column.")

    umap_columns = get_umap_cols(mdata, modality, key=key)
    data = PlotData(context.mdata, context.modality, layer=context.layer, obs_column=context.obs_column)
    coordinates = data.prep_embedding_scatter(
        modality,
        context.groupby,
        umap_columns,
        context.obs_column,
        key=key,
    )
    return _build_embedding_plot(
        context=context,
        coordinates=coordinates,
        x=umap_columns[0],
        y=umap_columns[1],
        title_text="UMAP",
        xaxis_title=umap_columns[0],
        yaxis_title=umap_columns[1],
        **kwargs,
    )
