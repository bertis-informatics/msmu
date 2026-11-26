"""
Decoy traces that the QC plots draw next to the targets when called with `include_decoy=True`.

Decoys are grey (hatched in bar charts) and share the targets' groups: a bar segment on top of each
group's bar, a box or violin beside each group's one, a bar beside each stacked bar. Histograms and
the missingness curve pool the observations into one target and one decoy trace.
"""

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from ..logging_utils import get_logger
from ._pdata import PlotData
from ._template import DEFAULT_TEMPLATE
from ._utils import ensure_msmu_templates_registered, get_template_colorway

logger = get_logger(__name__)

TARGET_TRACE_NAME = "Target"
DECOY_TRACE_NAME = "Decoy"
DECOY_COLOR = "#9A9A9A"
DECOY_BAR_PATTERN = "/"
TARGET_OFFSET_GROUP = "target"
DECOY_OFFSET_GROUP = "decoy"
# Target and decoy histograms are overlaid on shared bins, so each stays visible through the other.
OVERLAID_HISTOGRAM_OPACITY = 0.75


def resolve_decoy_data(data: PlotData, include_decoy: bool) -> PlotData | None:
    """Return the plot data over the decoys when they are asked for, or `None` to plot targets only."""
    if not include_decoy:
        return None
    if not data.has_decoy_features():
        logger.warning("Modality '%s' has no decoy features; plotting targets only.", data.modality)
        return None
    return data.for_decoys()


def add_decoy_bar_segments(fig: go.Figure, decoy_counts: pd.DataFrame, *, x: str, y: str, hovertemplate: str) -> None:
    """Stack one grey, hatched bar segment of decoy counts on top of each group's target bar."""
    fig.add_bar(
        x=decoy_counts[x].tolist(),
        y=decoy_counts[y].tolist(),
        text=decoy_counts[y].tolist(),
        textposition="outside",
        name=DECOY_TRACE_NAME,
        marker=dict(color=DECOY_COLOR, pattern_shape=DECOY_BAR_PATTERN),
        hovertemplate=hovertemplate,
    )
    # Stacked bars list their legend bottom-up by default; keep the groups' order, the decoys last.
    fig.update_layout(barmode="stack", legend_traceorder="normal")


def add_decoy_simple_boxes(fig: go.Figure, decoy_statistics: pd.DataFrame) -> None:
    """Draw a grey box of precomputed decoy statistics beside each group's target box."""
    _place_targets_beside_decoys(fig)
    for box_index, (group_name, row) in enumerate(decoy_statistics.iterrows()):
        fig.add_box(
            x=[group_name],
            q1=[row["25%"]],
            median=[row["50%"]],
            q3=[row["75%"]],
            lowerfence=[row["min"]],
            upperfence=[row["max"]],
            boxpoints=False,
            name=DECOY_TRACE_NAME,
            legendgroup=DECOY_TRACE_NAME,
            showlegend=box_index == 0,
            marker_color=DECOY_COLOR,
            offsetgroup=DECOY_OFFSET_GROUP,
        )
    fig.update_layout(boxmode="group")


def add_decoy_distributions(
    fig: go.Figure,
    trace_type: type[go.Box] | type[go.Violin],
    decoy_values: pd.DataFrame,
    *,
    x: str,
    y: str,
    **trace_options: Any,
) -> None:
    """Draw a grey box or violin of decoy values beside each group's target one."""
    _place_targets_beside_decoys(fig)
    decoy_trace = trace_type(
        x=decoy_values[x].tolist(),
        y=decoy_values[y].tolist(),
        name=DECOY_TRACE_NAME,
        offsetgroup=DECOY_OFFSET_GROUP,
        **trace_options,
    )
    decoy_trace.update(marker_color=DECOY_COLOR, line_color=DECOY_COLOR)
    fig.add_trace(decoy_trace)
    fig.update_layout(boxmode="group", violinmode="group")


def add_decoy_stacked_bars(
    fig: go.Figure, decoy_counts: pd.DataFrame, *, x: str, y: str, name: str, template: str | None
) -> None:
    """Draw hatched decoy bars, stacked by the same categories, beside each group's target bar."""
    # The figure gets its template later (msmu unless given); the decoy bars need its colors now.
    ensure_msmu_templates_registered()
    colorway = get_template_colorway(template or DEFAULT_TEMPLATE)
    color_by_category: dict[str, str] = {}
    for trace_index, trace in enumerate(fig.data):
        color_by_category[trace.name] = colorway[trace_index % len(colorway)]
        trace.update(
            marker_color=color_by_category[trace.name], legendgroup=trace.name, offsetgroup=TARGET_OFFSET_GROUP
        )

    for category, category_counts in decoy_counts.groupby(name, observed=True):
        category = str(category)
        color = color_by_category.setdefault(category, colorway[len(color_by_category) % len(colorway)])
        fig.add_bar(
            x=category_counts[x].tolist(),
            y=category_counts[y].tolist(),
            name=category,
            meta=category_counts[name].tolist(),
            hovertemplate=fig.data[0].hovertemplate,
            marker=dict(color=color, pattern_shape=DECOY_BAR_PATTERN),
            legendgroup=DECOY_TRACE_NAME,
            showlegend=False,
            offsetgroup=DECOY_OFFSET_GROUP,
        )
    # The decoy bars carry their categories' colors; one legend entry names the hatching instead.
    fig.add_bar(
        x=[None],
        y=[None],
        name=DECOY_TRACE_NAME,
        marker=dict(color=DECOY_COLOR, pattern_shape=DECOY_BAR_PATTERN),
        legendgroup=DECOY_TRACE_NAME,
    )
    fig.update_layout(barmode="relative")


def stack_class_frames(target_frame: pd.DataFrame, decoy_frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Stack the target and decoy plot frames, naming their traces in `name`, targets first."""
    class_frames = [target_frame.assign(**{name: TARGET_TRACE_NAME}), decoy_frame.assign(**{name: DECOY_TRACE_NAME})]
    stacked_frame = pd.concat(class_frames, ignore_index=True)
    stacked_frame[name] = pd.Categorical(stacked_frame[name], categories=[TARGET_TRACE_NAME, DECOY_TRACE_NAME])
    return stacked_frame


def color_decoy_curve(fig: go.Figure) -> None:
    """Draw the decoy line in grey."""
    fig.update_traces(marker_color=DECOY_COLOR, line_color=DECOY_COLOR, selector=dict(name=DECOY_TRACE_NAME))


def style_class_histogram(fig: go.Figure) -> None:
    """Overlay the target and decoy histograms, the decoys in grey."""
    fig.update_traces(opacity=OVERLAID_HISTOGRAM_OPACITY)
    fig.update_traces(marker_color=DECOY_COLOR, selector=dict(name=DECOY_TRACE_NAME))
    fig.update_layout(barmode="overlay", bargap=0)


def _place_targets_beside_decoys(fig: go.Figure) -> None:
    """Put the target traces drawn so far in one offset group, so the decoy group sits beside them."""
    fig.update_traces(offsetgroup=TARGET_OFFSET_GROUP)
