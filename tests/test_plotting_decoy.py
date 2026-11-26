"""QC plots called with ``include_decoy=True`` draw the decoy features next to the targets."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData

import msmu as mm
from msmu._plotting._decoy import DECOY_BAR_PATTERN, DECOY_COLOR, DECOY_OFFSET_GROUP, TARGET_OFFSET_GROUP

SAMPLES = ["s1", "s2", "s3", "s4"]
GROUPS = ["A", "A", "B", "B"]
# Targets t1-t3, then decoys d1 (in s1, s2 and s4) and d2 (in s4 only).
VALUES = np.array(
    [
        [20.0, 21.0, np.nan, 15.0, np.nan],
        [22.0, np.nan, 19.0, 16.0, np.nan],
        [21.0, 20.0, 18.0, np.nan, np.nan],
        [np.nan, 23.0, 17.0, 14.0, 13.0],
    ]
)


def _make_mdata(values: np.ndarray = VALUES) -> MuData:
    var = pd.DataFrame(
        {"decoy": [0, 0, 0, 1, 1], "charge": [2, 3, 2, 2, 3], "score": [10.0, 20.0, 30.0, 5.0, 8.0]},
        index=["t1", "t2", "t3", "d1", "d2"],
    )
    mdata = MuData({"protein": AnnData(X=values.copy(), obs=pd.DataFrame(index=SAMPLES), var=var)})
    mdata.obs["sample"] = SAMPLES
    mdata.obs["group"] = pd.Categorical(GROUPS, categories=["A", "B"])
    return mdata


def _trace(figure, name: str):
    return next(trace for trace in figure.data if trace.name == name)


def test_plot_id_stacks_the_decoy_count_on_each_group():
    figure = mm.pl.plot_id(_make_mdata(), modality="protein", groupby="group", include_decoy=True)

    decoy = _trace(figure, "Decoy")
    assert list(decoy.x) == ["A", "B"] and list(decoy.y) == [1, 2]
    assert decoy.marker.color == DECOY_COLOR and decoy.marker.pattern.shape == DECOY_BAR_PATTERN
    assert [list(_trace(figure, group).y) for group in ("A", "B")] == [[3], [3]]
    assert figure.layout.barmode == "stack"


def test_plot_id_counts_decoys_without_values_as_absent():
    # Targets are quantified, so decoys without a value count as 0 rather than by annotation.
    values = VALUES.copy()
    values[:, 3:] = np.nan

    figure = mm.pl.plot_id(_make_mdata(values), modality="protein", groupby="group", include_decoy=True)

    assert list(_trace(figure, "Decoy").y) == [0, 0]


def test_plot_intensity_histogram_overlays_one_target_and_one_decoy_trace():
    figure = mm.pl.plot_intensity(_make_mdata(), modality="protein", groupby="group", include_decoy=True)

    assert [trace.name for trace in figure.data] == ["Target", "Decoy"]
    assert sum(_trace(figure, "Target").y) == 9
    assert sum(_trace(figure, "Decoy").y) == 4
    assert _trace(figure, "Decoy").marker.color == DECOY_COLOR
    assert figure.layout.barmode == "overlay"


def test_plot_intensity_box_puts_a_decoy_box_beside_each_group():
    figure = mm.pl.plot_intensity(_make_mdata(), modality="protein", groupby="group", ptype="box", include_decoy=True)

    decoy_boxes = [trace for trace in figure.data if trace.name == "Decoy"]
    assert [(box.x[0], box.median[0]) for box in decoy_boxes] == [("A", 15.5), ("B", 13.5)]
    assert [box.showlegend for box in decoy_boxes] == [True, False]
    assert {box.offsetgroup for box in decoy_boxes} == {DECOY_OFFSET_GROUP}
    assert {_trace(figure, group).offsetgroup for group in ("A", "B")} == {TARGET_OFFSET_GROUP}
    assert figure.layout.boxmode == "group"


def test_plot_intensity_violin_draws_the_decoy_values_in_grey():
    figure = mm.pl.plot_intensity(_make_mdata(), modality="protein", groupby="group", ptype="vln", include_decoy=True)

    decoy = _trace(figure, "Decoy")
    assert sorted(zip(decoy.x, decoy.y)) == [("A", 15.0), ("A", 16.0), ("B", 13.0), ("B", 14.0)]
    assert decoy.line.color == DECOY_COLOR
    assert figure.layout.violinmode == "group"


def test_plot_missingness_draws_a_target_and_a_decoy_curve():
    figure = mm.pl.plot_missingness(_make_mdata(), modality="protein", include_decoy=True)

    assert [trace.name for trace in figure.data] == ["Target", "Decoy"]
    decoy = _trace(figure, "Decoy")
    assert decoy.line.color == DECOY_COLOR
    # d1 misses 1 of 4 samples and d2 misses 3: half the decoys at 25 %, all of them at 75 %.
    assert dict(zip(decoy.x, decoy.y))[25.0] == 50.0
    assert dict(zip(decoy.x, decoy.y))[75.0] == 100.0


def test_plot_var_stacks_hatched_decoy_bars_beside_the_targets_in_the_same_colors():
    figure = mm.pl.plot_var(_make_mdata(), modality="protein", groupby="group", var_column="charge", include_decoy=True)

    target_bars = {trace.name: trace for trace in figure.data if trace.offsetgroup == TARGET_OFFSET_GROUP}
    decoy_bars = {trace.name: trace for trace in figure.data if trace.offsetgroup == DECOY_OFFSET_GROUP}
    assert (list(decoy_bars["2"].x), list(decoy_bars["2"].y)) == (["A", "B"], [1, 1])
    assert (list(decoy_bars["3"].x), list(decoy_bars["3"].y)) == (["B"], [1])
    for category, decoy_bar in decoy_bars.items():
        assert decoy_bar.marker.color == target_bars[category].marker.color
        assert decoy_bar.marker.pattern.shape == DECOY_BAR_PATTERN
        assert decoy_bar.showlegend is False
    legend_entry = _trace(figure, "Decoy")
    assert legend_entry.marker.color == DECOY_COLOR and legend_entry.offsetgroup is None


def test_plot_var_histogram_counts_each_feature_once_per_class():
    figure = mm.pl.plot_var(
        _make_mdata(), modality="protein", groupby="group", var_column="score", ptype="hist", include_decoy=True
    )

    assert [trace.name for trace in figure.data] == ["Target", "Decoy"]
    assert (sum(_trace(figure, "Target").y), sum(_trace(figure, "Decoy").y)) == (3, 2)


def test_plot_var_box_puts_the_decoy_values_beside_each_group():
    figure = mm.pl.plot_var(
        _make_mdata(), modality="protein", groupby="group", var_column="score", ptype="box", include_decoy=True
    )

    decoy = _trace(figure, "Decoy")
    assert sorted(zip(decoy.x, decoy.y)) == [("A", 5.0), ("B", 5.0), ("B", 8.0)]


def test_group_colors_leave_the_decoys_grey():
    figure = mm.pl.plot_id(_make_mdata(), modality="protein", obs_column="sample", colorby="group", include_decoy=True)

    assert _trace(figure, "Decoy").marker.color == DECOY_COLOR
    assert _trace(figure, "s1").marker.color == _trace(figure, "s2").marker.color


@pytest.mark.parametrize(
    ("plot_function", "arguments"),
    [
        (mm.pl.plot_id, {"groupby": "group"}),
        (mm.pl.plot_intensity, {"groupby": "group"}),
        (mm.pl.plot_missingness, {}),
        (mm.pl.plot_var, {"groupby": "group", "var_column": "charge"}),
    ],
)
def test_include_decoy_without_decoys_plots_the_targets_with_a_warning(plot_function, arguments, caplog):
    targets_only = _make_mdata()
    targets_only.mod["protein"] = targets_only["protein"][:, :3].copy()
    targets_only.update()

    with caplog.at_level("WARNING", logger="msmu"):
        result = plot_function(targets_only, modality="protein", include_decoy=True, **arguments)

    assert result.to_json() == plot_function(targets_only, modality="protein", **arguments).to_json()
    assert "has no decoy features; plotting targets only" in caplog.text
