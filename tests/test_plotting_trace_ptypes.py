import pandas as pd

from msmu._plotting._ptypes import PlotPie, PlotHeatmap, PlotSimpleBox
from msmu._plotting._trace import Trace, TraceDescribed, TraceHeatmap


def test_trace_groups_by_name():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [10, 20, 30, 40],
            "name": ["A", "A", "B", "B"],
        }
    )
    trace = Trace(data=df, x="x", y="y", name="name")
    traces = trace()
    assert len(traces) == 2
    assert {t["name"] for t in traces} == {"A", "B"}
    for entry in traces:
        assert entry["meta"] in {"A", "B"}


def test_plot_simple_box_creates_expected_traces():
    df = pd.DataFrame(
        {
            "min": [1.0, 2.0],
            "25%": [1.5, 2.5],
            "50%": [2.0, 3.0],
            "75%": [2.5, 3.5],
            "max": [3.0, 4.0],
        },
        index=["A", "B"],
    )
    plot = PlotSimpleBox(data=df)
    fig = plot.figure()
    assert len(fig.data) == 4
    assert fig.data[0].type == "box"
    assert fig.data[2].type == "scatter"


def test_plot_heatmap_sets_texttemplate_for_small_data():
    df = pd.DataFrame([[0.1, -0.2], [0.3, 0.0]], columns=["A", "B"], index=["x", "y"])
    plot = PlotHeatmap(data=df)
    fig = plot.figure()
    assert len(fig.data) == 1
    assert fig.data[0].texttemplate == "%{z:.2f}"


def test_trace_heatmap_no_texttemplate_for_large_data():
    df = pd.DataFrame([[0.0]] * 25, columns=["A"])
    trace = TraceHeatmap(data=df)
    traces = trace()
    assert traces[0]["texttemplate"] is None


def test_trace_described_fences():
    df = pd.DataFrame(
        {
            "min": [1.0],
            "25%": [2.0],
            "50%": [3.0],
            "75%": [4.0],
            "max": [5.0],
        },
        index=["A"],
    )
    trace = TraceDescribed(data=df)
    traces = trace()
    assert traces[0]["lowerfence"] == [1.0]
    assert traces[0]["upperfence"] == [5.0]


def test_plot_pie_trace_type():
    df = pd.Series([2, 3], index=["A", "B"]).to_frame()
    plot = PlotPie(data=df)
    fig = plot.figure()
    assert fig.data[0].type == "pie"
