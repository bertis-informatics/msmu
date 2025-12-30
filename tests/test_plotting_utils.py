import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pytest

from msmu._plotting._utils import (
    apply_color_if_needed,
    format_modality,
    get_bin_info,
    get_pc_cols,
    get_umap_cols,
    merge_traces,
    resolve_plot_columns,
    resolve_obs_column,
    set_color,
)


def test_resolve_obs_column_requested(mdata):
    resolved = resolve_obs_column(mdata, "group")
    assert resolved == "group"
    assert isinstance(mdata.obs["group"].dtype, pd.CategoricalDtype)


def test_resolve_obs_column_fallback_creates_column(mdata):
    mdata_local = mdata.copy()
    mdata_local.uns["plotting"] = {}
    mdata_local.obs = pd.DataFrame(index=mdata_local.obs.index)

    resolved = resolve_obs_column(mdata_local, "missing_group")
    assert resolved == "missing_group"
    assert "missing_group" in mdata_local.obs.columns
    assert list(mdata_local.obs["missing_group"].astype(str)) == list(mdata_local.obs.index.astype(str))
    assert isinstance(mdata_local.obs["missing_group"].dtype, pd.CategoricalDtype)


def test_get_bin_info_numeric_values():
    data = pd.Series([0.0, 1.0, 2.0, 3.0])
    bin_info = get_bin_info(data, bins=2)
    assert bin_info["width"] == pytest.approx(1.5)
    assert bin_info["edges"] == [0.0, 1.5, 3.0]
    assert bin_info["centers"] == [0.75, 2.25]
    assert bin_info["labels"] == ["0.0 - 1.5", "1.5 - 3.0"]


def test_get_pc_cols_sorted_and_validated(mdata):
    pcs, columns = get_pc_cols(mdata, "protein", (2, 1))
    assert pcs == (1, 2)
    assert columns == ["PC_1", "PC_2"]


def test_get_pc_cols_invalid_length(mdata):
    with pytest.raises(ValueError, match="Only 2 PCs are allowed"):
        get_pc_cols(mdata, "protein", (1, 2, 3))


def test_get_umap_cols_valid(mdata):
    columns = get_umap_cols(mdata, "protein")
    assert columns == ["UMAP_1", "UMAP_2"]


def test_get_pc_cols_invalid_types(mdata):
    with pytest.raises(ValueError, match="PCs must be integers"):
        get_pc_cols(mdata, "protein", (1, "2"))


def test_get_pc_cols_duplicate(mdata):
    with pytest.raises(ValueError, match="PCs must be different"):
        get_pc_cols(mdata, "protein", (1, 1))


def test_get_pc_cols_missing_pca(mdata):
    mdata_local = mdata.copy()
    del mdata_local["protein"].obsm["X_pca"]
    with pytest.raises(ValueError, match="No PCA found"):
        get_pc_cols(mdata_local, "protein", (1, 2))


def test_get_umap_cols_missing_umap(mdata):
    mdata_local = mdata.copy()
    del mdata_local["protein"].obsm["X_umap"]
    with pytest.raises(ValueError, match="No UMAP found"):
        get_umap_cols(mdata_local, "protein")


def test_apply_color_if_needed_sets_color(mdata):
    fig = go.Figure()
    fig.add_trace(go.Scatter(name="A", x=[1], y=[1], mode="markers"))
    fig.add_trace(go.Scatter(name="B", x=[2], y=[2], mode="markers"))

    fig = apply_color_if_needed(
        fig,
        mdata=mdata,
        modality="psm",
        groupby="group",
        colorby="group",
        obs_column="group",
        template="plotly",
    )

    colorway = pio.templates["plotly"].layout["colorway"]
    assert fig.data[0].name == "A"
    assert fig.data[1].name == "B"
    assert fig.data[0].marker.color == colorway[0]
    assert fig.data[1].marker.color == colorway[1]


def test_apply_color_if_needed_ignored_when_groupby_differs(mdata, capsys):
    fig = go.Figure()
    fig.add_trace(go.Scatter(name="A", x=[1], y=[1], mode="markers"))
    fig.add_trace(go.Scatter(name="B", x=[2], y=[2], mode="markers"))

    fig = apply_color_if_needed(
        fig,
        mdata=mdata,
        modality="psm",
        groupby="batch",
        colorby="group",
        obs_column="sample",
        template="plotly",
    )

    out = capsys.readouterr().out
    assert "Ignoring 'colorby' parameter" in out
    assert fig.data[0].marker.color is None


def test_set_color_orders_traces(mdata):
    fig = go.Figure()
    fig.add_trace(go.Scatter(name="B", x=[1], y=[1], mode="markers"))
    fig.add_trace(go.Scatter(name="A", x=[2], y=[2], mode="markers"))

    fig = set_color(fig, mdata, "psm", "group", "group", template="plotly")
    assert fig.data[0].name == "A"
    assert fig.data[1].name == "B"


def test_resolve_plot_columns_defaults(mdata):
    groupby, obs_column = resolve_plot_columns(mdata, None, None)
    assert groupby == "group"
    assert obs_column == "group"


def test_merge_traces_overrides_options():
    traces = [{"x": [1], "y": [2], "marker": {"size": 5}}]
    merged = merge_traces(traces, {"marker": {"size": 10}, "mode": "lines"})
    assert merged[0]["marker"]["size"] == 10
    assert merged[0]["mode"] == "lines"


def test_format_modality_labels(mdata):
    assert format_modality(mdata, "psm") == "Precursor"
    assert format_modality(mdata, "peptide") == "Peptide"
    assert format_modality(mdata, "protein") == "Protein"
    assert format_modality(mdata, "phospho_site") == "Phospho_site"


def test_format_modality_invalid(mdata):
    with pytest.raises(ValueError, match="Unknown modality"):
        format_modality(mdata, "unknown")
