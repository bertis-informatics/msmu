import numpy as np
import pandas as pd
import pytest

from msmu._plotting._plots import (
    plot_correlation,
    plot_id,
    plot_intensity,
    plot_missingness,
    plot_pca,
    plot_umap,
    plot_upset,
    plot_var,
    plot_volcano,
)


def test_plot_id_defaults_to_fallback_obs_index(mdata):
    mdata_local = mdata.copy()
    mdata_local.uns["plotting"] = {}
    mdata_local.obs = pd.DataFrame(index=mdata_local.obs.index)

    fig = plot_id(mdata_local, modality="protein")

    assert {trace.name for trace in fig.data} == set(mdata_local.obs.index.astype(str))
    assert all(trace.type == "bar" for trace in fig.data)
    assert "__obs_idx__" not in mdata_local.obs.columns


def test_plot_id_fallback_obs_wins_var_name_collision(mdata):
    mdata_local = mdata.copy()
    mdata_local.uns["plotting"] = {}
    mdata_local.obs = pd.DataFrame(index=mdata_local.obs.index)
    mdata_local.mod["protein"].var["__obs_idx__"] = ["var_a", "var_b", "var_c"]

    fig = plot_id(mdata_local, modality="protein")

    assert {trace.name for trace in fig.data} == set(mdata_local.obs.index.astype(str))
    assert {trace.name for trace in fig.data}.isdisjoint({"var_a", "var_b", "var_c"})
    assert "__obs_idx__" not in mdata_local.obs.columns


def test_plot_var_defaults_to_fallback_obs_index(mdata):
    mdata_local = mdata.copy()
    mdata_local.uns["plotting"] = {}
    mdata_local.obs = pd.DataFrame(index=mdata_local.obs.index)

    fig = plot_var(mdata_local, modality="protein", var_column="class")

    assert len(fig.data) == 2
    assert all(trace.type == "bar" for trace in fig.data)
    assert "__obs_idx__" not in mdata_local.obs.columns


def test_plot_id_explicit_missing_obs_column_raises(mdata):
    with pytest.raises(ValueError, match="obs_column 'missing_group' is not present in mdata.obs"):
        plot_id(mdata, modality="protein", obs_column="missing_group")


def test_plot_id_explicit_fallback_obs_column_raises_when_missing(mdata):
    mdata_local = mdata.copy()
    mdata_local.uns["plotting"] = {}
    mdata_local.obs = pd.DataFrame(index=mdata_local.obs.index)

    with pytest.raises(ValueError, match="obs_column '__obs_idx__' is not present in mdata.obs"):
        plot_id(mdata_local, modality="protein", obs_column="__obs_idx__")


def test_plot_id_uses_precursor_label(mdata):
    fig = plot_id(mdata, modality="psm", groupby="group", obs_column="sample")
    assert len(fig.data) == 2
    assert all(trace.type == "bar" for trace in fig.data)


def test_plot_id_supports_duplicate_default_obs_column(mdata):
    fig = plot_id(mdata, modality="protein", groupby="group")
    assert len(fig.data) == 2
    assert all(trace.type == "bar" for trace in fig.data)


def test_plot_intensity_hist_builds_traces(mdata):
    fig = plot_intensity(
        mdata,
        modality="psm",
        groupby="group",
        ptype="hist",
        obs_column="sample",
        bins=2,
    )
    assert len(fig.data) == 2
    assert all(trace.type == "bar" for trace in fig.data)


def test_plot_missingness_builds_step_plot(mdata):
    fig = plot_missingness(mdata, modality="psm", obs_column="sample")
    assert len(fig.data) == 1
    assert fig.data[0].name == "Missingness"


def test_plot_functions_apply_msmu_template_without_set_templates(mdata):
    """Regression (BID-115): EVERY public pl.plot_* output carries the msmu house style even
    when the global Plotly default was never switched to msmu via set_templates().

    Covers all public plot functions — including plot_volcano, which takes a results frame
    (no PlotContext) and does not route through finalize_figure — so a future function that
    forgets the msmu template is caught here.
    """
    import plotly.io as pio

    de_results = pd.DataFrame(
        {
            "features": ["p1", "p2", "p3", "p4", "p5", "p6"],
            "log2fc": [2.0, -2.2, 0.1, 1.6, -1.9, 0.0],
            "p_value": [0.001, 0.002, 0.5, 0.01, 0.02, 0.9],
        }
    )
    builders = {
        "plot_id": lambda: plot_id(mdata, modality="protein", groupby="group", obs_column="sample"),
        "plot_intensity": lambda: plot_intensity(
            mdata, modality="psm", groupby="group", ptype="box", obs_column="sample"
        ),
        "plot_missingness": lambda: plot_missingness(mdata, modality="psm", obs_column="sample"),
        "plot_correlation": lambda: plot_correlation(
            mdata, modality="protein", groupby="group", obs_column="sample"
        ),
        "plot_var": lambda: plot_var(
            mdata, modality="psm", groupby="group", var_column="class", obs_column="sample"
        ),
        "plot_pca": lambda: plot_pca(mdata, modality="protein", groupby="group", obs_column="sample"),
        "plot_umap": lambda: plot_umap(mdata, modality="protein", groupby="group", obs_column="sample"),
        "plot_upset": lambda: plot_upset(mdata, modality="protein", groupby="group", obs_column="sample"),
        "plot_volcano": lambda: plot_volcano(de_results, ctrl="A", expr="B", log2fc_threshold=1.0),
    }

    original_default = pio.templates.default
    pio.templates.default = "plotly"  # a session that never opted into set_templates()
    try:
        for name, build in builders.items():
            fig = build()
            # per-figure template application, not the global default
            bg = fig.layout.template.layout.plot_bgcolor
            assert bg == "white", f"{name}: expected msmu template (white bg), got {bg!r}"
    finally:
        pio.templates.default = original_default


def test_plot_pca_and_umap(mdata):
    pca_fig = plot_pca(mdata, modality="protein", groupby="group", obs_column="sample")
    assert len(pca_fig.data) == 2
    assert all(trace.type == "scatter" for trace in pca_fig.data)

    umap_fig = plot_umap(mdata, modality="protein", groupby="group", obs_column="sample")
    assert len(umap_fig.data) == 2
    assert all(trace.type == "scatter" for trace in umap_fig.data)


def test_plot_pca_accepts_ndarray_obsm(mdata):
    mdata_local = mdata.copy()
    mdata_local["protein"].obsm["X_pca"] = np.asarray(mdata_local["protein"].obsm["X_pca"])

    fig = plot_pca(mdata_local, modality="protein", groupby="group", obs_column="sample")

    assert len(fig.data) == 2
    assert all(trace.type == "scatter" for trace in fig.data)
    assert fig.layout.xaxis.title.text.startswith("PC_1")
    assert fig.layout.yaxis.title.text.startswith("PC_2")


def test_plot_umap_accepts_ndarray_obsm(mdata):
    mdata_local = mdata.copy()
    mdata_local["protein"].obsm["X_umap"] = np.asarray(mdata_local["protein"].obsm["X_umap"])

    fig = plot_umap(mdata_local, modality="protein", groupby="group", obs_column="sample")

    assert len(fig.data) == 2
    assert all(trace.type == "scatter" for trace in fig.data)
    assert fig.layout.xaxis.title.text == "UMAP_1"
    assert fig.layout.yaxis.title.text == "UMAP_2"


def test_plot_pca_and_umap_with_custom_keys(mdata):
    mdata_local = mdata.copy()
    mdata_local["protein"].obsm["X_pca_custom"] = mdata_local["protein"].obsm["X_pca"].copy()
    mdata_local["protein"].uns["X_pca_custom"] = mdata_local["protein"].uns["X_pca"].copy()
    mdata_local["protein"].obsm["X_umap_custom"] = mdata_local["protein"].obsm["X_umap"].copy()

    pca_fig = plot_pca(
        mdata_local,
        modality="protein",
        groupby="group",
        obs_column="sample",
        key="X_pca_custom",
    )
    assert len(pca_fig.data) == 2
    assert all(trace.type == "scatter" for trace in pca_fig.data)

    umap_fig = plot_umap(
        mdata_local,
        modality="protein",
        groupby="group",
        obs_column="sample",
        key="X_umap_custom",
    )
    assert len(umap_fig.data) == 2
    assert all(trace.type == "scatter" for trace in umap_fig.data)


def test_plot_correlation_heatmap(mdata):
    fig = plot_correlation(mdata, modality="protein", groupby="group", obs_column="sample")
    assert len(fig.data) == 1
    assert fig.data[0].zmin == -1


def test_plot_var_stack_uses_categorical(mdata):
    fig = plot_var(mdata, modality="psm", groupby="group", var_column="class", obs_column="sample")
    assert len(fig.data) == 2
    assert all(trace.type == "bar" for trace in fig.data)


def test_plot_var_histogram_mode(mdata):
    fig = plot_var(
        mdata,
        modality="psm",
        groupby="group",
        var_column="score",
        obs_column="sample",
        ptype="hist",
        bins=3,
    )
    assert len(fig.data) == 2
    assert all(trace.type == "bar" for trace in fig.data)


def test_plot_var_simple_box_mode(mdata):
    fig = plot_var(
        mdata,
        modality="psm",
        groupby="group",
        var_column="score",
        obs_column="sample",
        ptype="simplebox",
    )
    assert len(fig.data) == 2
    assert all(trace.type == "box" for trace in fig.data)


def test_plot_var_violin_mode(mdata):
    fig = plot_var(
        mdata,
        modality="psm",
        groupby="group",
        var_column="score",
        obs_column="sample",
        ptype="vln",
    )
    assert len(fig.data) == 2
    assert all(trace.type == "violin" for trace in fig.data)


def test_plot_upset_builds_traces(mdata):
    fig = plot_upset(mdata, modality="protein", groupby="group", obs_column="sample")
    assert len(fig.data) >= 3


def test_plot_intensity_unknown_type_raises(mdata):
    try:
        plot_intensity(mdata, modality="psm", groupby="group", ptype="nope", obs_column="sample")
    except ValueError as exc:
        assert "Unknown plot type" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown plot type")


def test_plot_var_missing_column_raises(mdata):
    try:
        plot_var(mdata, modality="psm", groupby="group", obs_column="sample")
    except ValueError as exc:
        assert "var_column must be specified" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing var_column")


def test_plot_var_auto_ptype_numeric(mdata):
    fig = plot_var(mdata, modality="psm", groupby="group", var_column="score", obs_column="sample")
    assert fig.data[0].type == "bar"


def test_plot_intensity_box_and_violin(mdata):
    fig_box = plot_intensity(mdata, modality="psm", groupby="group", ptype="box", obs_column="sample")
    assert len(fig_box.data) == 2
    assert all(trace.type == "box" for trace in fig_box.data)

    fig_vln = plot_intensity(mdata, modality="psm", groupby="group", ptype="vln", obs_column="sample")
    assert len(fig_vln.data) == 2
    assert all(trace.type == "violin" for trace in fig_vln.data)


def test_plot_var_invalid_type_raises(mdata):
    try:
        plot_var(
            mdata,
            modality="psm",
            groupby="group",
            var_column="class",
            obs_column="sample",
            ptype="nope",
        )
    except ValueError as exc:
        assert "Unknown plot type" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid ptype")


def test_plot_upset_subset_filtering(mdata):
    fig = plot_upset(
        mdata,
        modality="protein",
        subset="A",
        subset_column="group",
        groupby="group",
        obs_column="sample",
    )
    assert len(fig.data) >= 3
