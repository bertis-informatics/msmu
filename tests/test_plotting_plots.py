from msmu._plotting._plots import (
    plot_correlation,
    plot_id,
    plot_intensity,
    plot_missingness,
    plot_pca,
    plot_umap,
    plot_upset,
    plot_var,
)


def test_plot_id_uses_precursor_label(mdata):
    fig = plot_id(mdata, modality="psm", groupby="group", obs_column="sample")
    assert len(fig.data) == 2
    assert all(trace.type == "bar" for trace in fig.data)


def test_plot_intensity_hist_builds_traces(mdata):
    fig = plot_intensity(mdata, modality="psm", groupby="group", ptype="hist", obs_column="sample", bins=2)
    assert len(fig.data) == 2
    assert all(trace.type == "bar" for trace in fig.data)


def test_plot_missingness_builds_step_plot(mdata):
    fig = plot_missingness(mdata, modality="psm", obs_column="sample")
    assert len(fig.data) == 1
    assert fig.data[0].name == "Missingness"


def test_plot_pca_and_umap(mdata):
    pca_fig = plot_pca(mdata, modality="protein", groupby="group", obs_column="sample")
    assert len(pca_fig.data) == 2
    assert all(trace.type == "scatter" for trace in pca_fig.data)

    umap_fig = plot_umap(mdata, modality="protein", groupby="group", obs_column="sample")
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
    assert len(fig.data) == 4
    assert any(trace.type == "box" for trace in fig.data)
    assert any(trace.type == "scatter" for trace in fig.data)


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
    assert len(fig_box.data) == 4
    assert any(trace.type == "box" for trace in fig_box.data)

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
