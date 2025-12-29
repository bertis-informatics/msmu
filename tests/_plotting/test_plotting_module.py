import math

import pytest

from msmu import pl


@pytest.mark.parametrize("func_name", pl.__all__)
def test_plotting_public_api_has_attributes(func_name):
    assert hasattr(pl, func_name), f"{func_name} missing from plotting module"


def test_plot_id_returns_bar_figure(plotting_mudata):
    fig = pl.plot_id(plotting_mudata, modality="protein", groupby="condition")

    assert fig.data, "plot_id should return traces"
    assert fig.data[0].type == "bar"


def test_plot_intensity_histogram(plotting_mudata):
    fig = pl.plot_intensity(plotting_mudata, modality="protein", groupby="condition", ptype="hist", bins=4)

    assert all(trace.type == "bar" for trace in fig.data)


def test_plot_intensity_invalid_type(plotting_mudata):
    with pytest.raises(ValueError):
        pl.plot_intensity(plotting_mudata, modality="protein", ptype="invalid")


def test_plot_missingness_generates_step_curve(plotting_mudata):
    fig = pl.plot_missingness(plotting_mudata, modality="protein")

    assert fig.data[0].mode == "lines+markers"
    assert fig.layout.xaxis.title.text.startswith("Data Completeness")


def test_plot_pca_scatter_labels_components(plotting_mudata):
    fig = pl.plot_pca(plotting_mudata, modality="protein", groupby="sample", colorby="condition", pcs=(1, 2))

    assert fig.data[0].type == "scatter"
    assert "PC_1" in fig.layout.xaxis.title.text
    assert "PC_2" in fig.layout.yaxis.title.text


def test_plot_umap_scatter_uses_expected_axes(plotting_mudata):
    fig = pl.plot_umap(plotting_mudata, modality="protein", groupby="sample", colorby="condition")

    assert fig.data[0].type == "scatter"
    assert fig.layout.xaxis.title.text == "UMAP_1"
    assert fig.layout.yaxis.title.text == "UMAP_2"


def test_plot_upset_sets_descriptive_title(plotting_mudata):
    fig = pl.plot_upset(plotting_mudata, modality="protein", groupby="condition")

    assert len(fig.data) >= 3
    assert "Intersection of Proteins" in fig.layout.title.text


def test_plot_correlation_masks_upper_triangle(plotting_mudata):
    fig = pl.plot_correlation(plotting_mudata, modality="protein", groupby="condition")

    assert fig.data[0].type == "heatmap"
    upper_value = fig.data[0].z[0][1]
    assert upper_value is None or (isinstance(upper_value, float) and math.isnan(upper_value))


def test_plot_var_categorical_stack(plotting_mudata):
    fig = pl.plot_var(
        plotting_mudata,
        modality="protein",
        groupby="condition",
        var_column="protein_class",
        ptype="stack",
    )

    assert fig.data[0].type == "bar"
    assert fig.layout.legend.title.text == "Protein class"


def test_plot_var_numeric_box(plotting_mudata):
    fig = pl.plot_var(
        plotting_mudata,
        modality="protein",
        groupby="condition",
        var_column="length",
        ptype="box",
    )

    assert all(trace.type == "box" for trace in fig.data)
    assert fig.layout.yaxis.title.text.startswith("Number of Protein")


def test_plot_var_simple_box(plotting_mudata):
    fig = pl.plot_var(
        plotting_mudata,
        modality="protein",
        groupby="condition",
        var_column="length",
        ptype="simple_box",
    )

    assert any(trace.type == "box" for trace in fig.data)


def test_plot_var_violin(plotting_mudata):
    fig = pl.plot_var(
        plotting_mudata,
        modality="protein",
        groupby="condition",
        var_column="length",
        ptype="violin",
    )

    assert all(trace.type == "violin" for trace in fig.data)


def test_plot_var_histogram(plotting_mudata):
    fig = pl.plot_var(
        plotting_mudata,
        modality="protein",
        groupby="condition",
        var_column="length",
        ptype="hist",
        bins=5,
    )

    assert all(trace.type == "bar" for trace in fig.data)


def test_plot_intensity_box(plotting_mudata):
    fig = pl.plot_intensity(
        plotting_mudata,
        modality="protein",
        groupby="condition",
        ptype="box",
    )

    assert any(trace.type == "box" for trace in fig.data)


def test_plot_intensity_violin(plotting_mudata):
    fig = pl.plot_intensity(
        plotting_mudata,
        modality="protein",
        groupby="condition",
        ptype="violin",
    )

    assert all(trace.type == "violin" for trace in fig.data)
