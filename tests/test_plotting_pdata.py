import numpy as np
import pandas as pd

from msmu._plotting._pdata import PlotData


def test_get_obs_casts_groupby_categorical(mdata):
    pdata = PlotData(mdata, "psm", obs_column="sample")
    obs_df = pdata._get_obs("sample", groupby="group")
    assert isinstance(obs_df["group"].dtype, pd.CategoricalDtype)
    assert list(obs_df["group"].cat.categories) == ["A", "B"]


def test_get_bin_info_raises_on_empty(mdata):
    pdata = PlotData(mdata, "psm", obs_column="sample")
    empty = pd.DataFrame()
    try:
        pdata._get_bin_info(empty, bins=3)
    except ValueError as exc:
        assert "empty data" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for empty data")


def test_prep_var_hist_outputs_expected_columns(mdata):
    pdata = PlotData(mdata, "psm", obs_column="sample")
    bin_info = pdata._get_bin_info(mdata["psm"].var["score"], bins=2)
    df = pdata.prep_var_hist("group", "score", "sample", bin_info)
    assert set(df.columns) == {"center", "label", "count", "frequency", "name"}
    assert df["name"].cat.categories.tolist() == ["A", "B"]


def test_prep_intensity_hist_counts_non_nan(mdata):
    pdata = PlotData(mdata, "psm", obs_column="sample")
    bin_info = pdata._get_bin_info(pdata._get_data(), bins=2)
    df = pdata.prep_intensity_hist("group", "sample", bin_info)
    non_nan = np.count_nonzero(~np.isnan(pdata._get_data().to_numpy()))
    assert df["count"].sum() == non_nan


def test_prep_missingness_step_has_expected_ranges(mdata):
    pdata = PlotData(mdata, "psm", obs_column="sample")
    df = pdata.prep_missingness_step("sample")
    assert set(df.columns) == {"missingness", "count", "ratio", "name"}
    assert df["name"].unique().tolist() == ["Missingness"]
    assert df["ratio"].max() == 100


def test_prep_pca_scatter_columns(mdata):
    pdata = PlotData(mdata, "protein", obs_column="sample")
    df = pdata.prep_pca_scatter("protein", "group", ["PC_1", "PC_2"], "sample")
    assert {"PC_1", "PC_2", "group"}.issubset(df.columns)
    assert isinstance(df["group"].dtype, pd.CategoricalDtype)


def test_prep_umap_scatter_columns(mdata):
    pdata = PlotData(mdata, "protein", obs_column="sample")
    df = pdata.prep_umap_scatter("protein", "group", ["UMAP_1", "UMAP_2"], "sample")
    assert {"UMAP_1", "UMAP_2", "group"}.issubset(df.columns)
    assert isinstance(df["group"].dtype, pd.CategoricalDtype)


def test_prep_id_upset_outputs(mdata):
    pdata = PlotData(mdata, "protein", obs_column="sample")
    combination_counts, item_counts = pdata.prep_id_upset("group", "sample")
    assert list(combination_counts.columns) == ["combination", "count"]
    assert isinstance(item_counts, pd.Series)


def test_prep_intensity_correlation_lower_triangle(mdata):
    pdata = PlotData(mdata, "protein", obs_column="sample")
    corrs = pdata.prep_intensity_correlation("group", "sample")
    assert pd.isna(corrs.iloc[0, 1])
