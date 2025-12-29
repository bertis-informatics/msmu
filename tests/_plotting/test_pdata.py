import numpy as np
import pandas as pd
import pytest

from msmu._plotting._pdata import PlotData


def _detection_mask(plot_data: PlotData) -> pd.DataFrame:
    obs = plot_data._get_obs(obs_column="sample", groupby="condition")
    merged = plot_data._get_data().notna().join(obs["condition"], how="left")
    return merged.groupby("condition", observed=True).any()


def _condition_detection_counts(plot_data: PlotData) -> pd.Series:
    mask = _detection_mask(plot_data)
    return mask.sum(axis=1)


def _lengths_by_condition(plot_data: PlotData) -> dict[str, list[float]]:
    mask = _detection_mask(plot_data)
    lengths = plot_data._get_var()["length"]
    return {condition: sorted(np.round(lengths[mask.loc[condition]].tolist(), 1)) for condition in mask.index}


def _intensity_value_counts(plot_data: PlotData) -> pd.Series:
    obs = plot_data._get_obs("sample", groupby="condition")
    orig = plot_data._get_data().T
    melt = pd.melt(orig, var_name="_obs", value_name="_value").dropna()
    merged = melt.join(obs, on="_obs", how="left")
    return merged.groupby("condition", observed=False).size()


def test_get_data_matches_modality(plot_data: PlotData, plotting_mudata):
    expected = plotting_mudata["protein"].to_df()
    result = plot_data._get_data()
    pd.testing.assert_frame_equal(result, expected)


def test_get_var_returns_variable_metadata(plot_data: PlotData, plotting_mudata):
    expected = plotting_mudata["protein"].var
    result = plot_data._get_var()
    pd.testing.assert_frame_equal(result, expected)


def test_get_varm_concatenates_var_and_varm(plot_data: PlotData):
    combined = plot_data._get_varm("loadings")
    assert {"protein_class", "length", "q_value"}.issubset(combined.columns)
    assert 0 in combined.columns and 1 in combined.columns


def test_get_obs_preserves_category_order(plot_data: PlotData):
    obs = plot_data._get_obs(obs_column="sample", groupby="condition")
    assert list(obs["sample"].cat.categories) == [f"s{i:02d}" for i in range(1, 11)]
    assert list(obs["condition"].cat.categories) == ["control", "experimental"]


def test_get_bin_info_returns_consistent_metadata(plot_data: PlotData):
    data = plot_data._get_data()
    info = plot_data._get_bin_info(data, bins=3)
    assert len(info["labels"]) == 3
    assert len(info["edges"]) == 4
    assert pytest.approx(info["width"]) == (np.nanmax(data.to_numpy()) - np.nanmin(data.to_numpy())) / 3


def test_prep_var_data_counts_categories(plot_data: PlotData):
    prep_df = plot_data.prep_var_data(groupby="condition", name="protein_class", obs_column="sample")
    count_col = prep_df.columns[-1]
    totals = prep_df.groupby("condition", observed=False)[count_col].sum().sort_index()
    expected = _condition_detection_counts(plot_data).sort_index()
    pd.testing.assert_series_equal(totals, expected, check_names=False)
    assert set(prep_df["protein_class"]) <= set(plot_data._get_var()["protein_class"].cat.categories)


def test_prep_var_bar_matches_detection_totals(plot_data: PlotData):
    prep_df = plot_data.prep_var_bar(groupby="condition", var_column="protein_class", obs_column="sample")
    count_col = prep_df.columns[-1]
    totals = prep_df.groupby("condition", observed=False)[count_col].sum().sort_index()
    expected = _condition_detection_counts(plot_data).sort_index()
    pd.testing.assert_series_equal(totals, expected, check_names=False)


def test_prep_var_box_surfaces_detected_lengths(plot_data: PlotData):
    prep_df = plot_data.prep_var_box(groupby="condition", var_column="length", obs_column="sample")
    expected_lengths = _lengths_by_condition(plot_data)
    for condition, expected in expected_lengths.items():
        actual = sorted(np.round(prep_df.loc[prep_df["condition"] == condition, "length"].tolist(), 1))
        assert actual == expected


def test_prep_var_simple_box_returns_descriptives(plot_data: PlotData):
    prep_df = plot_data.prep_var_simple_box(groupby="condition", var_column="length", obs_column="sample")
    expected_lengths = _lengths_by_condition(plot_data)
    assert list(prep_df.columns) == ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    for condition, lengths in expected_lengths.items():
        stats = pd.Series(lengths).describe()
        for column in prep_df.columns:
            assert prep_df.loc[condition, column] == pytest.approx(stats[column])


def test_prep_var_hist_builds_histogram_table(plot_data: PlotData):
    bin_info = plot_data._get_bin_info(plot_data._get_var()[["length"]], bins=2)
    prep_df = plot_data.prep_var_hist(
        groupby="condition",
        var_column="length",
        obs_column="sample",
        bin_info=bin_info,
    )
    assert {"center", "label", "count", "frequency", "name"} <= set(prep_df.columns)
    assert set(prep_df["name"]) == {"control", "experimental"}
    totals = prep_df.groupby("name", observed=False)["count"].sum().sort_index()
    expected = _condition_detection_counts(plot_data).sort_index()
    pd.testing.assert_series_equal(totals, expected, check_names=False)


def test_prep_id_bar_counts_ids(plot_data: PlotData):
    prep_df = plot_data.prep_id_bar(groupby="condition", obs_column="sample")
    totals = prep_df.set_index("condition")["_count"].sort_index()
    expected = _condition_detection_counts(plot_data).sort_index()
    pd.testing.assert_series_equal(totals, expected, check_names=False)


def test_prep_intensity_hist_uses_bins(plot_data: PlotData):
    bin_info = plot_data._get_bin_info(plot_data._get_data(), bins=3)
    prep_df = plot_data.prep_intensity_hist(groupby="condition", obs_column="sample", bin_info=bin_info)
    assert {"center", "label", "count", "frequency", "name"} <= set(prep_df.columns)
    assert set(prep_df["name"]) == {"control", "experimental"}
    totals = prep_df.groupby("name", observed=False)["count"].sum().sort_index()
    expected = _intensity_value_counts(plot_data).sort_index()
    pd.testing.assert_series_equal(totals, expected, check_names=False)


def test_prep_intensity_bar_flattens_values(plot_data: PlotData):
    prep_df = plot_data.prep_intensity_bar(groupby="condition", obs_column="sample")
    assert prep_df.columns.tolist() == ["condition", "_value"]
    counts = prep_df.groupby("condition", observed=False).size().sort_index()
    expected = _intensity_value_counts(plot_data).sort_index()
    pd.testing.assert_series_equal(counts, expected, check_names=False)


def test_prep_intensity_simple_box_returns_stats(plot_data: PlotData):
    prep_df = plot_data.prep_intensity_simple_box(groupby="condition", obs_column="sample")
    assert "mean" in prep_df.columns
    assert prep_df.index.tolist() == ["control", "experimental"]
    expected = _intensity_value_counts(plot_data).sort_index()
    counts = prep_df["count"].astype("int64").sort_index()
    pd.testing.assert_series_equal(counts, expected, check_names=False)


def test_prep_missingness_step_reports_percentages(plot_data: PlotData):
    prep_df = plot_data.prep_missingness_step(obs_column="sample")
    assert prep_df["name"].unique().tolist() == ["Missingness"]
    assert prep_df["missingness"].is_monotonic_increasing
    assert prep_df["ratio"].between(0, 100).all()


def test_prep_pca_scatter_joins_obs_metadata(plot_data: PlotData):
    prep_df = plot_data.prep_pca_scatter(
        modality="protein",
        groupby="condition",
        pc_columns=["PC_1", "PC_2"],
        obs_column="sample",
    )
    assert set(["PC_1", "PC_2", "condition"]).issubset(prep_df.columns)
    assert prep_df["condition"].cat.categories.tolist() == ["control", "experimental"]
    assert len(prep_df) == 10


def test_prep_umap_scatter_joins_obs_metadata(plot_data: PlotData):
    prep_df = plot_data.prep_umap_scatter(
        modality="protein",
        groupby="condition",
        umap_columns=["UMAP_1", "UMAP_2"],
        obs_column="sample",
    )
    assert set(["UMAP_1", "UMAP_2", "condition"]).issubset(prep_df.columns)
    assert prep_df["condition"].cat.categories.tolist() == ["control", "experimental"]
    assert len(prep_df) == 10


def test_prep_id_upset_returns_combination_and_item_counts(plot_data: PlotData):
    combos, items = plot_data.prep_id_upset(groupby="condition", obs_column="sample")
    assert combos["count"].sum() == len(plot_data._get_var())
    totals = _condition_detection_counts(plot_data).sort_index()
    pd.testing.assert_series_equal(items.sort_index(), totals, check_names=False)
    assert isinstance(items, pd.Series)


def test_prep_intensity_correlation_masks_upper_triangle(plot_data: PlotData):
    corr = plot_data.prep_intensity_correlation(groupby="condition", obs_column="sample")
    assert corr.shape == (2, 2)
    assert corr.columns.tolist() == ["control", "experimental"]
    assert corr.index.tolist() == ["control", "experimental"]
    assert np.isnan(corr.iloc[0, 1])
    assert corr.iloc[1, 0] <= 1
    assert np.allclose(np.diag(corr), 1)
