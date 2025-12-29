import pytest
import pandas as pd
from msmu._plotting._utils import resolve_obs_column, resolve_plot_columns


def test_resolve_obs_column_requested_exists(sample_mudata):
    sample_mudata.obs["my_col"] = ["a", "b"]
    col = resolve_obs_column(sample_mudata, requested="my_col")
    assert col == "my_col"
    assert isinstance(sample_mudata.obs["my_col"].dtype, pd.CategoricalDtype)


def test_resolve_obs_column_fallback(sample_mudata):
    # Ensure no default columns exist
    for col in ["sample", "filename"]:
        if col in sample_mudata.obs.columns:
            del sample_mudata.obs[col]

    col = resolve_obs_column(sample_mudata)
    assert col == "__obs_idx__"
    assert "__obs_idx__" in sample_mudata.obs.columns
    assert list(sample_mudata.obs["__obs_idx__"]) == list(sample_mudata.obs.index)


def test_resolve_obs_column_priority(sample_mudata):
    sample_mudata.obs["sample"] = ["s1", "s2"]
    sample_mudata.obs["filename"] = ["f1", "f2"]

    col = resolve_obs_column(sample_mudata)
    assert col == "sample"


def test_resolve_plot_columns_defaults(sample_mudata):
    sample_mudata.obs["sample"] = ["s1", "s2"]
    groupby, obs_col = resolve_plot_columns(sample_mudata, groupby=None, obs_column=None)

    assert obs_col == "sample"
    assert groupby == "sample"


def test_resolve_plot_columns_explicit(sample_mudata):
    sample_mudata.obs["group"] = ["g1", "g2"]
    # Ensure fallback for obs_column
    for col in ["sample", "filename"]:
        if col in sample_mudata.obs.columns:
            del sample_mudata.obs[col]

    groupby, obs_col = resolve_plot_columns(sample_mudata, groupby="group", obs_column=None)

    assert groupby == "group"
    assert obs_col == "__obs_idx__"
