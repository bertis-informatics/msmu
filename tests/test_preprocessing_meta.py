import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData

import msmu as mm


def _make_mdata(obs_index: list[str]) -> MuData:
    obs = pd.DataFrame(index=obs_index)
    var = pd.DataFrame(index=["feature_1"])
    x = np.ones((len(obs_index), 1))
    return MuData({"psm": AnnData(X=x, obs=obs, var=var)})


def test_add_meta_raises_when_default_index_matching_has_no_exact_matches():
    mdata = _make_mdata(["run_1", "run_2"])
    metadata = pd.DataFrame(
        {
            "condition": ["treated", "control"],
            "batch": ["B1", "B2"],
        },
        index=["/data/run_1.raw", "/data/run_2.raw"],
    )

    with pytest.raises(
        ValueError,
        match="metadata could not be matched to psm\\.obs using metadata key 'index' and obs key 'index'",
    ):
        mm.pp.add_meta(mdata, metadata)

    assert "condition" not in mdata.mod["psm"].obs.columns


def test_add_meta_matches_dataframe_input_by_index_only_with_exact_values():
    mdata = _make_mdata(["run_1.raw", "run_2.raw"])
    metadata = pd.DataFrame(
        {
            "condition": ["treated", "control"],
            "batch": ["B1", "B2"],
        },
        index=["run_1.raw", "run_2.raw"],
    )

    out = mm.pp.add_meta(mdata, metadata)

    assert out.obs["condition"].tolist() == ["treated", "control"]
    assert out.obs["batch"].tolist() == ["B1", "B2"]
    assert "meta" not in out.uns


@pytest.mark.parametrize(
    ("suffix", "writer"),
    [
        (
            ".csv",
            lambda frame, path: frame.to_csv(path, index=False),
        ),
        (
            ".tsv",
            lambda frame, path: frame.to_csv(path, sep="\t", index=False),
        ),
    ],
)
def test_add_meta_loads_path_like_tabular_sources(tmp_path, suffix, writer):
    mdata = _make_mdata(["run_1", "run_2"])
    metadata = pd.DataFrame(
        {
            "run": ["run_1", "run_2"],
            "condition": ["treated", "control"],
        }
    )
    path = tmp_path / f"metadata{suffix}"
    writer(metadata, path)

    out = mm.pp.add_meta(mdata, Path(path), metadata_on="run")

    assert out.obs["condition"].tolist() == ["treated", "control"]
    assert "meta" not in out.uns


def test_add_meta_loads_url_like_tsv_source(monkeypatch):
    mdata = _make_mdata(["run_1"])
    url = "https://example.test/metadata.tsv"
    opened: list[str] = []

    def fake_urlopen(source, **kwargs):
        opened.append(source)
        return io.BytesIO(b"run\tcondition\nrun_1\tcase\n")

    from msmu._core import _sources

    monkeypatch.setattr(_sources, "urlopen", fake_urlopen)

    out = mm.pp.add_meta(mdata, url, metadata_on="run")

    assert out.obs["condition"].tolist() == ["case"]
    assert opened == [url]
    assert "meta" not in out.uns


@pytest.mark.parametrize(("suffix", "separator"), [(".csv", ","), (".tsv", "\t")])
def test_add_meta_preserves_decimal_precision_from_text(tmp_path, suffix, separator):
    path = tmp_path / f"metadata{suffix}"
    path.write_text(
        f"run{separator}value\n"
        f"run_1{separator}3726.3721923828125\n"
        f"run_2{separator}2927.7894287109375\n"
    )

    out = mm.pp.add_meta(_make_mdata(["run_1", "run_2"]), path, metadata_on="run")

    values = out.obs["value"].to_numpy()
    assert values.dtype == np.dtype("float64")
    np.testing.assert_array_equal(values, [3726.3721923828125, 2927.7894287109375])


@pytest.mark.parametrize(("suffix", "separator"), [(".csv", ","), (".tsv", "\t")])
def test_add_meta_preserves_default_missing_value_tokens(tmp_path, suffix, separator):
    path = tmp_path / f"metadata{suffix}"
    path.write_text(
        f"run{separator}value\n"
        + "".join(f"run_{i}{separator}{value}\n" for i, value in enumerate(["NA", "NaN", "null", ""]))
    )

    out = mm.pp.add_meta(_make_mdata([f"run_{i}" for i in range(4)]), path, metadata_on="run")

    assert out.obs["value"].isna().all()


@pytest.mark.parametrize("suffix", [".csv", ".tsv"])
def test_add_meta_reports_empty_tabular_source(tmp_path, suffix):
    path = tmp_path / f"metadata{suffix}"
    path.write_text("")

    with pytest.raises(ValueError, match="is empty"):
        mm.pp.add_meta(_make_mdata(["run_1"]), path, metadata_on="run")


def test_add_meta_loads_parquet_sources(tmp_path):
    mdata = _make_mdata(["run_1"])
    metadata = pd.DataFrame({"run": ["run_1"], "condition": ["treated"]})
    path = tmp_path / "metadata.parquet"

    try:
        metadata.to_parquet(path, index=False)
    except ImportError:
        pytest.skip("Parquet engine unavailable")

    out = mm.pp.add_meta(mdata, path, metadata_on="run")

    assert out.obs["condition"].tolist() == ["treated"]
    assert "meta" not in out.uns


def test_add_meta_can_join_on_explicit_metadata_column_to_obs_index():
    mdata = _make_mdata(["fileA.raw", "fileB.raw"])
    metadata = pd.DataFrame(
        {
            "run_file": ["fileA.raw", "fileB.raw"],
            "condition": ["control", "case"],
        }
    )

    out = mm.pp.add_meta(mdata, metadata, metadata_on="run_file")

    assert out.obs["condition"].tolist() == ["control", "case"]


def test_add_meta_can_join_on_explicit_metadata_index_to_named_obs_column():
    mdata = _make_mdata(["sample_a", "sample_b"])
    mdata.mod["psm"].obs["run"] = ["fileA.raw", "fileB.raw"]
    metadata = pd.DataFrame(
        {"condition": ["control", "case"]},
        index=["fileA.raw", "fileB.raw"],
    )

    out = mm.pp.add_meta(mdata, metadata, metadata_on="index", obs_columns="run")

    assert out.obs["condition"].tolist() == ["control", "case"]


def test_add_meta_defaults_obs_matching_to_index_until_obs_columns_is_set():
    mdata = _make_mdata(["sample_a"])
    mdata.mod["psm"].obs["run"] = ["run_1"]
    metadata = pd.DataFrame({"run": ["run_1"], "condition": ["control"]})

    with pytest.raises(
        ValueError,
        match="metadata could not be matched to psm\\.obs using metadata key 'run' and obs key 'index'",
    ):
        mm.pp.add_meta(mdata, metadata, metadata_on="run")


def test_add_meta_rejects_duplicate_metadata_keys():
    mdata = _make_mdata(["run_1"])
    metadata = pd.DataFrame(
        {
            "run": ["run_1", "run_1"],
            "condition": ["control", "case"],
        }
    )

    with pytest.raises(ValueError, match="metadata key 'run' must be unique"):
        mm.pp.add_meta(mdata, metadata, metadata_on="run")
