import io
import logging

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData

import msmu as mm
from msmu._preprocessing import _meta as meta_module
from msmu._preprocessing import add_meta
from msmu._read_write import _reader_registry as rr
from msmu._tools import _sdrf_pipelines as sdrf_tools

_PANDAS_READ_CSV = pd.read_csv


def _make_mdata(obs_index: list[str]) -> MuData:
    obs = pd.DataFrame(index=obs_index)
    var = pd.DataFrame(index=["feature_1"])
    x = np.ones((len(obs_index), 1))
    return MuData({"psm": AnnData(X=x, obs=obs, var=var)})


def _write_sdrf(tmp_path, content: str):
    path = tmp_path / "sample.sdrf.tsv"
    path.write_text(content)
    return path


def _last_cmd_entry(mdata: MuData) -> dict[str, object]:
    key = max(mdata.uns["_cmd"], key=lambda value: int(value))
    return mdata.uns["_cmd"][key]


class _FakeValidationError:
    def __init__(self, message: str, error_type: int | None):
        self.message = message
        self.error_type = error_type


def test_read_sdrf_preserves_pandas_headers_and_duplicate_suffixes(tmp_path):
    path = _write_sdrf(
        tmp_path,
        "\t".join(
            [
                "source name",
                "assay name",
                "technology type",
                "characteristics[organism]",
                "characteristics[organism]",
                "comment[data file]",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "sample_1",
                "assay_1",
                "mass spectrometry",
                "Homo sapiens",
                "human",
                "run_1.raw",
            ]
        )
        + "\n",
    )

    metadata = mm.read_sdrf(path, validate_sdrf=False)

    assert metadata.columns.tolist() == [
        "source name",
        "assay name",
        "technology type",
        "characteristics[organism]",
        "characteristics[organism].1",
        "comment[data file]",
    ]
    assert metadata.loc[0, "characteristics[organism]"] == "Homo sapiens"
    assert metadata.loc[0, "characteristics[organism].1"] == "human"


def test_registry_read_sdrf_exposes_public_parser(tmp_path):
    path = _write_sdrf(
        tmp_path,
        "\t".join(["source name", "assay name", "technology type"])
        + "\n"
        + "\t".join(["sample_1", "assay_1", "mass spectrometry"])
        + "\n",
    )

    metadata = rr.read_sdrf(path, validate_sdrf=False)

    assert metadata.loc[0, "source name"] == "sample_1"
    assert metadata.loc[0, "assay name"] == "assay_1"


def test_read_sdrf_reads_url_like_sources_via_tabular_ingestion(monkeypatch):
    url = "https://example.test/sample.sdrf.tsv"
    content = (
        "\t".join(["source name", "assay name", "technology type"])
        + "\n"
        + "\t".join(["sample_1", "assay_1", "mass spectrometry"])
        + "\n"
    )
    opened: list[dict[str, object]] = []

    def fake_read_csv(source, *args, **kwargs):
        opened.append({"source": source, "kwargs": kwargs})
        if source == url:
            return _PANDAS_READ_CSV(io.StringIO(content), *args, **kwargs)
        return _PANDAS_READ_CSV(source, *args, **kwargs)

    monkeypatch.setattr(meta_module.pd, "read_csv", fake_read_csv)

    metadata = mm.read_sdrf(url, validate_sdrf=False)

    assert metadata.loc[0, "source name"] == "sample_1"
    assert metadata.attrs["sdrf_file"] == url
    assert opened[0]["source"] == url
    assert opened[0]["kwargs"]["sep"] == "\t"
    assert "header" not in opened[0]["kwargs"]


def test_add_meta_sdrf_validates_through_pipeline_and_only_logs_validation_result(monkeypatch, tmp_path):
    path = _write_sdrf(
        tmp_path,
        "\t".join(["source name", "assay name", "technology type", "comment[data file]"])
        + "\n"
        + "\t".join(["sample_1", "assay_1", "mass spectrometry", "run_1.raw"])
        + "\n",
    )
    mdata = _make_mdata(["run_1.raw"])
    validated: list[dict[str, object]] = []

    def fake_validate(sdrf, *, source=None, template="ms-proteomics", skip_ontology=True):
        validated.append(
            {
                "sdrf": sdrf,
                "source": source,
                "template": template,
                "skip_ontology": skip_ontology,
            }
        )

    monkeypatch.setattr(sdrf_tools, "validate_sdrf_dataframe", fake_validate)

    out = add_meta(mdata, path, format="sdrf", metadata_on="comment[data file]")

    assert out.obs["source name"].tolist() == ["sample_1"]
    assert len(validated) == 1
    assert isinstance(validated[0]["sdrf"], pd.DataFrame)
    assert validated[0]["sdrf"].columns.tolist() == [
        "source name",
        "assay name",
        "technology type",
        "comment[data file]",
    ]
    assert validated[0]["source"] == str(path)
    assert "meta" not in out.uns
    stdout = _last_cmd_entry(out)["stdout"]
    assert "Validating SDRF metadata for" in stdout
    assert "SDRF validation succeeded for" in stdout


def test_add_meta_sdrf_raw_dataframe_keeps_headers_before_matching(monkeypatch, tmp_path):
    path = _write_sdrf(
        tmp_path,
        "\t".join(
            [
                "source name",
                "assay name",
                "technology type",
                "comment[data file]",
                "factor value[condition]",
            ]
        )
        + "\n"
        + "\t".join(["sample_1", "assay_1", "mass spectrometry", "run_1.raw", "treated"])
        + "\n",
    )
    metadata = pd.read_csv(path, sep="\t")
    mdata = _make_mdata(["run_1.raw"])
    validated: list[dict[str, object]] = []

    def fake_validate_sdrf_file(sdrf_file, *, skip_ontology, source_name=None):
        validated.append(
            {
                "sdrf_file": sdrf_file,
                "skip_ontology": skip_ontology,
                "source_name": source_name,
            }
        )

    monkeypatch.setattr(meta_module, "validate_sdrf_file", fake_validate_sdrf_file)

    out = add_meta(mdata, metadata, format="sdrf", metadata_on="comment[data file]")

    assert out.obs["source name"].tolist() == ["sample_1"]
    assert out.obs["factor value[condition]"].tolist() == ["treated"]
    assert len(validated) == 1
    assert isinstance(validated[0]["sdrf_file"], pd.DataFrame)
    assert validated[0]["sdrf_file"].columns.tolist() == [
        "source name",
        "assay name",
        "technology type",
        "comment[data file]",
        "factor value[condition]",
    ]
    assert validated[0]["source_name"] == "DataFrame"


def test_add_meta_sdrf_defaults_to_metadata_index_when_metadata_on_is_omitted(tmp_path):
    path = _write_sdrf(
        tmp_path,
        "\t".join(["source name", "assay name", "technology type", "comment[data file]"])
        + "\n"
        + "\t".join(["sample_1", "assay_1", "mass spectrometry", "run_1.raw"])
        + "\n",
    )
    mdata = _make_mdata(["run_1"])

    with pytest.raises(
        ValueError,
        match="metadata could not be matched to psm\\.obs using metadata key 'index' and obs key 'index'",
    ):
        add_meta(mdata, path, format="sdrf", validate_sdrf=False)


def test_add_meta_sdrf_can_target_named_obs_column(tmp_path):
    path = _write_sdrf(
        tmp_path,
        "\t".join(
            [
                "source name",
                "assay name",
                "technology type",
                "comment[data file]",
                "factor value[condition]",
            ]
        )
        + "\n"
        + "\t".join(["sample_1", "assay_1", "mass spectrometry", "fileA.raw", "treated"])
        + "\n"
        + "\t".join(["sample_2", "assay_2", "mass spectrometry", "fileB.raw", "control"])
        + "\n",
    )
    mdata = _make_mdata(["sample_a", "sample_b"])
    mdata.mod["psm"].obs["run"] = ["fileA.raw", "fileB.raw"]

    out = add_meta(
        mdata,
        path,
        format="sdrf",
        metadata_on="comment[data file]",
        obs_columns="run",
        validate_sdrf=False,
    )

    assert out.obs["source name"].tolist() == ["sample_1", "sample_2"]
    assert out.obs["factor value[condition]"].tolist() == ["treated", "control"]


def test_tools_validate_sdrf_dataframe_uses_schema_validator(monkeypatch):
    schemas = pytest.importorskip("sdrf_pipelines.sdrf.schemas")
    sdrf_pkg = pytest.importorskip("sdrf_pipelines.sdrf.sdrf")
    calls: dict[str, object] = {}

    class FakeValidator:
        def __init__(self, registry):
            calls["registry"] = registry

        def validate(self, sdrf, template, use_ols_cache_only=False, skip_ontology=False):
            calls["sdrf"] = sdrf
            calls["template"] = template
            calls["use_ols_cache_only"] = use_ols_cache_only
            calls["skip_ontology"] = skip_ontology
            return []

    monkeypatch.setattr(sdrf_tools.metadata, "version", lambda package: "0.0")
    monkeypatch.setattr(schemas, "SchemaRegistry", lambda: "registry")
    monkeypatch.setattr(schemas, "SchemaValidator", FakeValidator)

    sdrf_tools.validate_sdrf_dataframe(
        pd.DataFrame({"source name": ["sample_1"], "assay name": ["assay_1"]}),
        source="inline",
    )

    assert isinstance(calls["sdrf"], sdrf_pkg.SDRFDataFrame)
    assert calls["sdrf"].columns == ["source name", "assay name"]
    assert calls["template"] == "ms-proteomics"
    assert calls["use_ols_cache_only"] is False
    assert calls["skip_ontology"] is True


def test_tools_validate_sdrf_dataframe_raises_for_pipeline_errors(monkeypatch):
    schemas = pytest.importorskip("sdrf_pipelines.sdrf.schemas")

    class FakeValidator:
        def __init__(self, registry):
            self.registry = registry

        def validate(self, sdrf, template, use_ols_cache_only=False, skip_ontology=False):
            return [
                _FakeValidationError("missing source name", logging.ERROR),
                _FakeValidationError("missing source name", logging.ERROR),
                _FakeValidationError("warning only", logging.WARNING),
            ]

    monkeypatch.setattr(sdrf_tools.metadata, "version", lambda package: "0.0")
    monkeypatch.setattr(schemas, "SchemaRegistry", lambda: object())
    monkeypatch.setattr(schemas, "SchemaValidator", FakeValidator)

    with pytest.raises(
        ValueError,
        match="SDRF validation failed for sample.sdrf.tsv: missing source name",
    ):
        sdrf_tools.validate_sdrf_dataframe(
            pd.DataFrame({"source name": ["sample_1"], "assay name": ["assay_1"]}),
            source="sample.sdrf.tsv",
            skip_ontology=False,
        )
