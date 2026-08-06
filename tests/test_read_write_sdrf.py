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


_SDRF_MIN = (
    "source name\tcharacteristics[organism]\tcomment[data file]\tcomment[label]\n"
    "t0h\tHomo sapiens\trun1.mzML\tTMT126\n"
    "t1h\tHomo sapiens\trun1.mzML\tTMT127\n"
)


def test_attach_sdrf_stores_dataframe_in_uns_without_touching_obs():
    sdrf = pd.DataFrame({"source name": ["t0h", "t1h"], "comment[label]": ["TMT126", "TMT127"]})
    mdata = _make_mdata(["TMT126", "TMT127"])

    out = mm.pp.attach_sdrf(mdata, sdrf, validate=False)

    stored = out.uns["sdrf"]
    assert isinstance(stored, pd.DataFrame)
    assert list(stored.columns) == ["source name", "comment[label]"]
    assert stored.shape == (2, 2)
    # obs is left untouched -- the SDRF lives only in uns
    assert list(out.mod["psm"].obs.columns) == []
    # the input mdata is not mutated (attach copies)
    assert "sdrf" not in mdata.uns


def test_attach_sdrf_reads_path_and_stores_whole_table(tmp_path):
    path = _write_sdrf(tmp_path, _SDRF_MIN)

    out = mm.pp.attach_sdrf(_make_mdata(["TMT126", "TMT127"]), path, validate=False)

    stored = out.uns["sdrf"]
    assert stored.shape == (2, 4)
    assert list(stored.columns)[0] == "source name"
    assert stored.loc[0, "comment[label]"] == "TMT126"


def test_attach_sdrf_survives_h5mu_roundtrip(tmp_path):
    import mudata

    path = _write_sdrf(tmp_path, _SDRF_MIN)
    out = mm.pp.attach_sdrf(_make_mdata(["TMT126", "TMT127"]), path, validate=False)

    h5 = tmp_path / "attached.h5mu"
    out.write_h5mu(str(h5))
    restored = mudata.read_h5mu(str(h5)).uns["sdrf"]

    assert isinstance(restored, pd.DataFrame)
    assert restored.shape == (2, 4)
    assert list(restored.columns) == [
        "source name",
        "characteristics[organism]",
        "comment[data file]",
        "comment[label]",
    ]
    assert restored.loc[1, "comment[label]"] == "TMT127"


def test_attach_sdrf_validates_when_requested(monkeypatch):
    seen = {}

    def fake_validate(dataframe, **kwargs):
        seen["called"] = True

    monkeypatch.setattr(meta_module, "validate_sdrf_file", fake_validate)

    mm.pp.attach_sdrf(
        _make_mdata(["TMT126"]),
        pd.DataFrame({"source name": ["t0h"], "comment[label]": ["TMT126"]}),
        validate=True,
    )

    assert seen.get("called")


def test_attach_sdrf_rejects_non_mudata():
    with pytest.raises(TypeError, match="MuData"):
        mm.pp.attach_sdrf(pd.DataFrame({"a": [1]}), pd.DataFrame({"source name": ["t0h"]}), validate=False)


_SDRF_TMT_FRAC = pd.DataFrame(
    {
        "comment[label]": ["TMT126", "TMT127", "TMT126", "TMT127"],
        "source name": ["t0h", "t1h", "t0h", "t1h"],
        "comment[fraction identifier]": ["1", "1", "2", "2"],  # varies within a channel
    }
)


def _tmt_mdata_with_sdrf(sdrf: pd.DataFrame | None = None) -> MuData:
    mdata = _make_mdata(["TMT126", "TMT127"])
    mdata.mod["psm"].uns["label"] = "tmt"
    return mm.pp.attach_sdrf(mdata, _SDRF_TMT_FRAC if sdrf is None else sdrf, validate=False)


def test_apply_sdrf_projects_functional_columns_and_skips_varying():
    obs = mm.pp.apply_sdrf_to_obs(_tmt_mdata_with_sdrf()).mod["psm"].obs
    # one source name per channel -> projected
    assert list(obs["source name"]) == ["t0h", "t1h"]
    # fraction varies within a channel -> not projectable, stays only in uns
    assert "comment[fraction identifier]" not in obs.columns


def test_apply_sdrf_auto_key_tmt_vs_label_free():
    tmt = mm.pp.apply_sdrf_to_obs(_tmt_mdata_with_sdrf())
    assert list(tmt.mod["psm"].obs["source name"]) == ["t0h", "t1h"]

    sdrf_lf = pd.DataFrame(
        {"comment[data file]": ["runA.mzML", "runB.mzML"], "source name": ["ctrl", "case"]}
    )
    m = mm.pp.attach_sdrf(_make_mdata(["runA.mzML", "runB.mzML"]), sdrf_lf, validate=False)
    lf = mm.pp.apply_sdrf_to_obs(m)  # no uns label -> defaults to comment[data file]
    assert list(lf.mod["psm"].obs["source name"]) == ["ctrl", "case"]


def test_apply_sdrf_set_index_replaces_obs_index():
    out = mm.pp.apply_sdrf_to_obs(_tmt_mdata_with_sdrf(), set_index="source name")
    assert list(out.mod["psm"].obs.index) == ["t0h", "t1h"]


def test_apply_sdrf_named_nonfunctional_column_raises():
    with pytest.raises(ValueError, match="not a function"):
        mm.pp.apply_sdrf_to_obs(_tmt_mdata_with_sdrf(), columns="comment[fraction identifier]")


def test_apply_sdrf_set_index_nonunique_raises():
    sdrf = pd.DataFrame({"comment[label]": ["TMT126", "TMT127"], "source name": ["dup", "dup"]})
    with pytest.raises(ValueError, match="not unique"):
        mm.pp.apply_sdrf_to_obs(_tmt_mdata_with_sdrf(sdrf), set_index="source name")


def test_apply_sdrf_requires_attached_sdrf():
    with pytest.raises(ValueError, match="No SDRF attached"):
        mm.pp.apply_sdrf_to_obs(_make_mdata(["TMT126"]))


def test_apply_sdrf_leaves_uns_sdrf_unchanged():
    mdata = _tmt_mdata_with_sdrf()
    out = mm.pp.apply_sdrf_to_obs(mdata)
    assert out.uns["sdrf"].shape == (4, 3)
    assert "comment[fraction identifier]" in out.uns["sdrf"].columns


def test_apply_sdrf_populates_mdata_level_obs():
    # projected columns must reach the MuData-level obs, not only the modality obs
    # (consumers such as correct_batch_effect read mdata.obs)
    out = mm.pp.apply_sdrf_to_obs(_tmt_mdata_with_sdrf())
    assert "source name" in out.obs.columns
    assert list(out.obs["source name"]) == ["t0h", "t1h"]


def test_apply_sdrf_warns_on_unmatched_obs_keys(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        meta_module.logger,
        "warning",
        lambda message, *args, **kwargs: seen.append(message % args if args else message),
    )
    sdrf = pd.DataFrame({"comment[label]": ["TMT126", "TMT127"], "source name": ["t0h", "t1h"]})
    mdata = _make_mdata(["TMT126", "TMT999"])  # TMT999 is absent from the SDRF
    mdata.mod["psm"].uns["label"] = "tmt"
    mdata = mm.pp.attach_sdrf(mdata, sdrf, validate=False)

    out = mm.pp.apply_sdrf_to_obs(mdata)

    assert any("absent from SDRF" in message for message in seen)
    assert out.mod["psm"].obs["source name"].iloc[0] == "t0h"
    assert pd.isna(out.mod["psm"].obs["source name"].iloc[1])
