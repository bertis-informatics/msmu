import io
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData

from msmu._read_write import _sdrf as sdrf_module
from msmu._read_write._sdrf import attach_sdrf_metadata, read_sdrf
from msmu._read_write import _reader_registry as rr


def _write_sdrf(tmp_path, content: str):
    path = tmp_path / "sample.sdrf.tsv"
    path.write_text(content)
    return path


def _fractionated_sdrf_content() -> tuple[list[str], list[list[str]]]:
    headers = [
        "source name",
        "sample name",
        "assay name",
        "technology type",
        "comment[data file]",
        "comment[fraction identifier]",
        "factor value[condition]",
    ]
    rows = [
        ["source_1", "sample_1", "assay_1", "mass spectrometry", "sample_1_fraction_1.raw", "1", "treated"],
        ["source_1", "sample_1", "assay_1", "mass spectrometry", "sample_1_fraction_2.raw", "2", "treated"],
        ["source_2", "sample_2", "assay_2", "mass spectrometry", "sample_2_fraction_1.raw", "1", "control"],
        ["source_2", "sample_2", "assay_2", "mass spectrometry", "sample_2_fraction_2.raw", "2", "control"],
    ]
    return headers, rows


def _write_fractionated_sdrf(tmp_path):
    headers, rows = _fractionated_sdrf_content()
    content = "\t".join(headers) + "\n" + "\n".join("\t".join(row) for row in rows) + "\n"
    return _write_sdrf(tmp_path, content)


def _collapsed_values(value: object) -> list[str]:
    if isinstance(value, (pd.Index, pd.Series, np.ndarray)):
        return [str(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]

    text = str(value)
    for separator in (";", "|", ","):
        if separator in text:
            return [part.strip() for part in text.split(separator) if part.strip()]
    return [text]


def _assert_collapsed_values(value: object, expected: list[str]) -> None:
    assert _collapsed_values(value) == expected


def _make_mdata(obs_index: list[str]) -> MuData:
    obs = pd.DataFrame(index=obs_index)
    var = pd.DataFrame(index=["feature_1"])
    x = np.ones((len(obs_index), 1))
    return MuData({"psm": AnnData(X=x, obs=obs, var=var)})


def _write_diann_report(tmp_path):
    path = tmp_path / "report.tsv"
    report = pd.DataFrame(
        {
            "Protein.Ids": ["sp|P1|P1_HUMAN", "sp|P2|P2_HUMAN"],
            "Protein.Group": ["P1", "P2"],
            "Modified.Sequence": ["PEPTIDEK", "ACDMK"],
            "Stripped.Sequence": ["PEPTIDEK", "ACDMK"],
            "Run": ["fileA.raw", "fileB.raw"],
            "Precursor.Charge": [2, 3],
            "Lib.Q.Value": [0.0, 0.0],
            "Global.Q.Value": [0.01, 0.02],
            "RT": [12.3, 18.4],
            "PEP": [0.001, 0.002],
            "Precursor.Id": ["PEPTIDEK2", "ACDMK3"],
            "Precursor.Quantity": [1000.0, 2000.0],
        }
    )
    report.to_csv(path, sep="\t", index=False)
    return path


class _FakeUrlResponse:
    def __init__(self, content: bytes, *, status: int = 200, headers: dict[str, str] | None = None):
        self._content = io.BytesIO(content)
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._content.close()

    def read(self, size=-1):
        return self._content.read(size)


def _dispatching_parse_sdrf_entrypoint():
    globals()["cli"]()


def test_read_sdrf_preserves_duplicate_supported_headers(tmp_path):
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
                "factor value[condition]",
            ]
        )
        + "\n"
        + "\t".join(["sample_1", "assay_1", "mass spectrometry", "Homo sapiens", "human", "run_1.raw", "treated"])
        + "\n",
    )

    metadata = read_sdrf(path, validate=False)

    assert metadata.columns.tolist() == [
        "source_name",
        "assay_name",
        "technology_type",
        "characteristics_organism",
        "characteristics_organism_2",
        "comment_data_file",
        "factor_value_condition",
    ]
    assert metadata.loc[0, "characteristics_organism"] == "Homo sapiens"
    assert metadata.loc[0, "characteristics_organism_2"] == "human"
    assert metadata.attrs["sdrf_columns"][4]["occurrence"] == 2


def test_attach_sdrf_metadata_matches_obs_by_data_file_stem(tmp_path):
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
        + "\n"
        + "\t".join(["sample_2", "assay_2", "mass spectrometry", "run_2.raw", "control"])
        + "\n",
    )
    mdata = _make_mdata(["run_1", "run_2"])

    out = attach_sdrf_metadata(mdata, path, validate=False)

    assert out.mod["psm"].obs["source_name"].tolist() == ["sample_1", "sample_2"]
    assert out.mod["psm"].obs["factor_value_condition"].tolist() == ["treated", "control"]
    assert out.obs["factor_value_condition"].tolist() == ["treated", "control"]
    assert out.uns["sdrf"]["matched_modalities"]["psm"]["matched_column"] == "comment_data_file"


def test_attach_sdrf_metadata_collapses_fractionated_rows_by_sample(tmp_path):
    path = _write_fractionated_sdrf(tmp_path)
    mdata = _make_mdata(["sample_1", "sample_2"])

    out = attach_sdrf_metadata(mdata, path, validate=False)

    assert out.mod["psm"].obs.index.tolist() == ["sample_1", "sample_2"]
    assert out.obs.index.tolist() == ["sample_1", "sample_2"]
    assert out.mod["psm"].obs["source_name"].tolist() == ["source_1", "source_2"]
    assert out.mod["psm"].obs["sample_name"].tolist() == ["sample_1", "sample_2"]
    assert out.obs["factor_value_condition"].tolist() == ["treated", "control"]
    _assert_collapsed_values(
        out.mod["psm"].obs.loc["sample_1", "comment_data_file"],
        ["sample_1_fraction_1.raw", "sample_1_fraction_2.raw"],
    )
    _assert_collapsed_values(out.mod["psm"].obs.loc["sample_1", "comment_fraction_identifier"], ["1", "2"])
    _assert_collapsed_values(
        out.obs.loc["sample_2", "comment_data_file"],
        ["sample_2_fraction_1.raw", "sample_2_fraction_2.raw"],
    )
    _assert_collapsed_values(out.obs.loc["sample_2", "comment_fraction_identifier"], ["1", "2"])
    assert out.uns["sdrf"]["matched_modalities"]["psm"]["matched_obs"] == 2


def test_attach_sdrf_metadata_stores_original_sdrf_table(tmp_path):
    path = _write_fractionated_sdrf(tmp_path)
    headers, rows = _fractionated_sdrf_content()
    mdata = _make_mdata(["sample_1_fraction_1", "sample_1_fraction_2", "sample_2_fraction_1", "sample_2_fraction_2"])

    out = attach_sdrf_metadata(mdata, path, validate=False)

    stored = out.uns["sdrf"]["table"]
    assert isinstance(stored, pd.DataFrame)
    assert stored.shape == (len(rows), len(headers))
    assert stored.columns.tolist() == headers
    assert stored["comment[data file]"].tolist() == [row[4] for row in rows]
    assert stored["sample name"].tolist() == [row[1] for row in rows]


def test_attach_sdrf_metadata_matches_tmt_obs_by_label(tmp_path):
    path = _write_sdrf(
        tmp_path,
        "\t".join(
            [
                "source name",
                "assay name",
                "technology type",
                "comment[label]",
                "factor value[condition]",
            ]
        )
        + "\n"
        + "\t".join(["sample_1", "assay_1", "mass spectrometry", "TMT126", "treated"])
        + "\n"
        + "\t".join(["sample_2", "assay_2", "mass spectrometry", "TMT127N", "control"])
        + "\n",
    )
    mdata = _make_mdata(["126", "127N"])

    out = attach_sdrf_metadata(mdata, path, validate=False)

    assert out.mod["psm"].obs["source_name"].tolist() == ["sample_1", "sample_2"]
    assert out.mod["psm"].obs["factor_value_condition"].tolist() == ["treated", "control"]
    assert out.uns["sdrf"]["matched_modalities"]["psm"]["matched_column"] == "comment_label"


def test_attach_sdrf_metadata_raises_when_no_obs_match(tmp_path):
    path = _write_sdrf(
        tmp_path,
        "\t".join(["source name", "assay name", "technology type", "comment[data file]"])
        + "\n"
        + "\t".join(["sample_1", "assay_1", "mass spectrometry", "run_1.raw"])
        + "\n",
    )
    mdata = _make_mdata(["missing_run"])

    with pytest.raises(ValueError, match="could not be matched"):
        attach_sdrf_metadata(mdata, path, validate=False)


def test_registry_read_sdrf_exposes_parser(tmp_path):
    path = _write_sdrf(
        tmp_path,
        "\t".join(["source name", "assay name", "technology type"])
        + "\n"
        + "\t".join(["sample_1", "assay_1", "mass spectrometry"])
        + "\n",
    )

    metadata = rr.read_sdrf(path, validate_sdrf=False)

    assert metadata.loc[0, "source_name"] == "sample_1"


def test_registry_read_sdrf_accepts_sdrf_file_keyword(tmp_path):
    path = _write_sdrf(
        tmp_path,
        "\t".join(["source name", "assay name", "technology type"])
        + "\n"
        + "\t".join(["sample_1", "assay_1", "mass spectrometry"])
        + "\n",
    )

    metadata = rr.read_sdrf(sdrf_file=path, validate_sdrf=False)

    assert metadata.loc[0, "assay_name"] == "assay_1"


def test_registry_read_sdrf_rejects_removed_sdrf_path_alias(tmp_path):
    with pytest.raises(TypeError, match="sdrf_path"):
        rr.read_sdrf(sdrf_path=tmp_path / "sample.sdrf.tsv")


def test_registry_read_sdrf_validates_by_default_with_click_group(tmp_path, monkeypatch):
    click = pytest.importorskip("click")
    path = _write_sdrf(
        tmp_path,
        "\t".join(["source name", "assay name", "technology type"])
        + "\n"
        + "\t".join(["sample_1", "assay_1", "mass spectrometry"])
        + "\n",
    )
    calls = []

    @click.group()
    def cli():
        pass

    @cli.command("validate-sdrf")
    @click.option("--sdrf_file")
    @click.option("--skip-ontology", is_flag=True)
    def validate_sdrf(sdrf_file, skip_ontology):
        calls.append((sdrf_file, skip_ontology))

    entrypoint = types.FunctionType(_dispatching_parse_sdrf_entrypoint.__code__, {"cli": cli})
    monkeypatch.setattr(sdrf_module.metadata, "version", lambda package: "0.0")
    monkeypatch.setattr(sdrf_module, "_load_parse_sdrf_entrypoint", lambda: entrypoint)

    metadata = rr.read_sdrf(path)

    assert metadata.loc[0, "source_name"] == "sample_1"
    assert calls == [(str(path), True)]


def test_read_sdrf_url_materializes_local_file_for_validation(monkeypatch):
    click = pytest.importorskip("click")
    url = "https://example.test/sample.sdrf.tsv"
    content = (
        "\t".join(["source name", "assay name", "technology type"])
        + "\n"
        + "\t".join(["sample_1", "assay_1", "mass spectrometry"])
        + "\n"
    ).encode()
    opened = []
    calls = []

    def urlopen(opened_url, *, timeout):
        opened.append((opened_url, timeout))
        return _FakeUrlResponse(content, headers={"Content-Length": str(len(content))})

    @click.group()
    def cli():
        pass

    @cli.command("validate-sdrf")
    @click.option("--sdrf_file")
    @click.option("--skip-ontology", is_flag=True)
    def validate_sdrf(sdrf_file, skip_ontology):
        path = Path(sdrf_file)
        calls.append(
            {
                "sdrf_file": sdrf_file,
                "skip_ontology": skip_ontology,
                "exists": path.exists(),
                "name": path.name,
            }
        )

    entrypoint = types.FunctionType(_dispatching_parse_sdrf_entrypoint.__code__, {"cli": cli})
    monkeypatch.setattr(sdrf_module.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(sdrf_module.metadata, "version", lambda package: "0.0")
    monkeypatch.setattr(sdrf_module, "_load_parse_sdrf_entrypoint", lambda: entrypoint)

    metadata = read_sdrf(url)

    assert metadata.loc[0, "source_name"] == "sample_1"
    assert metadata.attrs["sdrf_file"] == url
    assert opened == [(url, sdrf_module._SDRF_URL_TIMEOUT_SECONDS)]
    assert len(calls) == 1
    assert calls[0]["skip_ontology"] is True
    assert calls[0]["exists"] is True
    assert calls[0]["name"] == "remote.sdrf.tsv"
    assert not calls[0]["sdrf_file"].startswith("http")


def test_attach_sdrf_metadata_url_keeps_original_url_in_uns(monkeypatch):
    url = "https://example.test/sample.sdrf.tsv"
    content = (
        "\t".join(["source name", "assay name", "technology type", "comment[data file]"])
        + "\n"
        + "\t".join(["sample_1", "assay_1", "mass spectrometry", "run_1.raw"])
        + "\n"
    ).encode()

    def urlopen(opened_url, *, timeout):
        assert opened_url == url
        assert timeout == sdrf_module._SDRF_URL_TIMEOUT_SECONDS
        return _FakeUrlResponse(content, headers={"Content-Length": str(len(content))})

    monkeypatch.setattr(sdrf_module.urllib.request, "urlopen", urlopen)
    mdata = _make_mdata(["run_1"])

    out = attach_sdrf_metadata(mdata, url, validate=False)

    assert out.uns["sdrf"]["path"] == url
    assert out.mod["psm"].uns["sdrf"]["path"] == url
    assert out.mod["psm"].obs["source_name"].tolist() == ["sample_1"]


def test_read_sdrf_rejects_unsupported_url_scheme():
    with pytest.raises(ValueError, match="Unsupported SDRF URL scheme 'ftp'"):
        read_sdrf("ftp://example.test/sample.sdrf.tsv", validate=False)


def test_read_sdrf_url_rejects_download_larger_than_limit(monkeypatch):
    url = "https://example.test/large.sdrf.tsv"

    def urlopen(opened_url, *, timeout):
        assert opened_url == url
        return _FakeUrlResponse(b"", headers={"Content-Length": "11"})

    monkeypatch.setattr(sdrf_module.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(sdrf_module, "_MAX_SDRF_DOWNLOAD_BYTES", 10)

    with pytest.raises(ValueError, match="exceeds maximum download size"):
        read_sdrf(url, validate=False)


def test_validate_sdrf_file_accepts_function_entrypoint_returning_click_command(tmp_path, monkeypatch):
    click = pytest.importorskip("click")
    path = _write_sdrf(
        tmp_path,
        "\t".join(["source name", "assay name", "technology type"])
        + "\n"
        + "\t".join(["sample_1", "assay_1", "mass spectrometry"])
        + "\n",
    )
    calls = []

    @click.group()
    def cli():
        pass

    @cli.command("validate-sdrf")
    @click.option("--sdrf_file")
    @click.option("--skip-ontology", is_flag=True)
    def validate_sdrf(sdrf_file, skip_ontology):
        calls.append((sdrf_file, skip_ontology))

    monkeypatch.setattr(sdrf_module.metadata, "version", lambda package: "0.0")
    monkeypatch.setattr(sdrf_module, "_load_parse_sdrf_entrypoint", lambda: lambda: cli)

    sdrf_module.validate_sdrf_file(path)

    assert calls == [(str(path), True)]


def test_validate_sdrf_file_falls_back_to_parse_sdrf_executable(tmp_path, monkeypatch):
    path = _write_sdrf(
        tmp_path,
        "\t".join(["source name", "assay name", "technology type"])
        + "\n"
        + "\t".join(["sample_1", "assay_1", "mass spectrometry"])
        + "\n",
    )
    calls = []

    class CompletedProcess:
        returncode = 0
        stdout = ""
        stderr = ""

    def run(command, *, capture_output, check, text):
        calls.append((command, capture_output, check, text))
        return CompletedProcess()

    monkeypatch.setattr(sdrf_module.metadata, "version", lambda package: "0.0")
    monkeypatch.setattr(sdrf_module, "_load_parse_sdrf_entrypoint", lambda: lambda: "not a click command")
    monkeypatch.setattr(sdrf_module.shutil, "which", lambda executable: "/usr/bin/parse_sdrf")
    monkeypatch.setattr(sdrf_module.subprocess, "run", run)

    sdrf_module.validate_sdrf_file(path)

    assert calls == [
        (
            ["/usr/bin/parse_sdrf", "validate-sdrf", "--sdrf_file", str(path), "--skip-ontology"],
            True,
            False,
            True,
        )
    ]


def test_read_diann_merges_sdrf_file_by_data_file(tmp_path):
    sdrf_file = _write_sdrf(
        tmp_path,
        "\t".join(
            [
                "source name",
                "assay name",
                "technology type",
                "comment[data file]",
                "factor value[treatment]",
            ]
        )
        + "\n"
        + "\t".join(["S1", "run1", "mass spectrometry", "fileA.raw", "control"])
        + "\n"
        + "\t".join(["S2", "run2", "mass spectrometry", "fileB.raw", "case"])
        + "\n",
    )

    mdata = rr.read_diann(_write_diann_report(tmp_path), sdrf_file=sdrf_file, validate_sdrf=False)

    assert mdata.obs["source_name"].astype(str).tolist() == ["S1", "S2"]
    assert mdata.obs["factor_value_treatment"].astype(str).tolist() == ["control", "case"]
