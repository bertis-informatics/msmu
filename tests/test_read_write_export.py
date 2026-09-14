from pathlib import Path

import pandas as pd
import pytest
import msmu as mm

from msmu._read_write._export import to_readable, write_csv, write_flashlfq_input, write_pin


def _make_pin_ready_mdata(mdata):
    mdata = mdata.copy()
    mdata.mod["psm"].var["scan_num"] = [101, 102]
    mdata.mod["psm"].var["expmass"] = [101.0, 202.0]
    mdata.mod["psm"].var["score"] = [50.0, 60.0]
    mdata.mod["psm"].var["peptide_length"] = [2, 2]
    return mdata


def test_to_readable_include_exclude_and_quant(psm_mdata_export):
    df = to_readable(
        psm_mdata_export,
        modality="psm",
        include=["filename", "rt"],
        quantification=False,
    )
    assert df.columns.tolist() == ["filename", "rt"]

    df2 = to_readable(psm_mdata_export, modality="psm", exclude="extra", quantification=True)
    assert "extra" not in df2.columns
    assert "s1" in df2.columns


def test_write_flashlfq_input(tmp_path, psm_mdata_export):
    output = Path(tmp_path) / "flashlfq.tsv"
    write_flashlfq_input(psm_mdata_export, output)
    content = output.read_text().splitlines()
    assert "File Name" in content[0]
    assert "Protein Accession" in content[0]


def test_write_flashlfq_input_missing_required_column_raises(tmp_path, psm_mdata_export):
    mdata = psm_mdata_export.copy()
    mdata.mod["psm"].var = mdata.mod["psm"].var.drop(columns=["rt"])

    with pytest.raises(ValueError, match=r"Required columns missing from psm.var: \['rt'\]"):
        write_flashlfq_input(mdata, Path(tmp_path) / "flashlfq.tsv")


def test_write_csv_creates_file(tmp_path, psm_mdata_export):
    output = Path(tmp_path) / "psm.csv"
    write_csv(
        psm_mdata_export,
        modality="psm",
        filename=output,
        sep=",",
        include=["filename"],
        quantification=False,
    )
    content = output.read_text().splitlines()
    assert content[0] == "filename"


def test_exports_skip_logging_and_hashing(tmp_path, psm_mdata_export, monkeypatch):
    from msmu._core import _provenance as core

    mdata = _make_pin_ready_mdata(psm_mdata_export)
    mm.provenance.log(lambda mdata: mdata)(mdata)
    before = mm.provenance.get_log(mdata)

    def forbidden(_):
        raise AssertionError("Export must not calculate hashes")

    monkeypatch.setattr(core, "compute_hash", forbidden)
    with mm.provenance.options(hashing=True):
        assert isinstance(to_readable(mdata, modality="psm"), pd.DataFrame)
        write_csv(mdata, modality="psm", filename=tmp_path / "out.csv", sep=",")
        write_flashlfq_input(mdata, tmp_path / "flashlfq.tsv")
        assert write_pin(mdata, tmp_path / "out.pin") is None
    assert all((tmp_path / name).is_file() for name in ("out.csv", "flashlfq.tsv", "out.pin"))
    assert mm.provenance.get_log(mdata) == before


def test_write_pin_requires_filename(psm_mdata_export):
    with pytest.raises(TypeError, match="filename"):
        write_pin(psm_mdata_export)
    with pytest.raises(TypeError, match="filename"):
        write_pin(psm_mdata_export, filename=None)


def test_write_pin_writes_expected_schema(tmp_path, psm_mdata_export):
    mdata = _make_pin_ready_mdata(psm_mdata_export)

    output = tmp_path / "out.pin"
    assert write_pin(mdata, output) is None
    pin_df = pd.read_csv(output, sep="\t")
    assert pin_df.columns.tolist() == [
        "SpecId",
        "Label",
        "Peptide",
        "Proteins",
        "Charge",
        "ScanNr",
        "PepLen",
        "CalcMass",
        "ExpMass",
        "XCorr",
    ]
    assert pin_df["XCorr"].tolist() == [50.0, 60.0]
    assert pin_df["PepLen"].tolist() == [2, 2]


def test_write_pin_includes_decoy_rows(tmp_path, psm_mdata_export):
    mdata = _make_pin_ready_mdata(psm_mdata_export)
    mdata.mod["psm"].uns["decoy"] = pd.DataFrame(
        {
            "filename": ["decoy.raw"],
            "scan_num": [201],
            "charge": [2],
            "peptide": ["DD"],
            "proteins": ["DECOY_P1"],
            "calcmass": [300.0],
            "expmass": [301.0],
            "score": [5.0],
            "peptide_length": [2],
            "decoy": [1],
        },
        index=["d1"],
    )

    output = tmp_path / "out.pin"
    assert write_pin(mdata, str(output)) is None
    pin_df = pd.read_csv(output, sep="\t").set_index("SpecId", drop=False)
    assert pin_df.index.tolist() == ["f1", "f2", "d1"]
    assert pin_df["Label"].tolist() == [1, 1, -1]
    assert pin_df.loc["d1", "SpecId"] == "d1"
    assert pin_df.loc["d1", "XCorr"] == 5.0


def test_write_pin_missing_required_source_column_raises(tmp_path, psm_mdata_export):
    mdata = _make_pin_ready_mdata(psm_mdata_export)
    mdata.mod["psm"].var = mdata.mod["psm"].var.drop(columns=["score"])

    with pytest.raises(ValueError, match=r"Required columns missing from psm.var: \['score'\]"):
        write_pin(mdata, tmp_path / "out.pin")


def test_write_pin_missing_peptide_length_raises(tmp_path, psm_mdata_export):
    mdata = _make_pin_ready_mdata(psm_mdata_export)
    mdata.mod["psm"].var = mdata.mod["psm"].var.drop(columns=["peptide_length"])

    with pytest.raises(ValueError, match=r"Required columns missing from psm.var: \['peptide_length'\]"):
        write_pin(mdata, tmp_path / "out.pin")


def test_write_pin_missing_required_decoy_column_raises(tmp_path, psm_mdata_export):
    mdata = _make_pin_ready_mdata(psm_mdata_export)
    decoy_df = mdata.mod["psm"].var.copy()
    decoy_df.index = ["d1", "d2"]
    decoy_df["decoy"] = 1
    mdata.mod["psm"].uns["decoy"] = decoy_df.drop(columns=["score"])

    with pytest.raises(ValueError, match=r"Required columns missing from psm\.uns\['decoy'\]: \['score'\]"):
        write_pin(mdata, tmp_path / "out.pin")


def test_mdata_write_h5mu_preserves_provenance(tmp_path, psm_mdata_export):
    mdata = psm_mdata_export.copy()
    from msmu.provenance import log, get_log
    import mudata
    mdata = log(lambda mdata: mdata)(mdata)
    before = get_log(mdata)
    output = Path(tmp_path) / "test.h5mu"
    mdata.write_h5mu(output)
    assert output.exists()
    assert get_log(mudata.read_h5mu(output)) == before
