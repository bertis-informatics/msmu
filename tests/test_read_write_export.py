from pathlib import Path

import pandas as pd
import pytest

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


def test_write_pin_returns_expected_schema(psm_mdata_export):
    mdata = _make_pin_ready_mdata(psm_mdata_export)

    pin_df = write_pin(mdata)

    assert pin_df is not None
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


def test_write_pin_includes_decoy_rows(psm_mdata_export):
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

    pin_df = write_pin(mdata)

    assert pin_df is not None
    assert pin_df.index.tolist() == ["f1", "f2", "d1"]
    assert pin_df["Label"].tolist() == [1, 1, -1]
    assert pin_df.loc["d1", "SpecId"] == "d1"
    assert pin_df.loc["d1", "XCorr"] == 5.0


def test_write_pin_missing_required_source_column_raises(psm_mdata_export):
    mdata = _make_pin_ready_mdata(psm_mdata_export)
    mdata.mod["psm"].var = mdata.mod["psm"].var.drop(columns=["score"])

    with pytest.raises(ValueError, match=r"Required columns missing from psm.var: \['score'\]"):
        write_pin(mdata)


def test_write_pin_missing_peptide_length_raises(psm_mdata_export):
    mdata = _make_pin_ready_mdata(psm_mdata_export)
    mdata.mod["psm"].var = mdata.mod["psm"].var.drop(columns=["peptide_length"])

    with pytest.raises(ValueError, match=r"Required columns missing from psm.var: \['peptide_length'\]"):
        write_pin(mdata)


def test_write_pin_missing_required_decoy_column_raises(psm_mdata_export):
    mdata = _make_pin_ready_mdata(psm_mdata_export)
    decoy_df = mdata.mod["psm"].var.copy()
    decoy_df.index = ["d1", "d2"]
    decoy_df["decoy"] = 1
    mdata.mod["psm"].uns["decoy"] = decoy_df.drop(columns=["score"])

    with pytest.raises(ValueError, match=r"Required columns missing from psm\.uns\['decoy'\]: \['score'\]"):
        write_pin(mdata)


def test_mdata_write_h5mu_works_with_cmd_dict_list(tmp_path, psm_mdata_export):
    mdata = psm_mdata_export.copy()
    mdata.uns["_cmd"] = {"0": {"function": "demo", "payload": {"source": "test"}}}
    output = Path(tmp_path) / "test.h5mu"
    mdata.write_h5mu(output)
    assert output.exists()
