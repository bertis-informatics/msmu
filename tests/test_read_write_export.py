from pathlib import Path

from msmu._read_write._export import to_readable, write_csv, write_flashlfq_input


def test_to_readable_include_exclude_and_quant(psm_mdata_export):
    df = to_readable(psm_mdata_export, modality="psm", include=["filename", "rt"], quantification=False)
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


def test_write_csv_creates_file(tmp_path, psm_mdata_export):
    output = Path(tmp_path) / "psm.csv"
    write_csv(psm_mdata_export, modality="psm", filename=output, sep=",", include=["filename"], quantification=False)
    content = output.read_text().splitlines()
    assert content[0] == "filename"


def test_mdata_write_h5mu_works_with_cmd_dict_list(tmp_path, psm_mdata_export):
    mdata = psm_mdata_export.copy()
    mdata.uns["_cmd"] = {"0": {"function": "demo", "payload": {"source": "test"}}}
    output = Path(tmp_path) / "test.h5mu"
    mdata.write_h5mu(output)
    assert output.exists()
