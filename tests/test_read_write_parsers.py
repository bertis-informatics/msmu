import pandas as pd

from msmu._read_write._diann import DiannReader
from msmu._read_write._fragpipe import LfqFragPipeReader, TmtFragPipeReader
from msmu._read_write._maxquant import MaxLfqReader, MaxQuantReader, MaxTmtReader
from msmu._read_write._sage import LfqSageReader, SageReader, TmtSageReader


def test_diann_make_needed_columns_sets_decoy_and_lengths():
    reader = DiannReader("dummy.tsv")
    df = pd.DataFrame(
        {
            "Protein.Ids": ["sp|P1|P1_HUMAN"],
            "Stripped.Sequence": ["ACDK"],
            "Lib.Q.Value": [0.0],
        }
    )
    out = reader._make_needed_columns_for_identification(df)
    assert out["missed_cleavages"].iloc[0] == 1
    assert out["peptide_length"].iloc[0] == 4
    assert out["decoy"].iloc[0] == 0


def test_diann_split_merged_identification_quantification():
    reader = DiannReader("dummy.tsv")
    df = pd.DataFrame(
        {
            "filename": ["a", "b"],
            "Precursor.Quantity": [1.0, 2.0],
        },
        index=["a.1", "b.2"],
    )
    ident, quant = reader._split_merged_identification_quantification(df)
    assert "Precursor.Quantity" not in ident.columns
    assert quant.loc["a.1", "a"] == 1.0
    assert quant.loc["b.2", "b"] == 2.0


def test_sage_make_needed_columns_for_identification():
    reader = SageReader("id.tsv", None)
    df = pd.DataFrame(
        {
            "proteins": ["sp|P1|P1_HUMAN"],
            "filename": ["file1.raw"],
            "scannr": ["scan=42"],
            "peptide": ["ACDEK"],
            "label": [-1],
            "posterior_error": [-2.0],
        }
    )
    out = reader._make_needed_columns_for_identification(df)
    assert out["scan_num"].iloc[0] == 42
    assert out["stripped_peptide"].iloc[0] == "ACDEK"
    assert out["decoy"].iloc[0] == 1
    assert out["contaminant"].iloc[0] == 0
    assert out["PEP"].iloc[0] == 10 ** -2


def test_tmt_sage_rename_dict_for_obs():
    reader = TmtSageReader("id.tsv", "quant.tsv")
    quant_df = pd.DataFrame(columns=["tmt_1", "tmt_2"])
    rename = reader._make_rename_dict_for_obs(quant_df)
    assert rename["tmt_1"] == "126"
    assert rename["tmt_2"] == "127"


def test_lfq_sage_quantification_columns():
    reader = LfqSageReader("id.tsv", "quant.tsv")
    df = pd.DataFrame(
        {
            "peptide": ["AA", "BB"],
            "charge": [2, 3],
            "proteins": ["P1", "P2"],
            "q_value": [0.1, 0.2],
            "score": [1.0, 2.0],
            "spectral_angle": [0.1, 0.2],
            "file1.raw": [1.0, 2.0],
        }
    )
    out = reader._make_needed_columns_for_quantification(df)
    assert out.index.tolist() == ["AA", "BB"]
    assert "file1.raw" in out.columns


def test_fragpipe_tmt_reader_init_requires_quant_file():
    try:
        TmtFragPipeReader("id.tsv")
    except TypeError as exc:
        assert "quantification_file" in str(exc)
    else:
        raise AssertionError("Expected TmtFragPipeReader to require quantification_file")


def test_fragpipe_lfq_quantification_columns():
    reader = LfqFragPipeReader("id.tsv", "quant.tsv")
    df = pd.DataFrame(
        {
            "Modified Sequence": ["AA", "BB"],
            "Sample1 Intensity": [1.0, 2.0],
            "Sample2 Intensity": [0.0, 3.0],
        }
    )
    out = reader._make_needed_columns_for_quantification(df)
    rename = reader._make_rename_dict_for_obs(out)
    assert out.columns.tolist() == ["Sample1 Intensity", "Sample2 Intensity"]
    assert rename["Sample1 Intensity"] == "Sample1"


def test_maxquant_make_needed_columns_for_identification():
    reader = MaxQuantReader("id.tsv")
    df = pd.DataFrame(
        {
            "Reverse": ["+", ""],
            "Potential contaminant": ["", "+"],
            "Proteins": ["P1", "CON__P2"],
            "Leading proteins": ["REV__P3", "P4"],
        }
    )
    out = reader._make_needed_columns_for_identification(df)
    assert out["decoy"].tolist() == [1, 0]
    assert out["contaminant"].tolist() == [0, 1]
    assert out["proteins"].tolist()[0].startswith("rev_")
    assert out["proteins"].tolist()[1].startswith("contam_")


def test_maxquant_tmt_rename_dict_for_obs():
    reader = MaxTmtReader("id.tsv")
    quant_df = pd.DataFrame(columns=["Reporter intensity corrected 1", "Reporter intensity corrected 2"])
    rename = reader._make_rename_dict_for_obs(quant_df)
    assert rename["Reporter intensity corrected 1"] == "126"
    assert rename["Reporter intensity corrected 2"] == "127"


def test_maxquant_lfq_split_merged_identification_quantification():
    reader = MaxLfqReader("id.tsv")
    df = pd.DataFrame(
        {
            "filename": ["f1", "f2"],
            "peptide": ["AA", "AA"],
            "Intensity": [1.0, 2.0],
        }
    )
    ident, quant = reader._split_merged_identification_quantification(df)
    assert "Intensity" not in ident.columns
    assert quant.loc["AA", "f1"] == 1.0
