import pandas as pd
import polars as pl

from msmu._read_write._diann import DiannReader
from msmu._read_write._fragpipe import LfqFragPipeReader, TmtFragPipeReader
from msmu._read_write._maxquant import MaxLfqReader, MaxQuantReader, MaxTmtReader
from msmu._read_write._sage import LfqSageReader, SageReader, TmtSageReader


def test_diann_make_needed_columns_sets_decoy_and_lengths():
    reader = DiannReader("dummy.tsv", pd.DataFrame())
    df = pd.DataFrame(
        {
            "Protein.Ids": ["sp|P1|P1_HUMAN"],
            "Stripped.Sequence": ["ACDK"],
            "Modified.Sequence": ["ACDK"],
            "Run": ["sample.raw"],
            "Precursor.Charge": [2],
            "Precursor.Id": ["ACDK2"],
            "RT": [10.5],
            "PEP": [0.01],
            "Lib.Q.Value": [0.0],
            "Global.Q.Value": [0.01],
        }
    )
    out = reader._make_needed_columns_for_identification(pl.from_pandas(df))
    assert out["missed_cleavages"].iloc[0] == 1
    assert out["peptide_length"].iloc[0] == 4
    assert out["decoy"].iloc[0] == 0


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
            "charge": [2],
            "peptide_len": [5],
            "expmass": [565.2],
            "calcmass": [565.1],
            "rt": [10.5],
            "missed_cleavages": [0],
            "semi_enzymatic": [False],
            "hyperscore": [12.0],
            "spectrum_q": [0.01],
        }
    )
    out = reader._make_needed_columns_for_identification(pl.from_pandas(df))
    assert out["scan_num"].iloc[0] == 42
    assert out["stripped_peptide"].iloc[0] == "ACDEK"
    assert out["decoy"].iloc[0] == 1
    assert out["contaminant"].iloc[0] == 0
    assert out["PEP"].iloc[0] == 10**-2


def test_tmt_sage_rename_dict_for_obs():
    reader = TmtSageReader("id.tsv", "quant.tsv")
    quant_df = pd.DataFrame(columns=["tmt_1", "tmt_2"])
    rename = reader._make_rename_dict_for_obs(quant_df)
    assert rename["tmt_1"] == "TMT126"
    assert rename["tmt_2"] == "TMT127"


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
    out = reader._make_needed_columns_for_quantification(pl.from_pandas(df))
    assert out.index.tolist() == ["AA", "BB"]
    assert "file1.raw" in out.columns


def test_lfq_sage_read_keeps_rt_calcmass_var_columns_unique():
    reader = LfqSageReader(
        identification_file="id.tsv",
        identification_df=pl.from_pandas(pd.DataFrame(
            {
                "proteins": ["sp|P1|P1_HUMAN"],
                "filename": ["sample.raw"],
                "scannr": ["scan=42"],
                "peptide": ["ACDEK"],
                "label": [1],
                "posterior_error": [-2.0],
                "expmass": [565.2],
                "calcmass": [565.1],
                "charge": [2],
                "peptide_len": [5],
                "missed_cleavages": [0],
                "semi_enzymatic": [False],
                "hyperscore": [12.0],
                "spectrum_q": [0.01],
                "rt": [10.5],
            }
        )),
        quantification_file="quant.tsv",
        quantification_df=pl.from_pandas(pd.DataFrame(
            {
                "peptide": ["ACDEK"],
                "charge": [2],
                "proteins": ["sp|P1|P1_HUMAN"],
                "q_value": [0.01],
                "score": [12.0],
                "spectral_angle": [0.9],
                "sample.raw": [1000.0],
            }
        )),
    )

    mdata = reader.read()
    var_columns = mdata["psm"].var.columns.tolist()

    assert var_columns.count("rt") == 1
    assert var_columns.count("calcmass") == 1
    assert len(var_columns) == len(set(var_columns))


def test_fragpipe_tmt_reader_init_requires_identification_df():
    try:
        TmtFragPipeReader("id.tsv")
    except TypeError as exc:
        assert "identification_df" in str(exc)
    else:
        raise AssertionError("Expected TmtFragPipeReader to require identification_df")


def test_fragpipe_lfq_quantification_columns():
    reader = LfqFragPipeReader("id.tsv", "quant.tsv", drop_search_result=False)
    df = pd.DataFrame(
        {
            "Modified Sequence": ["AA", "BB"],
            "Sample1 Intensity": [1.0, 2.0],
            "Sample2 Intensity": [0.0, 3.0],
        }
    )
    out = reader._make_needed_columns_for_quantification(pl.from_pandas(df))
    rename = reader._make_rename_dict_for_obs(out)
    assert out.columns.tolist() == ["Sample1 Intensity", "Sample2 Intensity"]
    assert rename["Sample1 Intensity"] == "Sample1"


def test_fragpipe_tmt_extract_quant_from_raw_keeps_only_channels():
    reader = TmtFragPipeReader("id.tsv", pd.DataFrame(), drop_search_result=False)
    raw = pd.DataFrame(
        {
            "Spectrum": ["fileA.00010.00010.2", "fileA.00011.00011.2"],
            "Calculated Peptide Mass": [500.1, 510.2],
            "observed mass": [500.0, 510.0],
            "Hyperscore": [20.0, 25.0],
            "126": [100.0, 110.0],
            "127N": [200.0, 210.0],
        }
    )
    quant = reader._extract_quant_from_raw(pl.from_pandas(raw))
    # only the real TMT channels survive; renamed id cols (calcmass/expmass/score) do NOT leak
    assert quant.columns.tolist() == ["126", "127N"]
    assert quant.index.tolist() == ["fileA.10", "fileA.11"]
    assert quant.loc["fileA.10", "126"] == 100.0


def test_fragpipe_tmt_rename_dict_for_obs():
    reader = TmtFragPipeReader("id.tsv", pd.DataFrame(), drop_search_result=False)
    quant_df = pd.DataFrame(columns=["126", "127N"])
    rename = reader._make_rename_dict_for_obs(quant_df)
    assert rename["126"] == "TMT126"
    assert rename["127N"] == "TMT127N"


def test_maxquant_make_needed_columns_for_identification():
    reader = MaxQuantReader("id.tsv", pd.DataFrame())
    df = pd.DataFrame(
        {
            "Reverse": ["+", ""],
            "Potential contaminant": ["", "+"],
            "Proteins": ["P1", "CON__P2"],
            "Leading proteins": ["REV__P3", "P4"],
            "Sequence": ["AA", "BB"],
            "Modified sequence": ["_AA_", "_BB_"],
            "Length": [2, 2],
            "Missed cleavages": [0, 1],
            "Charge": [2, 3],
            "Raw file": ["f1", "f2"],
            "MS/MS Scan Number": [10, 20],
            "Retention time": [1.0, 2.0],
            "PEP": [0.01, 0.02],
        }
    )
    out = reader._make_needed_columns_for_identification(pl.from_pandas(df))
    # fresh build: the raw frame is read read-only, not mutated
    assert "decoy" not in df.columns
    assert out["decoy"].tolist() == [1, 0]
    assert out["contaminant"].tolist() == [0, 1]
    # decoy row takes Leading proteins; target row keeps Proteins. Accessions come out canonical
    # ("CON__" -> "Cont_"); a pipe-less decoy keeps its tag verbatim, since decoy tags vary per
    # engine and are never enumerated -- decoy status itself comes from the Reverse column.
    assert out["proteins"].tolist() == ["REV__P3", "Cont_P2"]
    # pre-rename columns are carried for _normalise_identification_df to rename
    assert out["Modified sequence"].tolist() == ["_AA_", "_BB_"]


def test_maxquant_tmt_rename_dict_for_obs():
    reader = MaxTmtReader("id.tsv", pd.DataFrame())
    quant_df = pd.DataFrame(columns=["Reporter intensity corrected 1", "Reporter intensity corrected 2"])
    rename = reader._make_rename_dict_for_obs(quant_df)
    assert rename["Reporter intensity corrected 1"] == "TMT126"
    assert rename["Reporter intensity corrected 2"] == "TMT127"


def test_maxquant_lfq_extract_quant_from_raw():
    reader = MaxLfqReader("id.tsv", pd.DataFrame())
    raw = pd.DataFrame(
        {
            "Raw file": ["f1", "f2", "f1"],
            "Modified sequence": ["AA", "AA", "BB"],
            "Intensity": [1.0, 2.0, 5.0],
        }
    )
    quant = reader._extract_quant_from_raw(pl.from_pandas(raw))
    # peptide-level sum, pivoted to peptide x filename
    assert quant.loc["AA", "f1"] == 1.0
    assert quant.loc["AA", "f2"] == 2.0
    assert quant.loc["BB", "f1"] == 5.0


def test_maxquant_tmt_extract_quant_from_raw():
    reader = MaxTmtReader("id.tsv", pd.DataFrame())
    raw = pd.DataFrame(
        {
            "Raw file": ["f1", "f1"],
            "MS/MS Scan Number": [10, 11],
            "Reporter intensity corrected 1": [100.0, 150.0],
            "Reporter intensity corrected 2": [200.0, 250.0],
            "Other": ["x", "y"],
        }
    )
    quant = reader._extract_quant_from_raw(pl.from_pandas(raw))
    # indexed filename.scan_num; only the reporter channels are kept
    assert quant.index.tolist() == ["f1.10", "f1.11"]
    assert quant.columns.tolist() == ["Reporter intensity corrected 1", "Reporter intensity corrected 2"]
    assert quant.loc["f1.10", "Reporter intensity corrected 1"] == 100.0
