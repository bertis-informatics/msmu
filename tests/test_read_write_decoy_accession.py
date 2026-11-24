"""Readers name a decoy from their decoy column, so no decoy shares an accession with a target.

DIA-NN does not tag decoy entries: a decoy carries the accession of the target it was made from.
Before the decoy prefix followed the decoy column, those decoys mapped onto target proteins in
``infer_protein`` and changed the target groups.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

import msmu as mm
from msmu._read_write._base_reader import SearchResultDataFrameConverter
from msmu._read_write._fragpipe import TmtFragPipeReader

DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"
DIANN_REPORT = DATA_DIRECTORY / "diann_dia" / "diann" / "report.parquet"
SAGE_LFQ_DIRECTORY = DATA_DIRECTORY / "sage_lfq" / "sage"
Q_VALUE_THRESHOLD = 0.01
DECOY_PREFIX = "decoy_"


def _accessions(protein_groups: pd.Series) -> set[str]:
    """Every accession in the groups; members are separated by ";" (reader) or "," (inference)."""
    return {accession for group in protein_groups for accession in re.split(r"[;,]", group)}


def _diann_peptides_with_inferred_proteins(mdata):
    mdata = mm.pp.filter(mdata, modality="psm", on="var", column="q_value", keep="lt", value=Q_VALUE_THRESHOLD)
    mdata = mm.pp.to_peptide(mdata)
    return mm.pp.infer_protein(mdata)


def test_diann_decoys_take_the_decoy_prefix_although_the_engine_does_not_tag_them():
    mdata = mm.read_diann(identification_file=str(DIANN_REPORT))

    decoy_accessions = _accessions(mdata["psm"].uns["decoy"]["proteins"])
    target_accessions = _accessions(mdata["psm"].var["proteins"])

    assert decoy_accessions
    assert all(accession.startswith(DECOY_PREFIX) for accession in decoy_accessions)
    assert not any(accession.startswith(DECOY_PREFIX) for accession in target_accessions)


def test_diann_decoys_do_not_change_the_target_protein_groups():
    mdata = _diann_peptides_with_inferred_proteins(mm.read_diann(identification_file=str(DIANN_REPORT)))
    target_only = mm.read_diann(identification_file=str(DIANN_REPORT))
    del target_only["psm"].uns["decoy"]
    target_only = _diann_peptides_with_inferred_proteins(target_only)

    decoy_group_accessions = _accessions(mdata["peptide"].uns["decoy"]["protein_group"])

    assert all(accession.startswith(DECOY_PREFIX) for accession in decoy_group_accessions)
    pd.testing.assert_series_equal(mdata["peptide"].var["protein_group"], target_only["peptide"].var["protein_group"])


def test_sage_decoys_take_the_decoy_prefix_in_place_of_the_engine_tag():
    mdata = mm.read_sage(
        identification_file=str(SAGE_LFQ_DIRECTORY / "results.sage.tsv"),
        quantification_file=str(SAGE_LFQ_DIRECTORY / "lfq.tsv"),
        label="label_free",
    )

    decoy_accessions = _accessions(mdata["psm"].uns["decoy"]["proteins"])
    target_accessions = _accessions(mdata["psm"].var["proteins"])

    assert decoy_accessions
    assert all(accession.startswith(DECOY_PREFIX) for accession in decoy_accessions)
    assert not any(accession.startswith(DECOY_PREFIX) for accession in target_accessions)
    assert not any(accession.startswith("rev_") for accession in decoy_accessions | target_accessions)


def test_fragpipe_still_recognises_decoys_by_the_tag_in_its_protein_column(tmp_path):
    """psm.tsv has no decoy column, so the reader reads FragPipe's tag before the parser removes it."""
    rows = []
    for row_number, protein in enumerate(["sp|P11111|A_HUMAN", "rev_sp|P22222|B_HUMAN"]):
        rows.append(
            {
                "Spectrum": f"runA.{100 + row_number}.{100 + row_number}.2",
                "Protein": protein,
                "Mapped Proteins": None,  # no other protein shares the peptide
                "Modified Peptide": f"PEPTIDEK{row_number}",
                "Peptide": f"PEPTIDEK{row_number}",
                "Retention": 600.0 + row_number,
                "Charge": 2,
                "Peptide Length": 9,
                "Number of Missed Cleavages": 0,
                "Calculated Peptide Mass": 1000.0 + row_number,
                "observed mass": 1000.5 + row_number,
                "Hyperscore": 20.0 + row_number,
                "126": 1.0,
                "127N": 2.0,
            }
        )
    path = tmp_path / "psm.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    identification_file, identification_df = SearchResultDataFrameConverter().convert([path])

    mdata = TmtFragPipeReader(identification_file=identification_file, identification_df=identification_df).read()

    assert mdata["psm"].var["proteins"].tolist() == ["P11111"]
    assert mdata["psm"].uns["decoy"]["proteins"].tolist() == ["decoy_P22222"]
