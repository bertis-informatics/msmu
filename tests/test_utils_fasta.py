from pathlib import Path

import pandas as pd
import pytest

from msmu._utils.fasta import (
    _get_protein_info_from_fasta,
    _map_fasta,
    _split_uniprot_fasta_entry,
    attach_fasta,
    map_fasta,
    parse_uniprot_accession,
    parse_uniprot_accession_group,
)


def _write_fasta(tmp_path: Path) -> Path:
    fasta = """>sp|P1|P1_HUMAN Protein One OS=Homo sapiens OX=9606 GN=GENE1
MKWVTFISLL
>tr|P2|P2_MOUSE Protein Two OS=Mus musculus OX=10090 GN=GENE2
MADEUPSEQ
"""
    path = tmp_path / "test.fasta"
    path.write_text(fasta)
    return path


def test_get_protein_info_from_fasta(tmp_path):
    fasta_path = _write_fasta(tmp_path)
    df = _get_protein_info_from_fasta(str(fasta_path))
    assert "P1" in df.index
    assert df.loc["P1", "Gene"] == "GENE1"
    assert df.loc["P2", "Organism"] == "Mus musculus"


def test_parse_uniprot_accession_handles_rev_contam():
    series = pd.Series(["sp|P1|P1_HUMAN;rev_sp|P2|P2_MOUSE;contam_sp|P3|P3_HUMAN"])
    parsed = parse_uniprot_accession(series)
    assert parsed[0] == "P1;rev_P2;Cont_P3"


@pytest.mark.parametrize(
    ("protein_entry", "expected_accession", "expected_is_contaminant"),
    [
        ("sp|P04264|K2C1_HUMAN", "P04264", False),
        # Marker in front: the convention our search pipeline emits.
        ("contam_sp|P02769|ALBU_BOVIN", "Cont_P02769", True),
        # Marker inside the accession: the Hao Lab universal contaminant FASTA.
        ("sp|Cont_P00722|BGAL_ECOLI", "Cont_P00722", True),
        ("tr|Cont_P30879|PEPC_PIG", "Cont_P30879", True),
        ("CON__P02769", "Cont_P02769", True),
        # Decoy tags are not enumerated, so an unfamiliar one still parses.
        ("rev_sp|Q8WU76-2|SCFD2_HUMAN", "rev_Q8WU76-2", False),
        ("DECOY_sp|Q8WU76|SCFD2_HUMAN", "rev_Q8WU76", False),
        # Decoy of a contaminant keeps both markers, so the two can later be removed together
        # without biasing target-decoy competition.
        ("rev_contam_sp|P02769|ALBU_BOVIN", "rev_Cont_P02769", True),
        # The protein name field is never inspected: this is a real protein, not a contaminant.
        ("sp|P12345|CONA_CANLI", "P12345", False),
        ("P12345", "P12345", False),
    ],
)
def test_parse_uniprot_accession_group_normalises_markers(protein_entry, expected_accession, expected_is_contaminant):
    accession, is_contaminant = parse_uniprot_accession_group(protein_entry)
    assert accession == expected_accession
    assert is_contaminant is expected_is_contaminant


def test_parse_uniprot_accession_group_keeps_self_duplicate_members_distinct():
    """An accession listed as both contaminant and target must stay distinguishable.

    Whether such a protein is a real identification or a contaminant is decided later, at the
    protein level, so the group string has to carry the evidence.
    """
    accession, is_contaminant = parse_uniprot_accession_group("contam_sp|P07339|CATD_HUMAN;sp|P07339|CATD_HUMAN")
    assert accession == "Cont_P07339;P07339"
    assert is_contaminant is True


def test_get_protein_info_from_fasta_indexes_hao_contaminant_entries(tmp_path):
    fasta = """>sp|Cont_P00722|BGAL_ECOLI Beta-galactosidase OS=Escherichia coli OX=83333 GN=lacZ
MKWVTFISLL
"""
    path = tmp_path / "contaminant.fasta"
    path.write_text(fasta)

    df = _get_protein_info_from_fasta(str(path))

    assert "Cont_P00722" in df.index
    assert df.loc["Cont_P00722", "Accession"] == "P00722"
    assert df.loc["Cont_P00722", "Gene"] == "lacZ"


def test_split_uniprot_fasta_entry_fallback():
    entry = "P1"
    source, accession, name = _split_uniprot_fasta_entry(entry)
    assert source == ""
    assert accession == "P1"
    assert name == ""


def test_map_fasta_maps_groups():
    fasta_meta = pd.DataFrame({"Gene": {"P1": "G1", "P2": "G2"}})
    mapped = _map_fasta("P1,P2;P2", fasta_meta, "Gene")
    first_group, second_group = mapped.split(";")
    assert set(first_group.split(",")) == {"G1", "G2"}
    assert second_group == "G2"


def test_attach_and_map_fasta(tmp_path, mdata):
    fasta_path = _write_fasta(tmp_path)
    out = attach_fasta(mdata, str(fasta_path))
    assert "protein_info" in out.uns

    out["protein"].var.index = ["P1", "P2", "P1;P2"]
    mapped = map_fasta(out, modality="protein", categories=["Gene"])
    assert mapped["protein"].var["Gene"].tolist()[0] == "GENE1"
    assert mapped["protein"].var["Gene"].tolist()[1] == "GENE2"
    assert set(mapped["protein"].var["Gene"].tolist()[2].split(";")) == {
        "GENE1",
        "GENE2",
    }
