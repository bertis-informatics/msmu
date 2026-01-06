from pathlib import Path

import pandas as pd

from msmu._utils.fasta import (
    _get_protein_info_from_fasta,
    _map_fasta,
    _split_uniprot_fasta_entry,
    attach_fasta,
    map_fasta,
    parse_uniprot_accession,
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
    assert parsed[0] == "P1;rev_P2;contam_P3"


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
    assert set(mapped["protein"].var["Gene"].tolist()[2].split(";")) == {"GENE1", "GENE2"}
