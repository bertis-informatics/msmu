"""Every reader must emit accessions in the same canonical form.

Readers used to disagree: the parser-based ones (Sage, DIA-NN, DELPI) emitted bare accessions
while MaxQuant and FragPipe passed the search engine's full ``sp|ACC|NAME`` string through. That
silently broke FASTA annotation, since ``protein_info`` is indexed by accession -- ``map_fasta``
looked up the full string, missed every key, and filled Gene/Description/Organism with blanks.
"""

from __future__ import annotations

import pandas as pd
import pytest

from msmu._utils.fasta import _map_fasta, parse_uniprot_accession_group


@pytest.fixture
def fasta_meta():
    """``protein_info`` as ``_get_protein_info_from_fasta`` builds it: indexed by accession."""
    return pd.DataFrame(
        {
            "Gene": {"P10000": "GENE1", "P11111": "GENE2", "Cont_P02769": "ALB"},
        }
    )


@pytest.mark.parametrize(
    ("search_engine_protein_string", "expected_gene"),
    [
        ("sp|P10000|X0_HUMAN", "GENE1"),  # MaxQuant / FragPipe style
        ("contam_sp|P02769|ALBU_BOVIN", "ALB"),
        ("sp|Cont_P02769|ALBU_BOVIN", "ALB"),  # Hao Lab contaminant FASTA
    ],
)
def test_parsed_accession_resolves_against_fasta_metadata(fasta_meta, search_engine_protein_string, expected_gene):
    accession, _ = parse_uniprot_accession_group(search_engine_protein_string)

    assert _map_fasta(accession, fasta_meta, "Gene") == expected_gene


def test_unparsed_protein_string_would_not_resolve(fasta_meta):
    """Pin the failure this guards against: the raw string finds nothing."""
    assert _map_fasta("sp|P10000|X0_HUMAN", fasta_meta, "Gene") == ""


def test_all_readers_agree_on_accession_form():
    """The same protein, however each engine spells it, parses to one accession."""
    spellings = [
        "sp|P11111|A_HUMAN",  # Sage / DIA-NN / DELPI / MaxQuant / FragPipe
        "P11111",  # bare accession (MaxQuant identifier rule)
    ]

    parsed = {parse_uniprot_accession_group(spelling)[0] for spelling in spellings}

    assert parsed == {"P11111"}
