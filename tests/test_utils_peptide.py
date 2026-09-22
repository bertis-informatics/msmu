"""Modified-peptide parsing: a residue's position must not depend on how modifications are written.

Every supported search engine writes modification tags differently, and several put letters inside
them -- DIA-NN's ``(UniMod:35)``, MaxQuant's ``(Phospho (STY))``. Counting those letters as residues
shifts every site after them, so the parser counts uppercase letters outside tags only.
"""

import pytest

from msmu._utils.peptide import (
    ModifiedResidue,
    parse_modified_peptide,
    residue_carries_modification,
)


def _tagged_residues(modified_peptide: str) -> list[tuple[str, int, tuple[str, ...]]]:
    return [
        (modified_residue.residue, modified_residue.position_in_peptide, modified_residue.tags)
        for modified_residue in parse_modified_peptide(modified_peptide)
        if modified_residue.tags
    ]


@pytest.mark.parametrize(
    ("modified_peptide", "phospho_tag"),
    [
        ("AC[+57.0215]M[+15.9949]PSGS[+79.9663]YTK", "[+79.9663]"),  # Sage
        ("AC(UniMod:4)M(UniMod:35)PSGS(UniMod:21)YTK", "(UniMod:21)"),  # DIA-NN
        ("_ACM(Oxidation (M))PSGS(Phospho (STY))YTK_", "(Phospho (STY))"),  # MaxQuant
        ("AC[160]M[147]PSGS[167]YTK", "[167]"),  # FragPipe
    ],
)
def test_the_phosphosite_is_residue_seven_in_every_notation(modified_peptide, phospho_tag):
    modified_residues = parse_modified_peptide(modified_peptide)

    assert "".join(modified_residue.residue for modified_residue in modified_residues) == "ACMPSGSYTK"
    assert modified_residues[6] == ModifiedResidue(residue="S", position_in_peptide=7, tags=(phospho_tag,))


@pytest.mark.parametrize(
    ("modified_peptide", "n_terminal_tag"),
    [
        ("[+42.0106]-AAKR", "[+42.0106]"),  # Sage
        ("(UniMod:1)AAKR", "(UniMod:1)"),  # DIA-NN
        ("_(Acetyl (Protein N-term))AAKR_", "(Acetyl (Protein N-term))"),  # MaxQuant
        ("n[43]AAKR", "[43]"),  # FragPipe
    ],
)
def test_an_n_terminal_tag_is_attached_to_residue_one(modified_peptide, n_terminal_tag):
    """A tag written before the first residue has no residue to follow; residue 1 is the one it modifies."""
    assert _tagged_residues(modified_peptide) == [("A", 1, (n_terminal_tag,))]


@pytest.mark.parametrize("modified_peptide", ["AAKR-[+0.984]", "AAKRc[17]"])
def test_a_c_terminal_tag_is_attached_to_the_last_residue(modified_peptide):
    assert [(residue, position) for residue, position, _ in _tagged_residues(modified_peptide)] == [("R", 4)]


def test_an_n_terminal_tag_and_a_residue_tag_on_residue_one_are_both_kept():
    assert _tagged_residues("(UniMod:1)S(UniMod:21)AKR") == [("S", 1, ("(UniMod:1)", "(UniMod:21)"))]


@pytest.mark.parametrize(
    "modified_peptide",
    [
        "PEPS[+79.97TIDE",  # unclosed
        "PEPS[+79.97)TIDE",  # mismatched closer
        "PEPS+79.97TIDE",  # mass outside any tag
        "PEPsTIDE",  # lowercase residue notation
        "[+42.0106]-",  # tag without residues
    ],
)
def test_an_unreadable_notation_raises_instead_of_guessing(modified_peptide):
    with pytest.raises(ValueError):
        parse_modified_peptide(modified_peptide)


def test_a_bare_tag_matches_any_residue_carrying_it():
    serine, threonine = parse_modified_peptide("S(UniMod:21)T(UniMod:21)K")[:2]

    assert residue_carries_modification(serine, "(UniMod:21)")
    assert residue_carries_modification(threonine, "(UniMod:21)")


def test_a_residue_qualified_tag_matches_that_residue_only():
    """FragPipe writes the modified residue's total mass, so the tag alone may not name one PTM."""
    serine, threonine = parse_modified_peptide("S[167]T[167]K")[:2]

    assert residue_carries_modification(serine, "S[167]")
    assert not residue_carries_modification(threonine, "S[167]")


def test_tags_are_matched_exactly_not_by_substring():
    """'[+79.97]' must not match a tag that merely contains it, nor a differently cased tag."""
    (serine,) = parse_modified_peptide("S[+79.9663]")
    (dia_serine,) = parse_modified_peptide("S(UniMod:21)")

    assert not residue_carries_modification(serine, "[+79.97]")
    assert not residue_carries_modification(dia_serine, "(unimod:21)")
