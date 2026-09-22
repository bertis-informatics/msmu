"""Peptide-related helper functions shared across readers.

These utilities stay free of reader-specific dependencies so they can be reused
wherever peptide string processing or basic precursor calculations are needed.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

# A modification tag opens with one of these and closes with its partner. Tags nest: MaxQuant writes
# "(Phospho (STY))".
MODIFICATION_TAG_CLOSER_BY_OPENER: dict[str, str] = {"[": "]", "(": ")", "{": "}"}
MODIFICATION_TAG_CLOSERS: frozenset[str] = frozenset(MODIFICATION_TAG_CLOSER_BY_OPENER.values())
# Outside a tag these carry no residue: MaxQuant wraps sequences in "_", flanking residues are cut
# off with ".", and Sage separates a terminal tag from the sequence with "-".
SEQUENCE_DELIMITERS: frozenset[str] = frozenset("_.-")
# FragPipe prefixes terminal modifications with a lowercase marker: "n[43]PEPTIDE", "PEPTIDEc[17]".
N_TERMINUS_MARKER: str = "n"
C_TERMINUS_MARKER: str = "c"


@dataclass(frozen=True)
class ModifiedResidue:
    """One residue of a modified peptide and the modification tags written against it."""

    residue: str
    position_in_peptide: int  # 1-based
    tags: tuple[str, ...]


def parse_modified_peptide(modified_peptide: str) -> list[ModifiedResidue]:
    """Split a modified peptide string into residues, each with the modification tags attached to it.

    Only uppercase letters outside a tag are residues; everything inside brackets is tag text,
    however many letters it holds. That is what keeps a residue's position independent of the
    notation: ``AC[+57.0215]S[+79.9663]K`` (Sage) and ``AC(UniMod:4)S(UniMod:21)K`` (DIA-NN) both put
    the phosphate on residue 3. A tag written before the first residue -- an N-terminal modification,
    as in ``[+42.0106]-AAK``, ``(UniMod:1)AAK``, ``_(Acetyl (Protein N-term))AAK_`` or ``n[43]AAK``
    -- is attached to residue 1; every other tag is attached to the residue it follows.

    Raises:
        ValueError: On an unbalanced tag, or a character that no supported notation uses outside a
            tag. An unknown notation then fails loudly instead of yielding shifted positions.
    """
    residues: list[str] = []
    tags_per_residue: list[list[str]] = []
    n_terminal_tags: list[str] = []

    character_index = 0
    while character_index < len(modified_peptide):
        character = modified_peptide[character_index]

        if character in MODIFICATION_TAG_CLOSER_BY_OPENER:
            tag_end_index = _find_modification_tag_end(modified_peptide, character_index)
            tag = modified_peptide[character_index : tag_end_index + 1]
            if residues:
                tags_per_residue[-1].append(tag)
            else:
                n_terminal_tags.append(tag)
            character_index = tag_end_index + 1
            continue

        if character in string.ascii_uppercase:
            residues.append(character)
            tags_per_residue.append(list(n_terminal_tags) if len(residues) == 1 else [])
        elif character in SEQUENCE_DELIMITERS:
            pass
        elif character == N_TERMINUS_MARKER and not residues:
            pass
        elif character == C_TERMINUS_MARKER and residues:
            pass
        else:
            raise ValueError(
                f"Unrecognised character {character!r} at index {character_index} of modified peptide "
                f"{modified_peptide!r}."
            )
        character_index += 1

    if n_terminal_tags and not residues:
        raise ValueError(f"Modified peptide {modified_peptide!r} has modification tags but no residues.")

    return [
        ModifiedResidue(residue=residue, position_in_peptide=residue_index + 1, tags=tuple(tags))
        for residue_index, (residue, tags) in enumerate(zip(residues, tags_per_residue))
    ]


def _find_modification_tag_end(modified_peptide: str, tag_start_index: int) -> int:
    """Index of the bracket closing the tag opened at ``tag_start_index``, honouring nested tags."""
    expected_closers: list[str] = []
    for character_index in range(tag_start_index, len(modified_peptide)):
        character = modified_peptide[character_index]
        if character in MODIFICATION_TAG_CLOSER_BY_OPENER:
            expected_closers.append(MODIFICATION_TAG_CLOSER_BY_OPENER[character])
        elif character in MODIFICATION_TAG_CLOSERS:
            if character != expected_closers[-1]:
                raise ValueError(
                    f"Mismatched {character!r} at index {character_index} of modified peptide "
                    f"{modified_peptide!r}; expected {expected_closers[-1]!r}."
                )
            expected_closers.pop()
            if not expected_closers:
                return character_index

    raise ValueError(f"Unclosed modification tag in modified peptide {modified_peptide!r}.")


def is_residue_qualified_modification(modification: str) -> bool:
    """Whether ``modification`` names its residue, as in ``S[167]``, rather than being a bare tag."""
    return len(modification) > 1 and modification[0] in string.ascii_uppercase


def residue_carries_modification(modified_residue: ModifiedResidue, modification: str) -> bool:
    """Whether a residue carries ``modification``.

    ``modification`` is either a tag exactly as written in the peptide string (``[+79.9663]``,
    ``(UniMod:21)``) or a tag qualified by its residue (``S[167]``), which matches that residue only.
    """
    if is_residue_qualified_modification(modification):
        return modified_residue.residue == modification[0] and modification[1:] in modified_residue.tags
    return modification in modified_residue.tags


def _make_stripped_peptide(peptide: str) -> str:
    """Return the unmodified amino acid sequence from a modified peptide string."""
    pattern = r"([A-Z]+)|(\[\+\d+\.\d+\])"
    split_peptide = re.findall(pattern, peptide)
    return "".join(item[0] for item in split_peptide if item[0])


def _count_missed_cleavages(peptide: str, enzyme: str = "trypsin") -> int:
    """Count missed cleavages for a tryptic digest."""
    if enzyme != "trypsin":
        raise ValueError("This helper currently only supports trypsin.")
    cleavage_sites = [match.start() + 1 for match in re.finditer(r"(?<=[KR])(?!P)", peptide)]
    return len(cleavage_sites)


def _calc_exp_mz(expmass: float, charge: int) -> float:
    """Calculate the experimental m/z from neutral mass and charge."""
    return (expmass + charge * 1.007276466812) / charge


def _get_peptide_length(peptide: str) -> int:
    """Return the length of a stripped peptide sequence."""
    return len(peptide)
