"""Peptide-related helper functions shared across readers.

These utilities stay free of reader-specific dependencies so they can be reused
wherever peptide string processing or basic precursor calculations are needed.
"""

from __future__ import annotations

import re


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
