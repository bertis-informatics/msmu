"""Consistent mass-spec filename-stem handling.

Readers build obs/var identifiers from raw run filenames, and SDRF matching
(``apply_sdrf_to_obs``, ``split_tmt``) matches those identifiers against
``comment[data file]``. Both sides must reduce a run name to the SAME bare stem,
so ``x.mzML``, ``x.mzML.gz`` and ``x.d`` all collapse to ``x``. Peeling only
*known* extensions keeps an in-name dot intact: ``sample.v2.mzML`` -> ``sample.v2``.

Two spellings share one extension list: :func:`strip_ms_extensions` (Python scalars,
pandas ``Series.map``) and :data:`MS_EXTENSION_REGEX` (polars/pandas vectorised
string ops). A test asserts the two agree.
"""

from pathlib import Path

# Known MS raw / peak-list / compression extensions (lower-case, no leading dot).
# Only trailing runs of these are stripped; the first unknown token stops the peel.
_MS_EXTENSIONS: frozenset[str] = frozenset(
    {
        "mzml", "mzxml", "mzmlb", "mzdata",  # open formats
        "raw",  # Thermo
        "d", "tdf", "tsf", "baf", "fid",  # Bruker / Agilent
        "wiff", "wiff2", "scan",  # SCIEX (.wiff, .wiff.scan)
        "mgf", "ms2",  # peak lists
        "gz", "zip", "bz2",  # compression wrappers
    }
)

# One-or-more trailing known extensions, anchored to the end, case-insensitive.
# Longest alternatives first so ``.wiff2`` is not shadowed by ``.wiff``.
MS_EXTENSION_REGEX: str = rf"(?i)(\.(?:{'|'.join(sorted(_MS_EXTENSIONS, key=len, reverse=True))}))+$"


def strip_ms_extensions(filename: str) -> str:
    """Return the bare run stem of ``filename``.

    Drops the directory path and any trailing *known* MS extensions
    (see ``_MS_EXTENSIONS``), repeatedly, so multi-part extensions collapse::

        "a/b/x.mzML"       -> "x"
        "x.mzML.gz"        -> "x"
        "x.wiff.scan"      -> "x"
        "x.d"              -> "x"
        "sample.v2.mzML"   -> "sample.v2"   # ".v2" is not a known extension
        "QExHF03751"       -> "QExHF03751"  # no extension
    """
    stem = Path(str(filename)).name
    while "." in stem:
        base, _dot, extension = stem.rpartition(".")
        if extension.lower() not in _MS_EXTENSIONS:
            break
        stem = base
    return stem
