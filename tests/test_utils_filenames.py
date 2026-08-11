import polars as pl
import pytest

from msmu._utils._filenames import MS_EXTENSION_REGEX, strip_ms_extensions

# (raw run name -> bare stem). Covers: no-ext, single, double/compression, SCIEX two-part,
# Bruker directory ext, case-insensitivity, and in-name dots that must be preserved.
_CASES = [
    ("QExHF03751", "QExHF03751"),  # no extension -> unchanged
    ("x.mzML", "x"),  # single ext
    ("a/b/c/x.mzML", "x"),  # directory dropped
    ("x.mzML.gz", "x"),  # double ext (compression wrapper)
    ("x.raw.gz", "x"),
    ("x.wiff.scan", "x"),  # SCIEX two-part
    ("x.d", "x"),  # Bruker directory ext
    ("x.wiff", "x"),
    ("sample.v2.mzML", "sample.v2"),  # in-name dot preserved (.v2 is not a known ext)
    ("run.2024.raw", "run.2024"),  # numeric in-name token preserved
    ("X.MZML", "X"),  # case-insensitive
    ("x.MzML.GZ", "x"),
]


@pytest.mark.parametrize("filename, expected", _CASES)
def test_strip_ms_extensions(filename, expected):
    assert strip_ms_extensions(filename) == expected


def test_python_and_polars_regex_agree():
    # The vectorised polars path (MS_EXTENSION_REGEX on a basename) must equal the scalar helper,
    # since readers use the regex while downstream/SDRF matching use the function.
    inputs = [case[0] for case in _CASES]
    polars_out = (
        pl.DataFrame({"filename": inputs})
        .select(pl.col("filename").str.split("/").list.last().str.replace(MS_EXTENSION_REGEX, ""))
        .to_series()
        .to_list()
    )
    python_out = [strip_ms_extensions(name) for name in inputs]
    assert polars_out == python_out
