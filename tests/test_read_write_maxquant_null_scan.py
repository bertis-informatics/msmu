"""MaxQuant TMT: a null ``MS/MS Scan Number`` must not drop features on the polars path.

The identification and quantification frames are joined on ``filename + "." + scan_num``. A single
null scan coerces the whole scan column to float on the way to pandas, so ``astype(str)`` yields
"123.0" for the identification side. If the quantification side formats the same column any other
way (e.g. a polars ``cast(Utf8)`` giving "123"), no index matches and every feature is silently
dropped at the ident/quant intersection -- a data-loss bug that integer-only scan numbers hide.
"""

import numpy as np
import pandas as pd
import pytest

from msmu._read_write._maxquant import MaxQuantDataFrameConverter, MaxTmtReader

from _parity_helpers import assert_polars_matches_golden

PLEX = 6
SCAN_COLUMN = "MS/MS Scan Number"


def _write_maxquant_tmt_file(path, scan_numbers) -> None:
    """Minimal MaxQuant TMT msms table with the given scan numbers (``""`` for a missing scan).

    Scan numbers are written as strings so the file on disk holds bare integers and blanks, the
    way MaxQuant writes them. Handing pandas a column of ints with ``None`` would silently make it
    float and write "101.0", which reads back as a float column and hides the bug under test.
    """
    n_rows = len(scan_numbers)
    frame = pd.DataFrame(
        {
            "Type": ["MULTI-MSMS"] * n_rows,
            "Reverse": [None] * n_rows,
            "Potential contaminant": [None] * n_rows,
            "Proteins": [f"sp|P{10000 + i}|X{i}_HUMAN" for i in range(n_rows)],
            "Leading proteins": [f"sp|P{10000 + i}|X{i}_HUMAN" for i in range(n_rows)],
            "Sequence": [f"PEPTIDEK{i}" for i in range(n_rows)],
            "Modified sequence": [f"_PEPTIDEK{i}_" for i in range(n_rows)],
            "Length": [9] * n_rows,
            "Missed cleavages": [0] * n_rows,
            "Charge": [2] * n_rows,
            "Raw file": ["raw_0"] * n_rows,
            SCAN_COLUMN: scan_numbers,
            "Retention time": [10.0 + i for i in range(n_rows)],
            "PEP": [0.001] * n_rows,
        }
    )
    for channel in range(1, PLEX + 1):
        frame[f"Reporter intensity corrected {channel}"] = np.arange(1, n_rows + 1, dtype=float) * channel

    frame.to_csv(path, sep="\t", index=False)


def _read_maxquant_tmt(path, as_polars: bool):
    converter = MaxQuantDataFrameConverter()
    identification_file, identification_df = converter.convert([path], as_polars=as_polars)
    return MaxTmtReader(identification_file=identification_file, identification_df=identification_df).read()


@pytest.mark.parametrize(
    "scan_numbers, case",
    [
        (["101", "102", "103"], "all-integer scans"),
        (["101", "", "103"], "one null scan"),
    ],
)
def test_maxquant_tmt_polars_keeps_every_feature(tmp_path, scan_numbers, case):
    """The polars path keeps as many features as rows, with or without null scan numbers."""
    path = tmp_path / "msms.txt"
    _write_maxquant_tmt_file(path, scan_numbers)

    mdata = _read_maxquant_tmt(path, as_polars=True)

    assert mdata["psm"].shape == (PLEX, len(scan_numbers)), f"features lost on the polars path ({case})"


def test_maxquant_tmt_polars_matches_pandas_with_null_scan(tmp_path):
    """polars and pandas produce the same features, var_names and intensities despite a null scan."""
    path = tmp_path / "msms.txt"
    _write_maxquant_tmt_file(path, ["101", "", "103"])

    assert_polars_matches_golden(lambda as_polars: _read_maxquant_tmt(path, as_polars), "maxquant_null_scan")


def test_maxquant_tmt_null_scan_index_is_shared_by_ident_and_quant(tmp_path):
    """The two frames the reader intersects must format the scan number identically."""
    path = tmp_path / "msms.txt"
    _write_maxquant_tmt_file(path, ["101", "", "103"])

    converter = MaxQuantDataFrameConverter()
    _, identification_df = converter.convert([path], as_polars=True)
    reader = MaxTmtReader(identification_file=None, identification_df=identification_df)

    normalised_identification = reader._normalise_identification_df(identification_df)
    quantification = reader._extract_quant_from_raw(identification_df)

    assert list(normalised_identification.index) == list(quantification.index)
