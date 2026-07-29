"""DIA-NN reader: polars path == pandas path (Phase 0 oracle).

Captures the pandas path as the reference while it still exists. Beyond clean-data parity, the
regression cases pin the parser-default divergences fixed in Phase 1: an entirely-empty Lib.Q.Value
column (§B-3) and a multi-file batch mixing files with/without the optional Decoy column (§B-4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msmu._read_write._base_reader import SearchResultDataFrameConverter
from msmu._read_write._diann import DiannReader

from _parity_helpers import assert_reader_mdata_equal


@pytest.fixture(autouse=True)
def _legacy_string_dtype():
    # Capture the historical (pandas 2.x) reader behaviour as the oracle. Under the pandas-3.0
    # Arrow-string default some pandas reader paths break for unrelated reasons (a separate
    # pandas-3 compat concern), so both paths are exercised with the legacy object-string dtype.
    with pd.option_context("future.infer_string", False):
        yield


_COLUMNS = [
    "Run",
    "Protein.Group",
    "Protein.Ids",
    "Modified.Sequence",
    "Stripped.Sequence",
    "Precursor.Id",
    "Precursor.Charge",
    "RT",
    "PEP",
    "Global.Q.Value",
    "Lib.Q.Value",
    "Decoy",
    "Precursor.Quantity",
]

_PEPTIDES = [
    ("AAAKPGGR", "sp|P11111|A_HUMAN", 2),
    ("KKPTTTR", "sp|P22222|B_HUMAN", 2),
    ("MSSSSKR", "sp|P33333|C_HUMAN", 3),
]


def _clean_frame() -> pd.DataFrame:
    rows = []
    for run in ("runA", "runB"):
        for index, (peptide, protein, charge) in enumerate(_PEPTIDES):
            rows.append(
                {
                    "Run": run,
                    "Protein.Group": protein,
                    "Protein.Ids": protein,
                    "Modified.Sequence": peptide,
                    "Stripped.Sequence": peptide,
                    "Precursor.Id": f"{peptide}{charge}",
                    "Precursor.Charge": charge,
                    "RT": 10.0 + index,
                    "PEP": 0.001,
                    "Global.Q.Value": 0.001,
                    "Lib.Q.Value": 0.0,
                    "Decoy": 0,
                    "Precursor.Quantity": 1000.0 * (index + 1) + (0.0 if run == "runA" else 500.0),
                }
            )
    return pd.DataFrame(rows, columns=_COLUMNS)


def _write(path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, sep="\t", index=False)


def _read(paths, *, as_polars: bool):
    if not isinstance(paths, list):
        paths = [paths]
    converter = SearchResultDataFrameConverter()
    identification_file, identification_df = converter.convert(paths, as_polars=as_polars)
    return DiannReader(identification_file=identification_file, identification_df=identification_df).read()


def test_diann_polars_matches_pandas_clean(tmp_path):
    """On clean multi-run input the two paths produce identical features, names and intensities."""
    path = tmp_path / "report.tsv"
    _write(path, _clean_frame())

    assert_reader_mdata_equal(_read(path, as_polars=True), _read(path, as_polars=False))


def test_diann_empty_libq_value_parity(tmp_path):
    """An entirely-empty Lib.Q.Value column (non-MBR run) must not break the polars path."""
    frame = _clean_frame()
    frame["Lib.Q.Value"] = ""  # written as blanks: pandas -> float NaN, polars -> String
    path = tmp_path / "report.tsv"
    _write(path, frame)

    assert_reader_mdata_equal(_read(path, as_polars=True), _read(path, as_polars=False))


def test_diann_multifile_optional_decoy_parity(tmp_path):
    """DIA-NN treats Decoy as optional, so a batch mixing files with/without it must still read."""
    with_decoy = _clean_frame()  # runA / runB, with a Decoy column
    without_decoy = _clean_frame().drop(columns=["Decoy"])
    without_decoy["Run"] = without_decoy["Run"].map({"runA": "runC", "runB": "runD"})  # distinct runs
    path_a = tmp_path / "with_decoy.tsv"
    path_b = tmp_path / "without_decoy.tsv"
    _write(path_a, with_decoy)
    _write(path_b, without_decoy)

    # Multi-file order is non-deterministic on the pandas ProcessPool path, so compare by name.
    assert_reader_mdata_equal(
        _read([path_a, path_b], as_polars=True),
        _read([path_a, path_b], as_polars=False),
        ordered=False,
    )
