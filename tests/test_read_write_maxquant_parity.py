"""MaxQuant reader: the polars-vs-pandas divergences beyond the null-scan case.

The null-scan divergence is covered in ``test_read_write_maxquant_null_scan.py``. This pins the two
fixed in Phase 1: §A-5b (a null ``Type`` row must be kept, not dropped only on polars) and §A-6 (a
null ``Raw file`` must not form a phantom ``"null"`` sample on the polars pivot).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msmu._read_write._maxquant import MaxQuantDataFrameConverter, MaxTmtReader, MaxLfqReader

from _parity_helpers import assert_polars_matches_golden

PLEX = 6


@pytest.fixture(autouse=True)
def _legacy_string_dtype():
    with pd.option_context("future.infer_string", False):
        yield


def _base_columns(n_rows, raw_files=None, types=None):
    raw_files = raw_files if raw_files is not None else ["raw_0"] * n_rows
    types = types if types is not None else ["MULTI-MSMS"] * n_rows
    return {
        "Type": types,
        "Reverse": [None] * n_rows,
        "Potential contaminant": [None] * n_rows,
        "Proteins": [f"sp|P{10000 + i}|X{i}_HUMAN" for i in range(n_rows)],
        "Leading proteins": [f"sp|P{10000 + i}|X{i}_HUMAN" for i in range(n_rows)],
        "Sequence": [f"PEPTIDEK{i}" for i in range(n_rows)],
        "Modified sequence": [f"_PEPTIDEK{i}_" for i in range(n_rows)],
        "Length": [9] * n_rows,
        "Missed cleavages": [0] * n_rows,
        "Charge": [2] * n_rows,
        "Raw file": raw_files,
        "MS/MS Scan Number": [str(101 + i) for i in range(n_rows)],
        "Retention time": [10.0 + i for i in range(n_rows)],
        "PEP": [0.001] * n_rows,
    }


def _write(path, frame):
    frame.to_csv(path, sep="\t", index=False)


def _read_tmt(path, *, as_polars):
    converter = MaxQuantDataFrameConverter()
    identification_file, identification_df = converter.convert([path], as_polars=as_polars)
    return MaxTmtReader(identification_file=identification_file, identification_df=identification_df).read()


def _read_lfq(path, *, as_polars):
    converter = MaxQuantDataFrameConverter()
    identification_file, identification_df = converter.convert([path], as_polars=as_polars)
    return MaxLfqReader(identification_file=identification_file, identification_df=identification_df).read()


def test_maxquant_tmt_null_type_parity(tmp_path):
    """A row whose ``Type`` is null must be kept on both paths (pandas ``~isin`` keeps NaN)."""
    columns = _base_columns(3, types=["MULTI-MSMS", None, "MULTI-MSMS"])
    frame = pd.DataFrame(columns)
    for channel in range(1, PLEX + 1):
        frame[f"Reporter intensity corrected {channel}"] = np.arange(1, 4, dtype=float) * channel
    path = tmp_path / "msms.txt"
    _write(path, frame)

    assert_polars_matches_golden(lambda as_polars: _read_tmt(path, as_polars=as_polars), "maxquant_tmt_null_type")


def test_maxquant_lfq_null_raw_file_dropped(tmp_path):
    """A null ``Raw file`` must not become a phantom ``null`` sample (pandas pivot_table drops it).

    Asserts the polars-correct behaviour directly rather than against pandas: under this venv's
    pandas 3.0 the pandas path additionally crashes building an AnnData for a null-filename PSM, so
    it cannot serve as the oracle here. The fix drops the null key before the pivot.
    """
    columns = _base_columns(3, raw_files=["raw_0", None, "raw_1"])
    frame = pd.DataFrame(columns)
    frame["Intensity"] = [100.0, 200.0, 300.0]
    path = tmp_path / "msms.txt"
    _write(path, frame)

    peptide = _read_lfq(path, as_polars=True)["peptide"]
    assert list(peptide.obs_names) == ["raw_0", "raw_1"], "null Raw file must not form a phantom sample"
