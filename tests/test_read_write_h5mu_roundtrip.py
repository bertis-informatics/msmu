"""Production ``write_h5mu`` robustness: a reader MuData must round-trip through .h5mu unaided.

Regression for the latent h5mu-serialisation bug found while building reader parity goldens. Two
independent failure modes on the anndata-0.13 / pandas-3 stack:

1. The raw search result kept in ``varm['search_result']`` preserves the search engine's original
   column names, and MaxQuant's ``MS/MS Scan Number`` (like FragPipe's ``Observed M/Z``) contains a
   ``/``, which h5py forbids as a dataset key -- so ``mdata.write_h5mu(...)`` raised on EVERY
   MaxQuant read, null scan or not.
2. The reader frames (polars -> pandas) carry pandas nullable / Arrow-backed string columns
   (including the obs/var index) that anndata refuses to write under ``infer_string=False`` unless
   ``anndata.settings.allow_write_nullable_strings`` is opted in.

These tests exercise the PUBLIC path (``read_maxquant`` -> ``mdata.write_h5mu`` -> ``read_h5mu``)
WITHOUT the parity helper and WITHOUT setting any anndata flag by hand, so they prove production is
self-sufficient: importing msmu must be enough.
"""

import anndata
import numpy as np
import pandas as pd
import pytest

import msmu as mm
from msmu._read_write._base_reader import SearchResultReader

from _parity_helpers import assert_reader_mdata_equal

PLEX = 6
SCAN_COLUMN = "MS/MS Scan Number"


def _write_maxquant_tmt_file(path, scan_numbers) -> None:
    """Minimal MaxQuant TMT msms table (mirrors tests/test_read_write_maxquant_null_scan.py)."""
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


@pytest.mark.parametrize(
    "scan_numbers, case",
    [
        (["101", "102", "103"], "all-integer scans"),
        (["101", "", "103"], "one null scan"),
    ],
)
def test_read_maxquant_write_h5mu_round_trips(tmp_path, scan_numbers, case):
    """read_maxquant -> write_h5mu -> read_h5mu succeeds and preserves .X, var and varm."""
    source = tmp_path / "msms.txt"
    _write_maxquant_tmt_file(source, scan_numbers)

    mdata = mm.read_maxquant(str(source), label="tmt", acquisition="dda")
    assert mdata["psm"].shape == (PLEX, len(scan_numbers)), f"features lost before write ({case})"
    assert "search_result" in mdata["psm"].varm, "search_result must be materialised in varm by default"

    out = tmp_path / "out.h5mu"
    mdata.write_h5mu(str(out))  # must not raise: the production path is self-sufficient
    assert out.exists()

    restored = mm.read_h5mu(str(out))

    # .X (NaN-aware), obs/var names and every var column survive the round-trip.
    assert_reader_mdata_equal(restored, mdata)

    # varm survived, and its "/"-bearing raw name was rewritten to an h5-safe form.
    restored_search_result = restored["psm"].varm["search_result"]
    assert not any("/" in str(col) for col in restored_search_result.columns)
    assert "MS_MS Scan Number" in restored_search_result.columns
    assert SCAN_COLUMN not in restored_search_result.columns


def test_make_h5_safe_varm_columns_replaces_slash_but_keeps_values():
    """"/"-bearing names are rewritten; "/"-free names and all values are untouched."""
    search_result_df = pd.DataFrame(
        {
            "MS/MS Scan Number": [1, 2],
            "Observed M/Z": [3.0, 4.0],
            "clean": ["a", "b"],
        }
    )
    out = SearchResultReader._make_h5_safe_varm_columns(search_result_df)

    assert list(out.columns) == ["MS_MS Scan Number", "Observed M_Z", "clean"]
    assert out["MS_MS Scan Number"].tolist() == [1, 2]
    assert out["clean"].tolist() == ["a", "b"]


def test_make_h5_safe_varm_columns_disambiguates_collision():
    """A rewrite that collides with an existing name gets a suffix; the clean name is kept."""
    search_result_df = pd.DataFrame({"A_B": [1, 2], "A/B": [3, 4]})

    out = SearchResultReader._make_h5_safe_varm_columns(search_result_df)

    assert list(out.columns) == ["A_B", "A_B_1"]
    assert out["A_B"].tolist() == [1, 2]
    assert out["A_B_1"].tolist() == [3, 4]


def test_make_h5_safe_varm_columns_is_noop_without_slash():
    """A frame with no "/" in any name is returned unchanged."""
    search_result_df = pd.DataFrame({"a": [1], "b": [2]})

    out = SearchResultReader._make_h5_safe_varm_columns(search_result_df)

    assert out is search_result_df


def test_importing_msmu_opts_into_nullable_string_writes():
    """Importing msmu makes write_h5mu robust to infer_string=False (task failure mode 2)."""
    assert anndata.settings.allow_write_nullable_strings is True
