"""Sage (TMT) reader: polars path == pandas path (Phase 0 oracle).

Clean-data parity is the reference that must survive the reader separation. The regression cases pin
the §A null/format fixes: a scannr with no ``scan=`` token now fails loudly (A-1), a zero-padded scan
maps to the same ident/quant index (A-2), and a null ``label`` resolves to target rather than
dropping the row from both target and decoy (A-4).
"""

from __future__ import annotations

import pandas as pd
import pytest

from msmu._read_write._base_reader import SearchResultDataFrameConverter
from msmu._read_write._sage import TmtSageReader

from _parity_helpers import assert_polars_matches_golden

PLEX = 6


@pytest.fixture(autouse=True)
def _legacy_string_dtype():
    with pd.option_context("future.infer_string", False):
        yield


# (peptide, protein, charge); one decoy is appended in the frame builder.
_PEPTIDES = [
    ("AAAKPGGR", "sp|P11111|A_HUMAN", 2),
    ("KKPTTTR", "sp|P22222|B_HUMAN", 2),
    ("MSSSSKR", "sp|P33333|C_HUMAN", 3),
]

_RUNS = ("runA.mzML", "runB.mzML")


def _scannr(scan_token: str) -> str:
    return f"controllerType=0 controllerNumber=1 {scan_token}"


def _rows(scan_tokens, labels):
    """(ident_rows, quant_rows) sharing filename+scannr so the reader's index intersects."""
    ident_rows, quant_rows = [], []
    cursor = 0
    for run in _RUNS:
        for (peptide, protein, charge) in _PEPTIDES:
            scannr = _scannr(scan_tokens[cursor])
            ident_rows.append(
                {
                    "filename": run,
                    "scannr": scannr,
                    "peptide": peptide,
                    "proteins": protein,
                    "label": labels[cursor],
                    "expmass": 1000.0 + cursor,
                    "calcmass": 1000.5 + cursor,
                    "charge": charge,
                    "peptide_len": len(peptide),
                    "missed_cleavages": 0,
                    "semi_enzymatic": 0,
                    "hyperscore": 20.0 + cursor,
                    "posterior_error": -2.0,
                    "spectrum_q": 0.001,
                    "rt": 10.0 + cursor,
                }
            )
            quant_row = {"filename": run, "scannr": scannr, "ion_injection_time": 50.0}
            for channel in range(1, PLEX + 1):
                quant_row[f"tmt_{channel}"] = float(channel) * (cursor + 1)
            quant_rows.append(quant_row)
            cursor += 1
    return pd.DataFrame(ident_rows), pd.DataFrame(quant_rows)


def _clean_scan_tokens():
    return [f"scan={1001 + i}" for i in range(len(_RUNS) * len(_PEPTIDES))]


def _clean_labels():
    labels = [1] * (len(_RUNS) * len(_PEPTIDES))
    labels[-1] = -1  # one decoy, to exercise decoy separation on both paths
    return labels


def _write(path, frame):
    frame.to_csv(path, sep="\t", index=False)


def _read(ident_path, quant_path):
    converter = SearchResultDataFrameConverter()
    ident_file, ident_df = converter.convert([ident_path])
    quant_file, quant_df = converter.convert([quant_path])
    return TmtSageReader(
        identification_file=ident_file,
        identification_df=ident_df,
        quantification_file=quant_file,
        quantification_df=quant_df,
    ).read()


def _write_pair(tmp_path, scan_tokens, labels):
    ident_df, quant_df = _rows(scan_tokens, labels)
    ident_path, quant_path = tmp_path / "results.sage.tsv", tmp_path / "tmt.tsv"
    _write(ident_path, ident_df)
    _write(quant_path, quant_df)
    return ident_path, quant_path


def test_sage_tmt_polars_matches_pandas_clean(tmp_path):
    """Clean TMT input: identical features, names, channels and intensities on both paths."""
    ident_path, quant_path = _write_pair(tmp_path, _clean_scan_tokens(), _clean_labels())
    assert_polars_matches_golden(lambda: _read(ident_path, quant_path), "sage_tmt_clean")


def test_sage_tmt_malformed_scannr_raises_clearly(tmp_path):
    """A scannr with no ``scan=<n>`` token (Sciex/Bruker) must fail loudly, not silently null.

    The policy for a malformed scan identifier is fail-loud (matching the old pandas ``int()``),
    so the fixed behaviour is a clear, scannr-identifying error -- not a MuData equal to pandas
    (pandas raises too). Today the polars path instead nulls the scan, empties the ident/quant
    intersection, and dies with an opaque error; this asserts the fixed contract.
    """
    tokens = _clean_scan_tokens()
    tokens[1] = "index=5"  # no scan= token
    ident_path, quant_path = _write_pair(tmp_path, tokens, _clean_labels())
    with pytest.raises(ValueError, match="scan"):
        _read(ident_path, quant_path)


def test_sage_tmt_leading_zero_scan_parity(tmp_path):
    """A zero-padded scan (scan=001001) must map to the same index on ident and quant sides."""
    tokens = _clean_scan_tokens()
    tokens[0] = "scan=001001"
    ident_path, quant_path = _write_pair(tmp_path, tokens, _clean_labels())
    assert_polars_matches_golden(lambda: _read(ident_path, quant_path), "sage_tmt_leading_zero")


def test_sage_tmt_null_label_parity(tmp_path):
    """A null ``label`` must resolve to target (decoy=0) as pandas does, not vanish."""
    labels = _clean_labels()
    labels[1] = None  # null label
    ident_path, quant_path = _write_pair(tmp_path, _clean_scan_tokens(), labels)
    assert_polars_matches_golden(lambda: _read(ident_path, quant_path), "sage_tmt_null_label")
