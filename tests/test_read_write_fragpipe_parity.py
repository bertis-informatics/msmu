"""FragPipe (TMT) reader: polars path == pandas path (Phase 0 oracle).

The FragPipe pandas branch breaks under the pandas-3.0 Arrow-string default (an unrelated compat
issue), so both paths are exercised with the legacy object-string dtype to make the pandas path a
usable oracle. Clean parity is the reference; the regression case pins the §A-5a fix -- a null
``Spectrum`` now fails loudly (matching pandas' ``.split``), not silently null filename/scan.
"""

from __future__ import annotations

import pandas as pd
import pytest

from msmu._read_write._base_reader import SearchResultDataFrameConverter
from msmu._read_write._fragpipe import TmtFragPipeReader

from _parity_helpers import assert_reader_mdata_equal

_CHANNELS = ["126", "127N", "128C"]


@pytest.fixture(autouse=True)
def _legacy_string_dtype():
    with pd.option_context("future.infer_string", False):
        yield


_PEPTIDES = [
    ("PEPTIDEKA", "sp|P11111|A_HUMAN", 2),
    ("PEPTIDEKB", "sp|P22222|B_HUMAN", 2),
    ("PEPTIDEKC", "sp|P33333|C_HUMAN", 3),
]

_RUNS = ("runA", "runB")


def _frame(spectra):
    rows = []
    cursor = 0
    for run in _RUNS:
        for (peptide, protein, charge) in _PEPTIDES:
            row = {
                "Spectrum": spectra[cursor],
                "Protein": protein,
                "Mapped Proteins": protein,  # non-null so the pandas astype(str) path is exercised cleanly
                "Modified Peptide": peptide,
                "Peptide": peptide,
                "Retention": 600.0 + cursor,
                "Charge": charge,
                "Peptide Length": len(peptide),
                "Number of Missed Cleavages": 0,
                "Calculated Peptide Mass": 1000.0 + cursor,
                "observed mass": 1000.5 + cursor,
                "Hyperscore": 20.0 + cursor,
            }
            for channel_index, channel in enumerate(_CHANNELS, start=1):
                row[channel] = float(channel_index) * (cursor + 1)
            rows.append(row)
            cursor += 1
    return pd.DataFrame(rows)


def _clean_spectra():
    return [f"{run}.{100 + i}.{100 + i}.2" for i, run in enumerate(r for r in _RUNS for _ in _PEPTIDES)]


def _write(path, frame):
    frame.to_csv(path, sep="\t", index=False)


def _read(path, *, as_polars: bool):
    converter = SearchResultDataFrameConverter()
    identification_file, identification_df = converter.convert([path], as_polars=as_polars)
    return TmtFragPipeReader(identification_file=identification_file, identification_df=identification_df).read()


def test_fragpipe_tmt_polars_matches_pandas_clean(tmp_path):
    """Clean merged TMT input: identical features, channels and intensities on both paths."""
    path = tmp_path / "psm.tsv"
    _write(path, _frame(_clean_spectra()))

    assert_reader_mdata_equal(_read(path, as_polars=True), _read(path, as_polars=False))


def test_fragpipe_tmt_null_spectrum_raises_clearly(tmp_path):
    """A null ``Spectrum`` must fail loudly (pandas' ``.split`` raises), not silently corrupt the index."""
    spectra = _clean_spectra()
    spectra[1] = None
    path = tmp_path / "psm.tsv"
    _write(path, _frame(spectra))

    with pytest.raises((ValueError, AttributeError, TypeError)):
        _read(path, as_polars=True)
