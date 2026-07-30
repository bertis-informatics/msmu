"""DELPI reader: polars path == pandas path (Phase 0 oracle).

DELPI had no test coverage at all. Clean parity is the reference; the regression case pins the §A-3
fix -- the same ``cast(Utf8)`` index bug once present in MaxQuant: a null ``pmsm_index`` used to make
the polars quant index (``run.<int>``) disagree with the ident index (``run.<float-str>`` after the
to_pandas promotion), collapsing the ident/quant intersection so features vanished. The quant is now
built through the shared pandas tail so both indexes format identically.
"""

from __future__ import annotations

import pandas as pd
import pytest

from msmu._read_write._base_reader import SearchResultDataFrameConverter
from msmu._read_write._delpi import DelpiReader

from _parity_helpers import assert_polars_matches_golden


@pytest.fixture(autouse=True)
def _legacy_string_dtype():
    with pd.option_context("future.infer_string", False):
        yield


_PRECURSORS = [
    ("PEPTIDEKA", "sp|P11111|A_HUMAN", 2),
    ("PEPTIDEKB", "sp|P22222|B_HUMAN", 2),
    ("PEPTIDEKC", "sp|P33333|C_HUMAN", 3),
]

_RUNS = ("runA", "runB")


def _frame(pmsm_indices):
    rows = []
    cursor = 0
    for run in _RUNS:
        for (peptide, fasta_id, charge) in _PRECURSORS:
            rows.append(
                {
                    "run_name": run,
                    "pmsm_index": pmsm_indices[cursor],
                    "fasta_id": fasta_id,
                    "peptide": peptide,
                    "modified_sequence": peptide,
                    "frame_num": 100 + cursor,
                    "precursor_charge": charge,
                    "sequence_length": len(peptide),
                    "posterior_error": 0.01,
                    "global_precursor_q_value": 0.001,
                    "score": 10.0 + cursor,
                    "is_decoy": 1 if cursor == len(_RUNS) * len(_PRECURSORS) - 1 else 0,
                    "ms2_area": 1000.0 * (cursor + 1),
                }
            )
            cursor += 1
    return pd.DataFrame(rows)


def _clean_pmsm():
    return [str(i + 1) for i in range(len(_RUNS) * len(_PRECURSORS))]


def _write(path, frame):
    frame.to_csv(path, sep="\t", index=False)


def _read(path):
    converter = SearchResultDataFrameConverter()
    identification_file, identification_df = converter.convert([path])
    return DelpiReader(identification_file=identification_file, identification_df=identification_df).read()


def test_delpi_polars_matches_pandas_clean(tmp_path):
    """Clean merged input: identical features, names and intensities on both paths."""
    path = tmp_path / "delpi.tsv"
    _write(path, _frame(_clean_pmsm()))

    assert_polars_matches_golden(lambda: _read(path), "delpi_clean")


def test_delpi_null_pmsm_index_parity(tmp_path):
    """A null ``pmsm_index`` must format identically on the ident and quant sides (no feature loss)."""
    pmsm = _clean_pmsm()
    pmsm[1] = ""  # blank pmsm_index for one row
    path = tmp_path / "delpi.tsv"
    _write(path, _frame(pmsm))

    assert_polars_matches_golden(lambda: _read(path), "delpi_null_pmsm")
