"""A workflow resumed from a checkpoint taken between two recorded calls must replay with verification.

``to_peptide`` and ``to_protein`` return a MuData whose MuData-level bookkeeping (modality maps, membership
masks, global var) used to be finished only by the next msmu call. Writing, re-loading, copying or updating
the returned object in between therefore changed its provenance hash, the next recorded call saw an input
that no recorded output matched, and ``pv.replay(verify=True)`` refused the history.
"""

from pathlib import Path

import mudata as md
import pytest

import msmu as mm
from msmu._core._hashing import compute_hash

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
Q_VALUE_FILTER = {"column": "q_value", "keep": "lt", "value": 0.01}


def _apply_q_value_filter(mdata: md.MuData, modality: str, on: str = "all") -> md.MuData:
    mdata = mm.pp.add_filter(mdata, modality=modality, **Q_VALUE_FILTER)
    return mm.pp.apply_filter(mdata, modality=modality, on=on)


def _read_dia() -> md.MuData:
    return mm.read_diann(identification_file=str(DATA_DIR / "diann_dia" / "diann" / "report.parquet"))


def _read_dda() -> md.MuData:
    # The LFQ table already creates a peptide modality, so to_peptide replaces it instead of adding one.
    dataset = DATA_DIR / "sage_lfq" / "sage"
    return mm.read_sage(
        identification_file=str(dataset / "results.sage.tsv"),
        quantification_file=str(dataset / "lfq.tsv"),
        label="label_free",
    )


def _to_peptide(read) -> md.MuData:
    return mm.pp.to_peptide(_apply_q_value_filter(read(), "psm"))


def _to_protein(read, rank_method: str) -> md.MuData:
    peptide = _apply_q_value_filter(_to_peptide(read), "peptide", on="var")
    return mm.pp.to_protein(mm.pp.infer_protein(peptide), top_n=3, rank_method=rank_method)


# checkpoint name -> (modality the resumed workflow filters, how to build the checkpoint)
CHECKPOINTS = {
    "dia to_peptide": ("peptide", lambda: _to_peptide(_read_dia)),
    "dia to_protein": ("protein", lambda: _to_protein(_read_dia, "total_intensity")),
    "dda to_peptide": ("peptide", lambda: _to_peptide(_read_dda)),
    "dda to_protein": ("protein", lambda: _to_protein(_read_dda, "median_intensity")),
}


def _save_and_continue(mdata: md.MuData, path: Path) -> md.MuData:
    mdata.write_h5mu(path)  # mudata updates the object in place after writing
    return mdata


def _save_and_reload(mdata: md.MuData, path: Path) -> md.MuData:
    mdata.write_h5mu(path)
    return mm.read_h5mu(path)


def _copy(mdata: md.MuData, path: Path) -> md.MuData:
    return mdata.copy()


def _update(mdata: md.MuData, path: Path) -> md.MuData:
    mdata.update()
    return mdata


RESUMPTIONS = {
    "save and continue": _save_and_continue,
    "save and reload": _save_and_reload,
    "copy": _copy,
    "update": _update,
}


@pytest.mark.parametrize("resumption", RESUMPTIONS)
@pytest.mark.parametrize("checkpoint_name", CHECKPOINTS)
def test_workflow_resumed_from_checkpoint_replays_with_verification(
    checkpoint_name: str, resumption: str, tmp_path: Path
) -> None:
    modality, build_checkpoint = CHECKPOINTS[checkpoint_name]
    checkpoint = build_checkpoint()
    checkpoint_hash = compute_hash(checkpoint)

    resumed = RESUMPTIONS[resumption](checkpoint, tmp_path / "checkpoint.h5mu")
    # The object must come back unchanged. A difference here is a copy or round-trip defect, not the
    # unfinished bookkeeping this test is about.
    assert compute_hash(resumed) == checkpoint_hash
    result = _apply_q_value_filter(resumed, modality, on="var")

    replayed = mm.pv.replay(result, verify=True)

    assert compute_hash(replayed) == compute_hash(result)
