"""Public-API sweep for msmu's MuData copy: every call that changes var or obs must hand back a MuData
that copy_mudata() brings to exactly the state mudata's own copy() produces (bookkeeping and provenance
hash), and must not leave the global var stale.

Runs the tutorial pipelines on the shipped DIA-NN, Sage DDA-LFQ and Sage TMT data, plus the var/obs
changing calls those pipelines do not reach (collapse_obs, split_tmt, to_ptm, adjust_ptm_by_protein,
reindex_obs, dt.drop, dt.concat).
"""

from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
from _mudata_bookkeeping import assert_copy_matches_mudata_copy

import msmu as mm
from msmu._core._hashing import compute_hash
from msmu._utils._mudata import _is_global_var_stale

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
Q_VALUE_FILTER = {"column": "q_value", "keep": "lt", "value": 0.01}


def _apply_q_value_filter(mdata: md.MuData, modality: str, on: str | None = None) -> md.MuData:
    mdata = mm.pp.add_filter(mdata, modality=modality, **Q_VALUE_FILTER)
    if on is None:
        return mm.pp.apply_filter(mdata, modality=modality)
    return mm.pp.apply_filter(mdata, modality=modality, on=on)


def _run_and_check(steps: list[tuple[str, callable]], stale_allowed: frozenset[str] = frozenset()) -> md.MuData:
    """Run each (label, step) in order; after every step the result must copy like mudata's copy().

    A step in ``stale_allowed`` may return a stale global var (it did before copy_mudata existed and
    its recorded output hash depends on that); the copy must still heal it.
    """
    mdata = None
    for label, step in steps:
        mdata = step(mdata)
        if label not in stale_allowed:
            assert not _is_global_var_stale(mdata), label
        assert_copy_matches_mudata_copy(mdata, label)
    return mdata


def _summarisation_steps(top_n_rank: dict, protein_kwargs: dict) -> list[tuple[str, callable]]:
    return [
        ("to_peptide", lambda m: mm.pp.to_peptide(m)),
        ("apply_filter peptide", lambda m: _apply_q_value_filter(m, "peptide", on="var")),
        ("infer_protein", lambda m: mm.pp.infer_protein(m)),
        ("to_protein", lambda m: mm.pp.to_protein(m, **top_n_rank, **protein_kwargs)),
        ("apply_filter protein", lambda m: _apply_q_value_filter(m, "protein", on="var")),
        ("calculate_missingness", lambda m: mm.pp.calculate_missingness(m, modality="protein")),
        ("tl.corr", lambda m: mm.tl.corr(m, modality="protein")),
        ("tl.pca", lambda m: mm.tl.pca(m, modality="protein", n_components=2)),
    ]


def _dia_lfq_steps() -> list[tuple[str, callable]]:
    dataset = DATA_DIR / "diann_dia"
    return [
        ("read_diann", lambda _: mm.read_diann(identification_file=str(dataset / "diann" / "report.parquet"))),
        ("attach_sdrf", lambda m: mm.pp.attach_sdrf(m, dataset / "meta.sdrf.tsv", validate=False)),
        ("apply_sdrf_to_obs", lambda m: mm.pp.apply_sdrf_to_obs(m)),
        ("apply_filter psm", lambda m: _apply_q_value_filter(m, "psm")),
        ("log2_transform", lambda m: mm.pp.log2_transform(m, modality="psm")),
        ("normalise", lambda m: mm.pp.normalise(m, modality="psm", method="median")),
        *_summarisation_steps({"top_n": 3, "rank_method": "total_intensity"}, {}),
        ("dt.drop psm decoy", lambda m: mm.dt.drop(m, target="psm.uns", key="decoy")),
    ]


def test_dia_lfq_pipeline_copies_like_mudata() -> None:
    _run_and_check(_dia_lfq_steps())


def test_dia_lfq_pipeline_replays_with_verification_after_h5mu_round_trip(tmp_path) -> None:
    """The per-step hashes recorded through copy_mudata must verify on replay, also after a file round trip."""
    recorded = _run_and_check(_dia_lfq_steps())
    path = tmp_path / "recorded.h5mu"
    recorded.write_h5mu(path)
    reread = mm.read_h5mu(path)

    replayed = mm.pv.replay(reread, verify=True)

    assert compute_hash(replayed) == compute_hash(reread) == compute_hash(recorded)


def test_dda_lfq_pipeline_copies_like_mudata() -> None:
    dataset = DATA_DIR / "sage_lfq"
    steps = [
        (
            "read_sage",
            lambda _: mm.read_sage(
                identification_file=str(dataset / "sage" / "results.sage.tsv"),
                quantification_file=str(dataset / "sage" / "lfq.tsv"),
                label="label_free",
            ),
        ),
        ("attach_sdrf", lambda m: mm.pp.attach_sdrf(m, dataset / "meta.sdrf.tsv", validate=False)),
        ("apply_sdrf_to_obs", lambda m: mm.pp.apply_sdrf_to_obs(m)),
        ("apply_filter psm", lambda m: _apply_q_value_filter(m, "psm")),
        ("to_peptide", lambda m: mm.pp.to_peptide(m)),
        ("apply_filter peptide", lambda m: _apply_q_value_filter(m, "peptide", on="var")),
        ("log2_transform", lambda m: mm.pp.log2_transform(m, modality="peptide")),
        ("normalise", lambda m: mm.pp.normalise(m, modality="peptide", method="median")),
        ("infer_protein", lambda m: mm.pp.infer_protein(m)),
        ("to_protein", lambda m: mm.pp.to_protein(m, top_n=3, rank_method="median_intensity")),
        ("apply_filter protein", lambda m: _apply_q_value_filter(m, "protein", on="var")),
    ]
    # read_sage with an LFQ table already creates a peptide modality, so to_peptide replaces it in
    # place and leaves the global var stale until the next copy (see _summarise.py).
    _run_and_check(steps, stale_allowed=frozenset({"to_peptide"}))


def test_tmt_pipeline_with_split_tmt_copies_like_mudata() -> None:
    dataset = DATA_DIR / "sage_tmt"

    def split_into_one_set(mdata: md.MuData) -> md.MuData:
        filenames = mdata["psm"].var["filename"].unique().tolist()
        return mm.pp.split_tmt(mdata, {filename: "set1" for filename in filenames})

    steps = [
        (
            "read_sage tmt",
            lambda _: mm.read_sage(
                identification_file=str(dataset / "sage" / "results.sage.tsv"),
                quantification_file=str(dataset / "sage" / "tmt.tsv"),
                label="tmt",
            ),
        ),
        ("attach_sdrf", lambda m: mm.pp.attach_sdrf(m, dataset / "meta.sdrf.tsv", validate=False)),
        ("apply_sdrf_to_obs", lambda m: mm.pp.apply_sdrf_to_obs(m)),
        ("apply_filter psm", lambda m: _apply_q_value_filter(m, "psm")),
        ("split_tmt", split_into_one_set),
        ("log2_transform", lambda m: mm.pp.log2_transform(m, modality="psm")),
        ("normalise", lambda m: mm.pp.normalise(m, modality="psm", method="median")),
        *_summarisation_steps({}, {"agg_method": "median_polish"}),
    ]
    _run_and_check(steps)


def test_collapse_obs_copies_like_mudata() -> None:
    obs = pd.DataFrame(
        {"sample": ["s1", "s1", "s2", "s2"], "filename": ["s1_f1", "s1_f2", "s2_f1", "s2_f2"]},
        index=["s1_f1", "s1_f2", "s2_f1", "s2_f2"],
    )
    x = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, 5.0], [10.0, 20.0, np.nan], [np.nan, np.nan, np.nan]])
    psm_adata = ad.AnnData(X=x, obs=obs, var=pd.DataFrame({"protein": ["P1", "P2", "P3"]}, index=["p1", "p2", "p3"]))
    peptide_adata = ad.AnnData(X=x[:, :2].copy(), obs=obs.copy(), var=pd.DataFrame(index=["pepA", "pepB"]))
    fractionated = md.MuData({"psm": psm_adata, "peptide": peptide_adata})

    _run_and_check([("collapse_obs", lambda _: mm.pp.collapse_obs(fractionated, sample_key="sample"))])


def test_to_ptm_copies_like_mudata() -> None:
    phospho = "[+79.97]"
    var = pd.DataFrame(
        [{"peptide": f"SPGS{phospho}PVLR", "stripped_peptide": "SPGSPVLR", "proteins": "P1", "count_psm": 5}]
    ).set_index("peptide", drop=False)
    var.index.name = None
    peptide_adata = ad.AnnData(X=np.array([[10.0], [11.0]]), obs=pd.DataFrame(index=["s1", "s2"]), var=var)
    peptide_mdata = md.MuData({"peptide": peptide_adata})
    peptide_mdata.uns["protein_info"] = pd.DataFrame({"Sequence": ["MAAASPGSPVLRKKQ"]}, index=["P1"])

    _run_and_check([("to_ptm", lambda _: mm.pp.to_ptm(peptide_mdata, modi_name="phospho", modification=phospho))])


def test_adjust_ptm_by_protein_copies_like_mudata(ptm_mdata: md.MuData, global_mdata: md.MuData) -> None:
    _run_and_check(
        [
            (
                "adjust_ptm_by_protein",
                lambda _: mm.pp.adjust_ptm_by_protein(ptm_mdata, global_mdata, modality="phospho_site", rescale=False),
            )
        ]
    )


def test_reindex_obs_copies_like_mudata(mdata: md.MuData) -> None:
    _run_and_check([("reindex_obs", lambda _: mm.utils.reindex_obs(mdata, column="sample"))])


def test_dt_concat_copies_like_mudata(mdata_factory) -> None:
    inputs = {"first": mdata_factory("first"), "second": mdata_factory("second")}
    _run_and_check([("dt.concat", lambda _: mm.dt.concat(inputs))])


@pytest.mark.parametrize("modality", ["psm", "protein"])
def test_in_place_data_calls_keep_the_copy_invariant(mdata: md.MuData, modality: str) -> None:
    steps = [
        ("dt.save_layer", lambda _: mm.dt.save_layer(mdata, modality=modality, layer="raw")),
        ("dt.assign", lambda m: mm.dt.assign(m, "flag", 1, modality=modality)),
        ("dt.drop var column", lambda m: mm.dt.drop(m, target=f"{modality}.var", key="flag")),
    ]
    _run_and_check(steps)
