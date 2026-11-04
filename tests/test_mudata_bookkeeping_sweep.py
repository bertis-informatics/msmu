"""Public-API sweep for msmu's MuData bookkeeping: every call that changes var or obs must hand back a
MuData that is already in the state mudata's own copy() produces (bookkeeping and provenance hash), so a
checkpoint written, re-loaded, copied or updated between two calls keeps its hash; copy_mudata() must
reproduce that state. A call must also leave its input unchanged, unless it returns that same object (the
in-place dt.* calls).

Runs the tutorial pipelines on the shipped DIA-NN, Sage DDA-LFQ and Sage TMT data, plus the var/obs
changing calls those pipelines do not reach (a sample annotation outside the SDRF, batch correction, an
embedding, scaling, observation filters, FASTA mapping, precursor purity, collapse_obs, split_tmt, to_ptm,
adjust_ptm_by_protein, reindex_obs, dt.drop, dt.concat).
"""

from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
from _mudata_bookkeeping import assert_bookkeeping_is_final

import msmu as mm
from msmu._core._hashing import compute_hash
from msmu._tools import _pyopenms

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
Q_VALUE_FILTER = {"column": "q_value", "keep": "lt", "value": 0.01}


def _apply_q_value_filter(mdata: md.MuData, modality: str, on: str | None = None) -> md.MuData:
    mdata = mm.pp.add_filter(mdata, modality=modality, **Q_VALUE_FILTER)
    if on is None:
        return mm.pp.apply_filter(mdata, modality=modality)
    return mm.pp.apply_filter(mdata, modality=modality, on=on)


def _apply_obs_filter(mdata: md.MuData, modality: str, column: str, value: str) -> md.MuData:
    mdata = mm.pp.add_filter(mdata, modality=modality, column=column, keep="eq", value=value, on="obs")
    return mm.pp.apply_filter(mdata, modality=modality, on="obs")


def _cohort_frame(mdata: md.MuData) -> pd.DataFrame:
    """A sample annotation outside the SDRF: the first half of the runs in cohort A, the rest in B."""
    run_names = mdata["psm"].obs_names
    half = len(run_names) // 2
    return pd.DataFrame({"cohort": ["A" if i < half else "B" for i in range(len(run_names))]}, index=run_names)


def _run_and_check(steps: list[tuple[str, callable]]) -> md.MuData:
    """Run each (label, step) in order; after every step the result must already be in the state mudata's
    copy() produces, and the step must not have changed its input (unless it returned that same object)."""
    mdata = None
    for label, step in steps:
        input_hash = compute_hash(mdata) if mdata is not None else None
        result = step(mdata)
        if mdata is not None and result is not mdata:
            assert compute_hash(mdata) == input_hash, f"{label} changed its input"
        assert_bookkeeping_is_final(result, label)
        mdata = result
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


def test_annotation_tool_and_observation_filter_calls_copy_like_mudata() -> None:
    """The DIA data through the calls the tutorial pipeline does not reach: SDRF applied without reordering,
    add_filter on its own, a sample annotation outside the SDRF, batch correction, an embedding, scaling,
    and an observation filter that drops samples from one modality only."""
    dataset = DATA_DIR / "diann_dia"
    steps = [
        ("read_diann", lambda _: mm.read_diann(identification_file=str(dataset / "diann" / "report.parquet"))),
        ("attach_sdrf", lambda m: mm.pp.attach_sdrf(m, dataset / "meta.sdrf.tsv", validate=False)),
        ("apply_sdrf_to_obs", lambda m: mm.pp.apply_sdrf_to_obs(m, order_obs=False)),
        ("add_filter psm", lambda m: mm.pp.add_filter(m, modality="psm", **Q_VALUE_FILTER)),
        ("apply_filter psm", lambda m: mm.pp.apply_filter(m, modality="psm")),
        ("add_meta", lambda m: mm.pp.add_meta(m, _cohort_frame(m))),
        ("log2_transform", lambda m: mm.pp.log2_transform(m, modality="psm")),
        ("to_peptide", lambda m: mm.pp.to_peptide(m)),
        ("infer_protein", lambda m: mm.pp.infer_protein(m)),
        ("to_protein", lambda m: mm.pp.to_protein(m, top_n=3, rank_method="total_intensity")),
        (
            "correct_batch_effect",
            lambda m: mm.pp.correct_batch_effect(m, modality="protein", method="median_center", category="cohort"),
        ),
        ("tl.umap", lambda m: mm.tl.umap(m, modality="protein", n_neighbors=2, random_state=0)),
        ("scale_data", lambda m: mm.pp.scale_data(m, modality="protein")),
        ("apply_filter protein on obs", lambda m: _apply_obs_filter(m, "protein", "cohort", "A")),
        ("tl.corr", lambda m: mm.tl.corr(m, modality="protein")),
    ]
    _run_and_check(steps)


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
        # drops samples from the middle modality only; psm and protein keep every sample
        ("apply_filter peptide on obs", lambda m: _apply_obs_filter(m, "peptide", "factor value[condition]", "G1")),
    ]
    # read_sage with an LFQ table already creates a peptide modality, so to_peptide replaces it.
    _run_and_check(steps)


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
        ("attach_fasta", lambda m: mm.utils.attach_fasta(m, str(DATA_DIR / "fasta" / "human_fasta.fasta"))),
        ("map_fasta protein", lambda m: mm.utils.map_fasta(m, modality="protein", categories=["Gene"])),
        ("map_fasta peptide", lambda m: mm.utils.map_fasta(m, modality="peptide", categories=["Gene"])),
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


def test_select_repr_protein_copies_like_mudata() -> None:
    protein_adata = ad.AnnData(
        X=np.array([[1.0, 2.0]]), obs=pd.DataFrame(index=["s1"]), var=pd.DataFrame(index=["P1", "P2-2,P3"])
    )
    mdata = md.MuData({"protein": protein_adata})
    mdata.uns["protein_info"] = pd.DataFrame({"Entry Type": ["sp", "sp", "tr"]}, index=["P1", "P2-2", "P3"])

    _run_and_check([("select_repr_protein", lambda _: mm.utils.select_repr_protein(mdata, modality="protein"))])


def test_compute_precursor_isolation_purity_copies_like_mudata(monkeypatch) -> None:
    class FakePrecursorPurityCalculator:
        mzml = None

        @classmethod
        def from_mudata(cls, mdata, tolerance=20.0, unit_ppm=True):
            return cls()

        def calculate_precursor_isolation_purities(self):
            return pd.DataFrame({"filename": ["run1.mzML", "run1.mzML"], "scan_num": [1, 2], "purity": [0.8, 0.9]})

    monkeypatch.setattr(_pyopenms, "PrecursorPurityCalculator", FakePrecursorPurityCalculator)
    psm_var = pd.DataFrame({"filename": ["run1", "run1"], "scan_num": [1, 2]}, index=["run1.1", "run1.2"])
    psm_adata = ad.AnnData(X=np.ones((1, 2)), obs=pd.DataFrame(index=["s1"]), var=psm_var)
    mdata = md.MuData({"psm": psm_adata})

    _run_and_check(
        [
            (
                "compute_precursor_isolation_purity",
                lambda _: mm.tl.compute_precursor_isolation_purity(mdata, mzml_paths=["/data/run1.mzML"]),
            )
        ]
    )


def test_adjust_ptm_by_protein_copies_like_mudata(ptm_mdata: md.MuData, global_mdata: md.MuData) -> None:
    _run_and_check(
        [
            (
                "adjust_ptm_by_protein",
                lambda _: mm.pp.adjust_ptm_by_protein(ptm_mdata, global_mdata, modality="phospho_site", rescale=False),
            )
        ]
    )


@pytest.mark.xfail(
    strict=True,
    reason="reindex_obs leaves the modality obsm frames on the old obs index; copying re-aligns them (separate defect)",
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
