import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData
from mudata import MuData

import msmu as mm
import msmu._utils as msmu_utils
from msmu._core._blockdiag import dense_block, to_dense_df, to_observed_sparse


def _make_tmt_mdata() -> MuData:
    obs = pd.DataFrame(index=["126", "127"])
    var = pd.DataFrame(
        {"filename": ["runA.raw", "runB.raw", "runA.raw", "runB.raw"]},
        index=["psm1", "psm2", "psm3", "psm4"],
    )
    adata = AnnData(
        X=np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        obs=obs,
        var=var,
        uns={"level": "psm", "label": "tmt"},
    )
    return MuData({"psm": adata})


def _make_tmt_psm_mdata(n_psm: int = 4000, n_channels: int = 6, n_sets: int = 5, seed: int = 7):
    """A richer PSM-level TMT MuData plus its filename->set map, for summarisation equivalence."""
    rng = np.random.default_rng(seed)
    files = [f"set{s}_f{f}.raw" for s in range(n_sets) for f in range(2)]
    fmap = {fn.rsplit(".", 1)[0]: f"set{i // 2}" for i, fn in enumerate(files)}
    n_pep = max(1, n_psm // 3)
    peptides = np.array([f"PEPTIDE{p}K" for p in range(n_pep)])
    pep_idx = rng.integers(0, n_pep, n_psm)
    var = pd.DataFrame(
        {
            "filename": rng.choice(files, n_psm),
            "peptide": peptides[pep_idx],
            "stripped_peptide": peptides[pep_idx],
            "proteins": [f"P{p % 500}" for p in pep_idx],
            "PEP": rng.random(n_psm).astype(np.float32),
            "purity": rng.random(n_psm).astype(np.float32),
        },
        index=[f"psm{i}" for i in range(n_psm)],
    )
    x = rng.lognormal(12, 2, (n_channels, n_psm)).astype(np.float32)
    x[rng.random((n_channels, n_psm)) < 0.35] = np.nan
    adata = AnnData(
        X=x,
        obs=pd.DataFrame(index=[str(126 + i) for i in range(n_channels)]),
        var=var,
        uns={"level": "psm", "label": "tmt"},
    )
    return MuData({"psm": adata}), fmap


def _dense_split_reference(mdata: MuData, fmap: dict) -> MuData:
    """The dense block-diagonal split built BY HAND (never calls ``split_tmt``).

    Reproduces exactly the dense block-diagonal split -- obs ordering, the block placement of each
    PSM's channel values into its own set, and NaN everywhere else -- so split_tmt's sparse output
    can be checked against an oracle that survives the deletion of the dense builder.
    """
    psm = mdata.mod["psm"]
    source = np.asarray(psm.X, dtype=float)  # (n_channels, n_psm); read as float64, cast back below
    n_channels, n_psm = source.shape
    channels = list(psm.obs_names)
    set_labels = psm.var["filename"].str.rsplit(".", n=1).str[0].map(fmap)
    set_names = list(pd.unique(set_labels))  # first-occurrence order, matching split_tmt
    obs_names = [f"{channel}_{set_name}" for set_name in set_names for channel in channels]

    set_code = set_labels.map({name: i for i, name in enumerate(set_names)}).to_numpy()
    # Match split_tmt's stored dtype (the source .X dtype, e.g. float32); a float64 block would
    # normalise/rollup with slightly different rounding than the sparse path and fail on tolerance.
    block = np.full((n_channels * len(set_names), n_psm), np.nan, dtype=psm.X.dtype)
    for psm_index in range(n_psm):
        base = set_code[psm_index] * n_channels
        block[base : base + n_channels, psm_index] = source[:, psm_index]

    var = psm.var.copy()
    var["set"] = set_labels.to_numpy()
    reference_adata = AnnData(X=block, obs=pd.DataFrame(index=pd.Index(obs_names)), var=var)
    reference_adata.uns = dict(psm.uns)
    reference_mdata = MuData({"psm": reference_adata})
    reference_mdata.var = mdata.var.copy()
    reference_mdata.uns = dict(mdata.uns)
    return reference_mdata


def test_split_tmt_sparse_matches_dense_exactly():
    """split_tmt stores a SciPy sparse .X identical (obs/var/values/NaN) to the hand-built dense result."""
    mdata, fmap = _make_tmt_psm_mdata()
    out_sparse = mm.pp.split_tmt(mdata, fmap)
    out_dense = _dense_split_reference(mdata, fmap)

    x_sparse = out_sparse.mod["psm"].X
    assert sp.issparse(x_sparse)
    assert not sp.issparse(out_dense.mod["psm"].X)
    assert list(out_sparse.mod["psm"].obs_names) == list(out_dense.mod["psm"].obs_names)
    assert list(out_sparse.mod["psm"].var_names) == list(out_dense.mod["psm"].var_names)

    # dense_block restores absent cells as NaN; compare values and NaN-pattern exactly.
    reconstructed = dense_block(x_sparse).astype(np.float32)
    dense = np.asarray(out_dense.mod["psm"].X, dtype=np.float32)
    assert np.array_equal(np.isnan(reconstructed), np.isnan(dense))
    assert np.array_equal(np.nan_to_num(reconstructed), np.nan_to_num(dense))
    # And the sparse form actually stores far less than the dense block-diagonal.
    assert x_sparse.nnz < x_sparse.shape[0] * x_sparse.shape[1]


def test_split_tmt_sparse_to_peptide_matches_dense():
    """split_tmt -> to_peptide equals the hand-built dense reference -> to_peptide (median rollup)."""
    mdata, fmap = _make_tmt_psm_mdata()
    kwargs = dict(agg_method="median", purity_threshold=None, top_n=None, calculate_q=False)

    peptide_sparse = mm.pp.to_peptide(mm.pp.split_tmt(mdata, fmap), **kwargs)
    peptide_dense = mm.pp.to_peptide(_dense_split_reference(mdata, fmap), **kwargs)

    a = peptide_sparse.mod["peptide"]
    b = peptide_dense.mod["peptide"]
    assert list(a.obs_names) == list(b.obs_names)
    assert list(a.var_names) == list(b.var_names)
    xa = np.asarray(a.X, dtype=float)
    xb = np.asarray(b.X, dtype=float)
    assert np.array_equal(np.isnan(xa), np.isnan(xb))
    # median rollup is order-independent -> exact match.
    assert np.allclose(np.nan_to_num(xa), np.nan_to_num(xb), rtol=0, atol=0)


def test_split_tmt_on_sparse_input_preserves_nan():
    """split_tmt on a sparse input .X restores absent cells as NaN, never 0 (§C-1)."""
    mdata, fmap = _make_tmt_psm_mdata()
    dense_x = np.asarray(mdata.mod["psm"].X, dtype=np.float32)
    mdata.mod["psm"].X = to_observed_sparse(dense_x, dtype=np.float32)  # as a sparse-only reader would

    out = mm.pp.split_tmt(mdata, fmap)
    reference, _ = _make_tmt_psm_mdata()  # same seed -> same data, still dense
    expected = _dense_split_reference(reference, fmap)

    got = dense_block(out.mod["psm"].X)
    want = np.asarray(expected.mod["psm"].X, dtype=float)
    assert np.array_equal(np.isnan(got), np.isnan(want)), "absent cells must stay NaN, not become 0"
    assert np.allclose(np.nan_to_num(got), np.nan_to_num(want), rtol=1e-4, atol=1e-3)


def test_split_tmt_unmapped_filename_raises():
    """An incomplete set map must raise, not scatter PSMs into a phantom set (§C-2)."""
    mdata = _make_tmt_mdata()  # filenames runA.raw / runB.raw
    with pytest.raises(ValueError, match="no set mapping"):
        mm.pp.split_tmt(mdata, {"runA": "set1"})  # runB unmapped


def test_to_peptide_purity_filter_on_sparse_stays_sparse_and_matches_dense():
    """to_peptide with a purity filter keeps the block-diagonal sparse through the mask, and the
    peptide result matches the dense path (the mask drops feature columns, never densifies)."""
    mdata, fmap = _make_tmt_psm_mdata()
    reference, _ = _make_tmt_psm_mdata()  # same seed -> identical data incl. purity
    kwargs = dict(agg_method="median", purity_threshold=0.7, top_n=None, calculate_q=False)

    peptide_sparse = mm.pp.to_peptide(mm.pp.split_tmt(mdata, fmap), **kwargs)
    peptide_dense = mm.pp.to_peptide(_dense_split_reference(reference, fmap), **kwargs)

    a = peptide_sparse.mod["peptide"]
    b = peptide_dense.mod["peptide"]
    assert list(a.obs_names) == list(b.obs_names)
    assert list(a.var_names) == list(b.var_names)
    xa = _psm_values(peptide_sparse, "peptide")
    xb = _psm_values(peptide_dense, "peptide")
    assert np.array_equal(np.isnan(xa), np.isnan(xb))
    assert np.allclose(np.nan_to_num(xa), np.nan_to_num(xb), rtol=1e-4, atol=1e-3)


def test_summarisation_prep_purity_filter_keeps_sparse():
    """A purity/column filter on a sparse psm keeps the quant a SparseQuant (never densified)."""
    from msmu._preprocessing._summarisation import SummarisationPrep, SparseQuant

    mdata, fmap = _make_tmt_psm_mdata()
    adata = mm.pp.split_tmt(mdata, fmap).mod["psm"]
    prep = SummarisationPrep(adata, col_to_groupby="peptide", has_decoy=False)
    prep.filter_dict = {"purity": ("gt", 0.7)}

    _, quantification_df, _ = prep.prep()
    assert isinstance(quantification_df, SparseQuant), "purity filter must keep the block-diagonal sparse"


def test_split_tmt_is_preprocessing_api():
    assert hasattr(mm.pp, "split_tmt")
    assert "split_tmt" not in msmu_utils.__all__
    assert not hasattr(msmu_utils, "split_tmt")


def test_split_tmt_splits_channels_by_run_set():
    mdata = _make_tmt_mdata()

    out = mm.pp.split_tmt(mdata, {"runA": "set1", "runB": "set2"})

    assert list(out.mod["psm"].obs_names) == ["126_set1", "127_set1", "126_set2", "127_set2"]
    assert list(out.mod["psm"].var_names) == ["psm1", "psm2", "psm3", "psm4"]
    assert out.mod["psm"].uns["label"] == "tmt"
    # split_tmt now yields a sparse block-diagonal .X by default; to_dense_df restores the
    # structurally-absent cross-set cells as NaN (anndata's .to_df would 0-fill them).
    dense = to_dense_df(out.mod["psm"])
    assert dense.loc["126_set1", "psm1"] == 1.0
    assert dense.loc["126_set2", "psm2"] == 2.0
    assert np.isnan(dense.loc["126_set1", "psm2"])


def test_split_tmt_derives_map_from_sdrf():
    mdata = _make_tmt_mdata()  # var filenames runA.raw / runB.raw
    sdrf = pd.DataFrame(
        {
            "comment[data file]": ["runA.mzML", "runB.mzML"],
            "comment[sample preparation batch]": ["setX", "setY"],
        }
    )
    mdata = mm.pp.attach_sdrf(mdata, sdrf, validate=False)

    out = mm.pp.split_tmt(mdata)  # map=None -> derive from SDRF

    assert list(out.mod["psm"].var["set"]) == ["setX", "setY", "setX", "setY"]
    assert list(out.mod["psm"].obs_names) == ["126_setX", "127_setX", "126_setY", "127_setY"]


def test_split_tmt_map_none_without_sdrf_raises():
    with pytest.raises(ValueError, match="attach_sdrf"):
        mm.pp.split_tmt(_make_tmt_mdata())  # no SDRF attached


def test_split_tmt_map_none_conflicting_batch_raises():
    mdata = _make_tmt_mdata()
    sdrf = pd.DataFrame(
        {
            "comment[data file]": ["runA.mzML", "runA.mzML"],  # same file -> two batches
            "comment[sample preparation batch]": ["setX", "setY"],
        }
    )
    mdata = mm.pp.attach_sdrf(mdata, sdrf, validate=False)
    with pytest.raises(ValueError, match="multiple"):
        mm.pp.split_tmt(mdata)


def test_split_tmt_map_from_sdrf_custom_set_key():
    # SDRF encodes the set in a non-default column (no sample preparation batch present)
    mdata = _make_tmt_mdata()
    sdrf = pd.DataFrame(
        {"comment[data file]": ["runA.mzML", "runB.mzML"], "factor value[batch]": ["b1", "b2"]}
    )
    mdata = mm.pp.attach_sdrf(mdata, sdrf, validate=False)
    out = mm.pp.split_tmt(mdata, set_key="factor value[batch]")
    assert list(out.mod["psm"].var["set"]) == ["b1", "b2", "b1", "b2"]


def test_apply_sdrf_after_split_uses_auto_composite_key():
    # split_tmt records set_key in uns; apply(on=None) then auto-builds the (label, set) composite
    mdata = _make_tmt_mdata()  # obs 126/127, var filenames runA/runB
    sdrf = pd.DataFrame(
        {
            "comment[data file]": ["runA.mzML", "runA.mzML", "runB.mzML", "runB.mzML"],
            "comment[label]": ["126", "127", "126", "127"],
            "comment[sample preparation batch]": ["setX", "setX", "setY", "setY"],
            "source name": ["s1", "s2", "s3", "s4"],
        }
    )
    mdata = mm.pp.attach_sdrf(mdata, sdrf, validate=False)
    split = mm.pp.split_tmt(mdata)  # map=None -> records tmt_split_set_key
    assert split.uns["tmt_split_set_key"] == "comment[sample preparation batch]"

    out = mm.pp.apply_sdrf_to_obs(split)  # on=None -> auto composite, no manual on needed

    obs = out.mod["psm"].obs
    assert list(obs.index) == ["126_setX", "127_setX", "126_setY", "127_setY"]
    assert list(obs["source name"]) == ["s1", "s2", "s3", "s4"]


def _psm_values(mdata, modality):
    x = mdata.mod[modality].X
    return dense_block(x).astype(np.float32) if sp.issparse(x) else np.asarray(x, dtype=np.float32)


def test_log2_transform_keeps_sparse_and_matches_dense():
    """log2 on a sparse psm stays sparse (elementwise on stored values) and matches the dense result."""
    mdata, fmap = _make_tmt_psm_mdata()
    s = mm.pp.log2_transform(mm.pp.split_tmt(mdata, fmap), "psm")
    d = mm.pp.log2_transform(_dense_split_reference(mdata, fmap), "psm")

    assert sp.issparse(s.mod["psm"].X)  # sparse-preserving
    xa, xb = _psm_values(s, "psm"), _psm_values(d, "psm")
    assert np.array_equal(np.isnan(xa), np.isnan(xb))
    assert np.allclose(np.nan_to_num(xa), np.nan_to_num(xb), rtol=1e-4, atol=1e-3)


def test_normalise_median_on_sparse_matches_dense():
    """normalise(median) on a sparse psm produces the same result as on the dense psm."""
    mdata, fmap = _make_tmt_psm_mdata()
    s = mm.pp.normalise(mm.pp.split_tmt(mdata, fmap), method="median", modality="psm")
    d = mm.pp.normalise(_dense_split_reference(mdata, fmap), method="median", modality="psm")
    xa, xb = _psm_values(s, "psm"), _psm_values(d, "psm")
    assert np.array_equal(np.isnan(xa), np.isnan(xb))
    assert np.allclose(np.nan_to_num(xa), np.nan_to_num(xb), rtol=1e-4, atol=1e-3)


def test_plot_get_data_on_sparse_returns_nan_not_zero():
    """PlotData._get_data restores absent cells as NaN (not 0) for a sparse psm."""
    from msmu._plotting._pdata import PlotData

    mdata, fmap = _make_tmt_psm_mdata()
    s = mm.pp.split_tmt(mdata, fmap)
    d = _dense_split_reference(mdata, fmap)
    a = PlotData(s, "psm")._get_data().to_numpy(dtype=float)
    b = PlotData(d, "psm")._get_data().to_numpy(dtype=float)
    assert np.array_equal(np.isnan(a), np.isnan(b))  # absent -> NaN, never 0
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b), rtol=1e-4, atol=1e-3)
