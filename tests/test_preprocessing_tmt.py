import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from mudata import MuData

import msmu as mm
import msmu._utils as msmu_utils
from msmu._core._blockdiag import dense_block


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


def test_split_tmt_sparse_matches_dense_exactly():
    """sparse=True stores a SciPy sparse .X identical (obs/var/values/NaN) to the dense result."""
    mdata, fmap = _make_tmt_psm_mdata()
    out_sparse = mm.pp.split_tmt(mdata, fmap, sparse=True)
    out_dense = mm.pp.split_tmt(mdata, fmap, sparse=False)

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
    """split_tmt(sparse) -> to_peptide equals split_tmt(dense) -> to_peptide (median rollup)."""
    mdata, fmap = _make_tmt_psm_mdata()
    kwargs = dict(agg_method="median", purity_threshold=None, top_n=None, calculate_q=False)

    peptide_sparse = mm.pp.to_peptide(mm.pp.split_tmt(mdata, fmap, sparse=True), **kwargs)
    peptide_dense = mm.pp.to_peptide(mm.pp.split_tmt(mdata, fmap, sparse=False), **kwargs)

    a = peptide_sparse.mod["peptide"]
    b = peptide_dense.mod["peptide"]
    assert list(a.obs_names) == list(b.obs_names)
    assert list(a.var_names) == list(b.var_names)
    xa = np.asarray(a.X, dtype=float)
    xb = np.asarray(b.X, dtype=float)
    assert np.array_equal(np.isnan(xa), np.isnan(xb))
    # median rollup is order-independent -> exact match.
    assert np.allclose(np.nan_to_num(xa), np.nan_to_num(xb), rtol=0, atol=0)


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
    assert out.mod["psm"].to_df().loc["126_set1", "psm1"] == 1.0
    assert out.mod["psm"].to_df().loc["126_set2", "psm2"] == 2.0
    assert np.isnan(out.mod["psm"].to_df().loc["126_set1", "psm2"])


def _psm_values(mdata, modality):
    x = mdata.mod[modality].X
    return dense_block(x).astype(np.float32) if sp.issparse(x) else np.asarray(x, dtype=np.float32)


def test_log2_transform_keeps_sparse_and_matches_dense():
    """log2 on a sparse psm stays sparse (elementwise on stored values) and matches the dense result."""
    mdata, fmap = _make_tmt_psm_mdata()
    s = mm.pp.log2_transform(mm.pp.split_tmt(mdata, fmap, sparse=True), "psm")
    d = mm.pp.log2_transform(mm.pp.split_tmt(mdata, fmap, sparse=False), "psm")

    assert sp.issparse(s.mod["psm"].X)  # sparse-preserving
    xa, xb = _psm_values(s, "psm"), _psm_values(d, "psm")
    assert np.array_equal(np.isnan(xa), np.isnan(xb))
    assert np.allclose(np.nan_to_num(xa), np.nan_to_num(xb), rtol=1e-4, atol=1e-3)


def test_normalise_median_on_sparse_matches_dense():
    """normalise(median) on a sparse psm produces the same result as on the dense psm."""
    mdata, fmap = _make_tmt_psm_mdata()
    s = mm.pp.normalise(mm.pp.split_tmt(mdata, fmap, sparse=True), method="median", modality="psm")
    d = mm.pp.normalise(mm.pp.split_tmt(mdata, fmap, sparse=False), method="median", modality="psm")
    xa, xb = _psm_values(s, "psm"), _psm_values(d, "psm")
    assert np.array_equal(np.isnan(xa), np.isnan(xb))
    assert np.allclose(np.nan_to_num(xa), np.nan_to_num(xb), rtol=1e-4, atol=1e-3)


def test_plot_get_data_on_sparse_returns_nan_not_zero():
    """PlotData._get_data restores absent cells as NaN (not 0) for a sparse psm."""
    from msmu._plotting._pdata import PlotData

    mdata, fmap = _make_tmt_psm_mdata()
    s = mm.pp.split_tmt(mdata, fmap, sparse=True)
    d = mm.pp.split_tmt(mdata, fmap, sparse=False)
    a = PlotData(s, "psm")._get_data().to_numpy(dtype=float)
    b = PlotData(d, "psm")._get_data().to_numpy(dtype=float)
    assert np.array_equal(np.isnan(a), np.isnan(b))  # absent -> NaN, never 0
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b), rtol=1e-4, atol=1e-3)
