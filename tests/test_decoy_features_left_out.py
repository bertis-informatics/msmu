"""Analyses of the quantification leave out the decoy features.

Decoys are to move into ``.var`` next to the targets, marked by ``decoy == 1``. Every function that
analyses the quantification must then give the result it gives without the decoy rows, and the
functions that rewrite the quantification must leave the decoy values as they were.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData

import msmu as mm
from msmu._core._blockdiag import dense_block, to_observed_sparse
from msmu._preprocessing._normalisation import ADJUSTMENT_STATUS_NOT_QUANTIFIED, PTMProteinAdjuster

SAMPLES = ["s1", "s2", "s3", "s4", "s5", "s6"]
GROUPS = ["A", "A", "A", "B", "B", "B"]
BATCHES = ["x", "y", "x", "y", "x", "y"]
FEATURE_COUNT = 16
# Interleaved with the targets, so putting the decoy columns back in place is exercised.
DECOY_POSITIONS = [2, 5, 9, 13]
IS_DECOY = np.isin(np.arange(FEATURE_COUNT), DECOY_POSITIONS)
FEATURE_NAMES = [f"decoy_P{index}" if is_decoy else f"P{index}" for index, is_decoy in enumerate(IS_DECOY)]
TARGET_NAMES = [name for name, is_decoy in zip(FEATURE_NAMES, IS_DECOY) if not is_decoy]
DECOY_NAMES = [name for name, is_decoy in zip(FEATURE_NAMES, IS_DECOY) if is_decoy]


def _make_values(*, has_missing_cells: bool) -> np.ndarray:
    random_generator = np.random.default_rng(0)
    values = random_generator.normal(20.0, 2.0, size=(len(SAMPLES), FEATURE_COUNT))
    values[:, IS_DECOY] -= 4.0  # decoys are quantified lower than targets
    if has_missing_cells:
        values[0, 0] = np.nan
        values[3, DECOY_POSITIONS[0]] = np.nan
    return values


def _make_mdata(*, is_sparse: bool = False, include_decoys: bool = True, has_missing_cells: bool = True) -> MuData:
    kept_features = np.ones(FEATURE_COUNT, dtype=bool) if include_decoys else ~IS_DECOY
    values = _make_values(has_missing_cells=has_missing_cells)[:, kept_features]
    var = pd.DataFrame(
        {"decoy": IS_DECOY.astype(int), "class": ["a", "b"] * (FEATURE_COUNT // 2)},
        index=FEATURE_NAMES,
    ).loc[kept_features]
    obs = pd.DataFrame({"group": GROUPS, "batch": BATCHES}, index=SAMPLES)
    matrix = to_observed_sparse(values, dtype=np.float64) if is_sparse else values
    mdata = MuData({"protein": AnnData(X=matrix, obs=obs, var=var)})
    mdata.obs["sample"] = SAMPLES
    mdata.obs["group"] = pd.Categorical(GROUPS, categories=["A", "B"])
    mdata.obs["batch"] = pd.Categorical(BATCHES, categories=["x", "y"])
    return mdata


def _dense_values(adata: AnnData) -> np.ndarray:
    return dense_block(adata.X) if hasattr(adata.X, "tocsc") else np.asarray(adata.X)


def test_pca_leaves_out_decoys_and_gives_them_no_loadings():
    result = mm.tl.pca(_make_mdata(), modality="protein")["protein"]
    expected = mm.tl.pca(_make_mdata(include_decoys=False), modality="protein")["protein"]

    pd.testing.assert_frame_equal(result.obsm["X_pca"], expected.obsm["X_pca"])
    pd.testing.assert_frame_equal(result.varm["X_pca"].loc[TARGET_NAMES], expected.varm["X_pca"])
    assert result.varm["X_pca"].loc[DECOY_NAMES].isna().all().all()


def test_umap_leaves_out_decoys():
    result = mm.tl.umap(_make_mdata(), modality="protein", n_neighbors=3, random_state=0)["protein"]
    expected = mm.tl.umap(_make_mdata(include_decoys=False), modality="protein", n_neighbors=3, random_state=0)

    pd.testing.assert_frame_equal(result.obsm["X_umap"], expected["protein"].obsm["X_umap"])


def test_corr_leaves_out_decoys():
    result = mm.tl.corr(_make_mdata(), modality="protein")["protein"]
    expected = mm.tl.corr(_make_mdata(include_decoys=False), modality="protein")["protein"]

    np.testing.assert_array_equal(result.obsp["X_corr"], expected.obsp["X_corr"])


def test_run_de_neither_tests_decoys_nor_counts_them_in_the_correction():
    result = mm.tl.run_de(_make_mdata(), modality="protein", category="group", ctrl="A", expr="B")
    expected = mm.tl.run_de(_make_mdata(include_decoys=False), modality="protein", category="group", ctrl="A", expr="B")

    pd.testing.assert_frame_equal(result.to_df(), expected.to_df())


@pytest.mark.parametrize(
    ("method", "is_sparse", "group_var"),
    [
        ("median", False, None),
        ("median", True, None),
        ("median", True, "class"),
        ("quantile", False, None),
        ("quantile", True, None),
        ("pairwise_median", True, None),
    ],
)
def test_normalise_computes_from_targets_and_leaves_decoys_unchanged(method, is_sparse, group_var):
    mdata = _make_mdata(is_sparse=is_sparse)
    original_values = _dense_values(mdata["protein"])

    result = mm.pp.normalise(mdata, method=method, modality="protein", group_var=group_var)["protein"]
    expected = mm.pp.normalise(
        _make_mdata(is_sparse=is_sparse, include_decoys=False), method=method, modality="protein", group_var=group_var
    )["protein"]

    result_values = _dense_values(result)
    np.testing.assert_allclose(result_values[:, ~IS_DECOY], _dense_values(expected), equal_nan=True)
    np.testing.assert_array_equal(result_values[:, IS_DECOY], original_values[:, IS_DECOY])
    assert hasattr(result.X, "tocsc") is is_sparse
    pd.testing.assert_frame_equal(result.uns["normalisation"]["summary"], expected.uns["normalisation"]["summary"])


@pytest.mark.parametrize(
    ("method", "is_sparse", "has_missing_cells"),
    [
        ("median_center", False, True),
        ("median_center", True, True),
        # ComBat refuses missing values.
        ("combat", False, False),
    ],
)
def test_correct_batch_effect_estimates_from_targets_and_leaves_decoys_unchanged(method, is_sparse, has_missing_cells):
    mdata = _make_mdata(is_sparse=is_sparse, has_missing_cells=has_missing_cells)
    original_values = _dense_values(mdata["protein"])

    result = mm.pp.correct_batch_effect(mdata, modality="protein", method=method, category="batch")["protein"]
    expected = mm.pp.correct_batch_effect(
        _make_mdata(is_sparse=is_sparse, include_decoys=False, has_missing_cells=has_missing_cells),
        modality="protein",
        method=method,
        category="batch",
    )["protein"]

    result_values = _dense_values(result)
    np.testing.assert_allclose(result_values[:, ~IS_DECOY], _dense_values(expected), equal_nan=True)
    np.testing.assert_array_equal(result_values[:, IS_DECOY], original_values[:, IS_DECOY])


def test_to_ptm_builds_no_site_from_a_decoy_peptide():
    """The decoy carries its target's accession, as DIA-NN decoys did, so only its flag tells it apart."""
    phospho = "[+79.97]"
    target_peptide, decoy_peptide = f"SPGS{phospho}PVLR", f"S{phospho}PGSPVLR"
    peptides = pd.DataFrame(
        {
            "peptide": [target_peptide, decoy_peptide],
            "stripped_peptide": ["SPGSPVLR", "SPGSPVLR"],
            "proteins": ["P1", "P1"],
            "count_psm": [5, 5],
            "decoy": [0, 1],
        },
        index=[target_peptide, decoy_peptide],
    )
    mdata = MuData({"peptide": AnnData(X=np.full((2, 2), 10.0), obs=pd.DataFrame(index=["s1", "s2"]), var=peptides)})
    mdata.uns["protein_info"] = pd.DataFrame({"Sequence": ["MAAASPGSPVLRKKQ"]}, index=["P1"])

    result = mm.pp.to_ptm(mdata, modi_name="phospho", modification=phospho)

    assert list(result["phospho_site"].var_names) == ["P1|S8"]


def test_adjust_ptm_by_protein_never_takes_a_decoy_protein_as_denominator():
    """A decoy protein flagged in the global data is skipped even when the site's accession maps to it."""
    ptm_mdata = MuData(
        {
            "phospho_site": AnnData(
                X=np.ones((len(SAMPLES), 1)),
                obs=pd.DataFrame(index=SAMPLES),
                var=pd.DataFrame({"modified_protein": ["P1"]}, index=["P1|S30"]),
            )
        }
    )
    global_mdata = MuData(
        {
            "protein": AnnData(
                X=np.ones((len(SAMPLES), 1)),
                obs=pd.DataFrame(index=SAMPLES),
                var=pd.DataFrame({"decoy": [1]}, index=["P1"]),
            )
        }
    )
    global_mdata.uns["protein_map"] = pd.DataFrame({"initial_protein": ["P1"], "protein_group": ["P1"]})

    adjuster = PTMProteinAdjuster(ptm_mdata, global_mdata, ptm_mod="phospho_site", global_mod="protein")

    assert adjuster.resolution["adjustment_status"].to_dict() == {"P1|S30": ADJUSTMENT_STATUS_NOT_QUANTIFIED}


@pytest.mark.parametrize(
    ("plot_function", "arguments"),
    [
        (mm.pl.plot_id, {"groupby": "group"}),
        (mm.pl.plot_intensity, {"groupby": "group"}),
        (mm.pl.plot_missingness, {}),
        (mm.pl.plot_correlation, {"groupby": "group"}),
        (mm.pl.plot_upset, {"groupby": "group"}),
        (mm.pl.plot_var, {"groupby": "group", "var_column": "class"}),
    ],
)
def test_plots_leave_out_decoys(plot_function, arguments):
    result = plot_function(_make_mdata(), modality="protein", **arguments)
    expected = plot_function(_make_mdata(include_decoys=False), modality="protein", **arguments)

    assert result.to_json() == expected.to_json()
