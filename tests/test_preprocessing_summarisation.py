import numpy as np
import pandas as pd
import pytest

from msmu._preprocessing._summarisation import (
    Aggregator,
    FeatureRanker,
    Scorer,
    SummarisationPrep,
    _directlfq_rollup,
    _median_polish,
)


def test_feature_ranker_total_intensity():
    id_df = pd.DataFrame({"peptide": ["p1", "p2", "p3"]}, index=["f1", "f2", "f3"])
    quant_df = pd.DataFrame({"s1": [1.0, 5.0, 2.0], "s2": [1.0, 5.0, 0.5]}, index=id_df.index)
    ranked = FeatureRanker.total_intensity(id_df.copy(), quant_df, col_to_groupby="peptide")
    assert "rank_score" in ranked.columns
    assert ranked.loc["f2", "rank"] == 1.0


def test_scorer_best_pep_and_score():
    scorer = Scorer.best_pep([0.2, 0.05, 0.1])
    assert scorer.picked_pep == 0.05
    assert scorer.picked_score > 0


def test_scorer_func_invalid():
    with pytest.raises(ValueError, match="not recognized"):
        Scorer.func("nope")


def test_aggregator_peptide_quantification():
    id_df = pd.DataFrame(
        {
            "peptide": ["p1", "p1", "p2"],
            "proteins": ["A", "A", "B"],
            "stripped_peptide": ["p1", "p1", "p2"],
            "PEP": [0.1, 0.2, 0.3],
        },
        index=["f1", "f2", "f3"],
    )
    quant_df = pd.DataFrame({"s1": [1.0, 2.0, 3.0], "s2": [1.0, 2.0, 3.0]}, index=id_df.index)
    agg = Aggregator.peptide(
        identification_df=id_df,
        quantification_df=quant_df,
        decoy_df=None,
        agg_method="median",
        score_method="best_pep",
        protein_col="proteins",
        peptide_col="peptide",
    )
    ident = agg.aggregate_identification()
    quant = agg.aggregate_quantification()
    assert ident.loc["p1", "count_psm"] == 2
    assert quant.shape[0] == 2


def test_median_polish_additive_matrix_recovers_column_effects():
    # An exactly additive matrix: value = overall + row_effect + col_effect.
    overall = 5.0
    row_effects = np.array([0.0, 1.0, -2.0])
    col_effects = np.array([0.0, 3.0, -1.0, 2.0])
    matrix = overall + row_effects[:, None] + col_effects[None, :]

    estimates = _median_polish(matrix)

    # The profile is the column effects up to one constant: the log2 fitted total over the features,
    # which on an exactly additive matrix is the observed total.
    expected = overall + col_effects + np.log2(np.sum(2.0**row_effects))
    np.testing.assert_allclose(estimates, expected, atol=1e-8)
    np.testing.assert_allclose(np.sum(2.0**estimates), np.sum(2.0**matrix), rtol=1e-10)


def test_median_polish_is_robust_to_a_single_outlier_peptide():
    base = np.array(
        [
            [10.0, 11.0, 12.0],
            [10.0, 11.0, 12.0],
            [10.0, 11.0, 12.0],
        ]
    )
    with_outlier = base.copy()
    with_outlier[0, 0] = 100.0  # one wildly high peptide value

    estimates = _median_polish(with_outlier)

    # The outlier must not drag the sample-0 estimate away from the consensus profile. It does enter
    # the level, which by convention is the summed intensity of every observed cell (BID-334), so
    # the profile is compared up to one constant.
    np.testing.assert_allclose(estimates - estimates[0], [0.0, 1.0, 2.0], atol=1e-8)


def test_median_polish_handles_missing_values():
    matrix = np.array(
        [
            [10.0, np.nan, 12.0],
            [10.0, 11.0, np.nan],
            [np.nan, np.nan, np.nan],  # sample 1 has no other observation -> stays NaN here
        ]
    )
    estimates = _median_polish(matrix)

    assert not np.isnan(estimates[0])
    assert not np.isnan(estimates[1])  # sample 1 still has one observed value (row 1)
    assert not np.isnan(estimates[2])


def test_median_polish_fully_missing_sample_is_nan():
    matrix = np.array(
        [
            [10.0, np.nan, 12.0],
            [11.0, np.nan, 13.0],
        ]
    )
    estimates = _median_polish(matrix)

    assert np.isnan(estimates[1])  # sample 1 has no observations at all
    assert not np.isnan(estimates[0])
    assert not np.isnan(estimates[2])


def test_median_polish_ignores_fully_missing_features():
    # Regression: all-NaN peptide rows (e.g. shared peptides masked out during the protein
    # rollup) must not drag the protein level toward zero. The estimate has to match the same
    # matrix with those empty rows removed and stay on the input's log2 scale, not collapse to ~0.
    signal = np.array([16.64, 16.73, 16.55, 16.55, 15.99, 16.53])
    empty = np.full_like(signal, np.nan)
    with_empty_rows = np.vstack([signal, empty, empty, empty])

    estimates = _median_polish(with_empty_rows)

    np.testing.assert_allclose(estimates, _median_polish(signal[None, :]), atol=1e-8)
    np.testing.assert_allclose(estimates, signal, atol=1e-8)
    assert np.nanmedian(estimates) > 5.0  # not collapsed toward zero


def test_median_polish_level_survives_majority_missing_features():
    # A few observed peptides among many all-NaN rows (a large protein group quantified by only
    # a handful of unique peptides). The recovered level must track the ~21 observations, not 0.
    observed = np.array(
        [
            [20.8, 20.9, 20.8, 20.8, 19.0, 20.8],
            [21.1, 21.3, 21.1, 21.0, 17.8, 21.1],
            [21.7, 21.7, 21.5, 21.7, 20.0, 21.4],
        ]
    )
    padded = np.full((50, observed.shape[1]), np.nan)
    padded[[10, 25, 40], :] = observed

    estimates = _median_polish(padded)

    np.testing.assert_allclose(estimates, _median_polish(observed), atol=1e-8)
    assert np.nanmin(estimates) > 15.0  # on the ~21 log2 scale, not collapsed toward zero


def test_median_polish_all_missing_matrix_is_all_nan():
    estimates = _median_polish(np.full((3, 4), np.nan))
    assert estimates.shape == (4,)
    assert np.all(np.isnan(estimates))


def test_median_polish_matches_r_medpolish_when_every_feature_is_shared():
    # R: medpolish(m, eps=1e-4, maxiter=10, na.rm=TRUE); overall + col. Every row is observed in at
    # least two samples and every sample is linked, so the comparability rules change nothing. The
    # between-sample differences are medpolish's; the level is anchored to the fitted total intensity
    # instead of medpolish's overall (BID-334), so the two are compared up to one constant.
    complete = np.array([[20, 21, 22], [23, 25, 24], [19, 20, 22], [24, 22, 21]], dtype=float)
    r_complete = np.array([20.5, 21.5, 22.0])
    estimates = _median_polish(complete)
    np.testing.assert_allclose(estimates - estimates[0], r_complete - r_complete[0], atol=1e-10)

    with_gaps = np.array(
        [[20, 21, np.nan], [23, np.nan, 24], [np.nan, 20, 22], [24, 22, 21]],
        dtype=float,
    )
    r_with_gaps = np.array([20.875, 21.375, 21.875])
    estimates = _median_polish(with_gaps)
    np.testing.assert_allclose(estimates - estimates[0], r_with_gaps - r_with_gaps[0], atol=1e-10)


def test_median_polish_features_seen_in_one_sample_do_not_vote_on_the_difference():
    # Regression (BID-330): a peptide observed in one sample has a zero residual by construction
    # and used to vote "no difference" in that sample's column median. With such peptides in the
    # majority, both samples collapsed onto the overall level and the protein read as unchanged.
    # The difference must come from the shared peptides alone; the single-sample peptides enter the
    # level only. Every peptide here sits exactly on its sample's level (row effect 0), so the fitted
    # total is nine peptides at the level: [22, 18] + log2(9).
    shared = [[22.0, 18.0]] * 3
    wt_only = [[22.0, np.nan]] * 5
    ko_only = [[np.nan, 18.0]]
    estimates = _median_polish(np.array(shared + wt_only + ko_only))

    np.testing.assert_allclose(estimates, np.array([22.0, 18.0]) + np.log2(9), atol=1e-8)


def test_median_polish_difference_is_the_median_shared_peptide_ratio():
    matrix = np.array(
        [
            [22.0, 20.0],  # the only shared peptide: difference 2
            [23.0, np.nan],
            [24.0, np.nan],
            [np.nan, 19.0],
            [np.nan, 21.0],
        ]
    )
    estimates = _median_polish(matrix)

    np.testing.assert_allclose(estimates[0] - estimates[1], 2.0, atol=1e-8)


def test_median_polish_two_samples_without_a_shared_feature_are_nan():
    matrix = np.array([[23.0, np.nan], [24.0, np.nan], [np.nan, 19.0], [np.nan, 21.0]])
    assert np.all(np.isnan(_median_polish(matrix)))


def test_median_polish_single_sample_protein_keeps_its_level():
    # Nothing to compare against, so the level is the summed intensity of every observed peptide.
    matrix = np.array([[22.0, np.nan, np.nan], [21.0, np.nan, np.nan]])
    estimates = _median_polish(matrix)

    np.testing.assert_allclose(estimates[0], np.log2(2.0**22 + 2.0**21), atol=1e-8)
    assert np.all(np.isnan(estimates[1:]))


def test_median_polish_isolated_sample_is_nan_and_the_linked_samples_keep_their_estimates():
    # Sample 3 is observed through a feature nobody else sees; it is dropped on its own, and the
    # three linked samples are summarised exactly as if it had never been there.
    linked = np.array([[20.0, 21.0, 22.0], [21.0, 22.0, 23.0]])
    with_isolated = np.array(
        [
            [20.0, 21.0, 22.0, np.nan],
            [21.0, 22.0, 23.0, np.nan],
            [np.nan, np.nan, np.nan, 25.0],
        ]
    )
    estimates = _median_polish(with_isolated)

    np.testing.assert_allclose(estimates[:3], _median_polish(linked), atol=1e-8)
    assert np.isnan(estimates[3])


def test_median_polish_keeps_the_largest_linked_group():
    # Samples 0-2 are linked by two features and samples 3-4 by another; nothing links the two
    # groups, so only the larger one is placed on a scale, exactly as if polished on its own.
    larger = np.array([[20.0, 21.0, 22.0], [21.0, 22.0, 23.0]])
    split = np.array(
        [
            [20.0, 21.0, 22.0, np.nan, np.nan],
            [21.0, 22.0, 23.0, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 24.0, 25.0],
        ]
    )
    estimates = _median_polish(split)

    np.testing.assert_allclose(estimates[:3], _median_polish(larger), atol=1e-8)
    assert np.all(np.isnan(estimates[3:]))
    # Sample order must not matter: with the smaller group first the answer is the same, reversed.
    np.testing.assert_allclose(_median_polish(split[:, ::-1]), estimates[::-1], atol=1e-8)


def test_median_polish_differences_equal_a_polish_of_the_shared_features():
    # Features seen in one sample set the level only. The between-sample differences must be those
    # of a polish of the shared features alone -- including where the iteration stops, which the
    # single-sample rows must not influence (their residual is just the last column median).
    with_singletons = np.array(
        [
            [20.6, 20.6, np.nan],
            [19.6, np.nan, np.nan],
            [np.nan, 22.3, 20.2],
            [17.9, np.nan, np.nan],
        ]
    )
    shared_only = with_singletons[[0, 2]]
    estimates = _median_polish(with_singletons)
    reference = _median_polish(shared_only)

    np.testing.assert_allclose(estimates - estimates[0], reference - reference[0], atol=1e-10)


def test_median_polish_groups_tied_for_largest_are_all_nan():
    # Samples 0-1 and 2-3 share no feature and are the same size: nothing singles out either
    # group, and returning one would fabricate the between-group comparison the matrix cannot
    # support.
    matrix = np.array(
        [
            [20.0, 21.0, np.nan, np.nan],
            [22.0, 23.0, np.nan, np.nan],
            [np.nan, np.nan, 24.0, 25.0],
        ]
    )
    assert np.all(np.isnan(_median_polish(matrix)))


def test_median_polish_level_is_the_fitted_total_and_does_not_depend_on_missing_cells():
    # Regression (BID-334): the level used to be Tukey's ``overall`` -- the median peptide -- which
    # does not grow with the number of peptides. A first fix summed the *observed* cells, which falls
    # by about log2 of the observed fraction and so penalised sparse proteins. The level is the sum of
    # the *fitted* peptide intensities: on an exactly additive block it equals the observed total, and
    # blanking cells does not move it.
    # Column effects symmetric around 0 so that the sweeps converge exactly with a gap per row (the
    # median of any three of them is still 0); a sum over the observed 12 of 16 cells would sit
    # log2(16 / 12) = 0.415 lower.
    overall, row_effects, col_effects = 20.0, np.array([0.0, 1.5, -1.0, 2.0]), np.array([-1.0, 0.0, 0.0, 1.0])
    complete = overall + row_effects[:, None] + col_effects[None, :]
    with_gaps = complete.copy()
    with_gaps[[0, 1, 2, 3], [1, 3, 0, 2]] = np.nan

    complete_estimates = _median_polish(complete)
    np.testing.assert_allclose(np.sum(2.0**complete_estimates), np.sum(2.0**complete), rtol=1e-10)
    np.testing.assert_allclose(_median_polish(with_gaps), complete_estimates, atol=1e-8)


def test_median_polish_level_grows_with_the_number_of_peptides():
    one_peptide = np.array([[20.0, 21.0, 22.0]])
    four_peptides = np.repeat(one_peptide, 4, axis=0)

    np.testing.assert_allclose(_median_polish(four_peptides) - _median_polish(one_peptide), 2.0, atol=1e-8)


def test_median_polish_single_sample_peptides_move_the_level_but_not_the_differences():
    shared = np.array([[22.0, 20.0, 21.0], [23.0, 21.0, 22.5]])
    with_single_sample_peptide = np.vstack([shared, [[30.0, np.nan, np.nan]]])
    base = _median_polish(shared)
    extended = _median_polish(with_single_sample_peptide)

    np.testing.assert_allclose(extended - extended[0], base - base[0], atol=1e-8)
    assert np.all(extended > base)


def test_median_polish_and_directlfq_share_the_total_intensity_level():
    # Both follow the total-intensity convention; on an exactly additive block (no residuals) the
    # fitted total of median polish and the observed total directlfq rescales to are the same number.
    pytest.importorskip("directlfq")
    block = 20.0 + np.array([0.0, 1.5, -1.0])[:, None] + np.array([0.0, 0.5, 1.0])[None, :]

    np.testing.assert_allclose(_median_polish(block), _directlfq_rollup(block), atol=1e-6)


def test_aggregator_quantification_median_polish():
    id_df = pd.DataFrame(
        {
            "peptide": ["p1", "p2", "p3"],
            "protein_group": ["A", "A", "B"],
            "stripped_peptide": ["p1", "p2", "p3"],
            "PEP": [0.1, 0.2, 0.3],
        },
        index=["f1", "f2", "f3"],
    )
    quant_df = pd.DataFrame(
        {"s1": [10.0, 12.0, 5.0], "s2": [11.0, 13.0, 6.0]},
        index=id_df.index,
    )
    agg = Aggregator.protein(
        identification_df=id_df,
        quantification_df=quant_df,
        decoy_df=None,
        agg_method="median_polish",
        score_method="best_pep",
        protein_col="protein_group",
    )
    quant = agg.aggregate_quantification()

    # Two protein groups, both samples present.
    assert sorted(quant.index) == ["A", "B"]
    assert list(quant.columns) == ["s1", "s2"]
    # Protein B has a single peptide, so its estimate is just that peptide's value.
    np.testing.assert_allclose(quant.loc["B", ["s1", "s2"]].to_numpy(dtype=float), [5.0, 6.0], atol=1e-8)


def test_directlfq_rollup_additive_matrix_recovers_column_effects():
    # DirectLFQ aligns feature traces then medians; on an exactly additive log2 matrix the
    # per-sample profile should recover the column effects up to a constant offset.
    row_effects = np.array([0.0, 1.0, 2.0, 3.0])
    col_effects = np.array([10.0, 11.0, 12.0])
    matrix = row_effects[:, None] + col_effects[None, :]

    estimates = _directlfq_rollup(matrix)

    # The absolute level is anchored by directlfq; only the between-sample shape is defined.
    centered = estimates - estimates[0]
    np.testing.assert_allclose(centered, col_effects - col_effects[0], atol=1e-8)


def test_directlfq_rollup_is_robust_to_a_single_outlier_peptide():
    base = np.array([0.0, 1.0, 2.0, 3.0])[:, None] + np.array([10.0, 11.0, 12.0])[None, :]
    with_outlier = base.copy()
    with_outlier[0, :] = with_outlier[0, :] + 8.0  # one wildly shifted peptide

    estimates = _directlfq_rollup(with_outlier)

    centered = estimates - estimates[0]
    np.testing.assert_allclose(centered, [0.0, 1.0, 2.0], atol=1e-8)


def test_directlfq_rollup_single_feature_returns_that_feature():
    estimates = _directlfq_rollup(np.array([[10.0, 11.0, 12.0]]))
    np.testing.assert_allclose(estimates, [10.0, 11.0, 12.0], atol=1e-8)


def test_directlfq_rollup_fully_missing_sample_is_nan():
    matrix = np.array(
        [
            [10.0, np.nan, 12.0],
            [11.0, np.nan, 13.0],
        ]
    )
    estimates = _directlfq_rollup(matrix)

    assert np.isnan(estimates[1])  # sample 1 has no observations at all
    assert not np.isnan(estimates[0])
    assert not np.isnan(estimates[2])


def test_directlfq_rollup_all_missing_matrix_is_all_nan():
    estimates = _directlfq_rollup(np.full((3, 3), np.nan))
    assert np.all(np.isnan(estimates))


def test_aggregator_quantification_directlfq():
    id_df = pd.DataFrame(
        {
            "peptide": ["p1", "p2", "p3"],
            "protein_group": ["A", "A", "B"],
            "stripped_peptide": ["p1", "p2", "p3"],
            "PEP": [0.1, 0.2, 0.3],
        },
        index=["f1", "f2", "f3"],
    )
    quant_df = pd.DataFrame(
        {"s1": [10.0, 12.0, 5.0], "s2": [11.0, 13.0, 6.0]},
        index=id_df.index,
    )
    agg = Aggregator.protein(
        identification_df=id_df,
        quantification_df=quant_df,
        decoy_df=None,
        agg_method="directlfq",
        score_method="best_pep",
        protein_col="protein_group",
    )
    quant = agg.aggregate_quantification()

    # Two protein groups, both samples present.
    assert sorted(quant.index) == ["A", "B"]
    assert list(quant.columns) == ["s1", "s2"]
    # Protein B has a single peptide, so its estimate is just that peptide's value.
    np.testing.assert_allclose(quant.loc["B", ["s1", "s2"]].to_numpy(dtype=float), [5.0, 6.0], atol=1e-8)


@pytest.mark.parametrize("rejected_method", ["median_polish", "directlfq"])
def test_to_peptide_rejects_matrix_rollups(mdata, rejected_method):
    from msmu._preprocessing._summarise import to_peptide

    with pytest.raises(ValueError, match="to_protein"):
        to_peptide(mdata, agg_method=rejected_method)


def test_summarisation_prep_filters_and_ranks(simple_adata):
    prep = SummarisationPrep(simple_adata, col_to_groupby="peptide", has_decoy=False)
    prep.filter_dict = {"score": ("lt", 0.5)}
    prep.rank_tuple = ("total_intensity", 1)
    _, quant, _ = prep.prep()

    assert np.isnan(quant.loc["f2", "s1"])


def test_to_protein_uses_every_peptide_by_default():
    # A top-N cut ranks on a NaN-skipping median, which favours peptides seen in few samples --
    # exactly the ones that cannot compare samples under median polish -- so no cut is the default.
    import inspect

    import msmu as mm

    assert inspect.signature(mm.pp.to_protein).parameters["top_n"].default is None


def test_peptide_and_protein_summarisation_share_tied_pep_policy():
    import msmu as mm
    from anndata import AnnData
    from mudata import MuData

    peptides = ["AA", "BB", "CC", "DD", "EE", "FF"]
    ident = pd.DataFrame({
        "peptide": peptides,
        "stripped_peptide": peptides,
        "proteins": [f"P{i}" for i in range(6)],
        "PEP": [0.1] * 6,
    }, index=[f"psm_{i}" for i in range(6)])
    psm = AnnData(
        X=np.ones((1, 5)), obs=pd.DataFrame(index=["sample"]), var=ident.iloc[:5].copy(),
        uns={"decoy": ident.iloc[5:].copy()},
    )
    mdata = mm.pp.to_peptide(MuData({"psm": psm}))
    np.testing.assert_array_equal(mdata["peptide"].var["q_value"], [0.4] * 5)
    np.testing.assert_array_equal(mdata["peptide"].uns["decoy"]["q_value"], [0.4])

    mdata = mm.pp.to_protein(mm.pp.infer_protein(mdata))
    np.testing.assert_array_equal(mdata["protein"].var["q_value"], [0.4] * 5)
    np.testing.assert_array_equal(mdata["protein"].uns["decoy"]["q_value"], [0.4])
