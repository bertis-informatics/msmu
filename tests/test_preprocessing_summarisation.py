import numpy as np
import pandas as pd
import pytest

from msmu._preprocessing._summarisation import (
    Aggregator,
    FeatureRanker,
    Scorer,
    SummarisationPrep,
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

    # Per-sample estimate should equal overall + col_effect (row median is 0 here).
    expected = overall + col_effects
    np.testing.assert_allclose(estimates, expected, atol=1e-8)


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

    # The outlier should not drag the sample-0 estimate away from the consensus.
    np.testing.assert_allclose(estimates, [10.0, 11.0, 12.0], atol=1e-8)


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


def test_summarisation_prep_filters_and_ranks(simple_adata):
    prep = SummarisationPrep(simple_adata, col_to_groupby="peptide", has_decoy=False)
    prep.filter_dict = {"score": ("lt", 0.5)}
    prep.rank_tuple = ("total_intensity", 1)
    _, quant, _ = prep.prep()

    assert np.isnan(quant.loc["f2", "s1"])
