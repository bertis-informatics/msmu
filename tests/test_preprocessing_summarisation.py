import numpy as np
import pandas as pd
import pytest

from msmu._preprocessing._summarisation import Aggregator, FeatureRanker, Scorer, SummarisationPrep


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


def test_summarisation_prep_filters_and_ranks(simple_adata):
    prep = SummarisationPrep(simple_adata, col_to_groupby="peptide", has_decoy=False)
    prep.filter_dict = {"score": ("lt", 0.5)}
    prep.rank_tuple = ("total_intensity", 1)
    _, quant, _ = prep.prep()

    assert np.isnan(quant.loc["f2", "s1"])
