"""Denominator resolution and the output contract for adjust_ptm_by_protein.

The adjustment finds a site's denominator through the accessions it was localised on, translated
into a global protein group via the global container's own ``protein_map``. These tests pin the
cases that decide whether a site is adjustable at all, and the promise that nothing is dropped.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData

import msmu as mm
from msmu._preprocessing._normalisation import (
    ADJUSTMENT_STATUS_ADJUSTED,
    ADJUSTMENT_STATUS_NO_GLOBAL_GROUP,
    ADJUSTMENT_STATUS_NOT_IN_GLOBAL_FASTA,
    ADJUSTMENT_STATUS_NOT_QUANTIFIED,
    ADJUSTMENT_STATUS_SHARED_GROUPS,
    PTMProteinAdjuster,
)

SAMPLES = ["s1", "s2", "s3", "s4"]


def _make_ptm(site_to_accessions: dict[str, str]) -> MuData:
    """PTM container whose sites carry only the accessions they were localised on."""
    sites = list(site_to_accessions)
    values = np.arange(len(SAMPLES) * len(sites), dtype=float).reshape(len(SAMPLES), len(sites))
    adata = AnnData(
        X=values,
        obs=pd.DataFrame(index=SAMPLES),
        var=pd.DataFrame({"modified_protein": list(site_to_accessions.values())}, index=sites),
    )
    return MuData({"phospho_site": adata})


def _make_global(
    quantified_groups: list[str],
    accession_to_group: dict[str, str],
    fasta_accessions: list[str] | None = None,
) -> MuData:
    """Global container: a protein matrix indexed by group, plus infer_protein's protein_map."""
    values = np.arange(len(SAMPLES) * len(quantified_groups), dtype=float).reshape(len(SAMPLES), len(quantified_groups))
    adata = AnnData(
        X=values,
        obs=pd.DataFrame(index=SAMPLES),
        var=pd.DataFrame(index=quantified_groups),
    )
    mdata = MuData({"protein": adata})
    mdata.uns["protein_map"] = pd.DataFrame(
        {
            "initial_protein": list(accession_to_group),
            "protein_group": list(accession_to_group.values()),
        }
    )
    if fasta_accessions is not None:
        mdata.uns["protein_info"] = pd.DataFrame(index=fasta_accessions)
    return mdata


def _statuses(ptm_mdata: MuData, global_mdata: MuData) -> dict[str, str]:
    adjuster = PTMProteinAdjuster(ptm_mdata, global_mdata, ptm_mod="phospho_site", global_mod="protein")
    return adjuster.resolution["adjustment_status"].to_dict()


def test_site_is_adjustable_when_its_peptide_was_never_seen_in_global():
    """The case that motivated the accession route.

    Enrichment means a phospho peptide is routinely absent from the global run, but its protein is
    usually quantified there from other peptides. Keying on the accession rather than the peptide is
    what makes that denominator reachable -- and there is no peptide-level lookup left to fail.
    """
    ptm = _make_ptm({"P1|S30": "P1"})
    global_mdata = _make_global(quantified_groups=["P1"], accession_to_group={"P1": "P1"})

    assert _statuses(ptm, global_mdata) == {"P1|S30": ADJUSTMENT_STATUS_ADJUSTED}


def test_paralogs_indistinguishable_in_global_resolve_to_their_shared_group():
    """Accessions that global could not tell apart are one group, so there is no ambiguity."""
    ptm = _make_ptm({"P2|S30;P7|S45": "P2;P7"})
    global_mdata = _make_global(
        quantified_groups=["P2,P7"],
        accession_to_group={"P2": "P2,P7", "P7": "P2,P7"},
    )

    adjuster = PTMProteinAdjuster(ptm, global_mdata, ptm_mod="phospho_site", global_mod="protein")

    assert adjuster.resolution["adjustment_status"].tolist() == [ADJUSTMENT_STATUS_ADJUSTED]
    assert adjuster.resolution["denominator_group"].tolist() == ["P2,P7"]


def test_site_spanning_two_quantified_groups_is_left_unadjusted():
    """No valid denominator exists, so this is a refusal rather than a lookup failure.

    The measured signal is a sum over both groups, and protein rollup values carry a per-protein
    offset that makes them incomparable across groups -- neither picking one nor summing them is
    defined.
    """
    ptm = _make_ptm({"P3|S30;P9|S45": "P3;P9"})
    global_mdata = _make_global(
        quantified_groups=["P3", "P9"],
        accession_to_group={"P3": "P3", "P9": "P9"},
    )

    assert _statuses(ptm, global_mdata) == {"P3|S30;P9|S45": ADJUSTMENT_STATUS_SHARED_GROUPS}


def test_group_without_quantification_is_reported_separately():
    """The accession maps to a group, but that group never reached the protein matrix."""
    ptm = _make_ptm({"P5|S10": "P5"})
    global_mdata = _make_global(quantified_groups=["P1"], accession_to_group={"P5": "P5"})

    assert _statuses(ptm, global_mdata) == {"P5|S10": ADJUSTMENT_STATUS_NOT_QUANTIFIED}


def test_accession_in_global_fasta_but_not_identified_is_not_a_configuration_error():
    ptm = _make_ptm({"P8|S10": "P8"})
    global_mdata = _make_global(
        quantified_groups=["P1"],
        accession_to_group={"P1": "P1"},
        fasta_accessions=["P1", "P8"],
    )

    assert _statuses(ptm, global_mdata) == {"P8|S10": ADJUSTMENT_STATUS_NO_GLOBAL_GROUP}


def test_accession_missing_from_global_fasta_is_flagged_as_a_different_database():
    """Absent-from-results and absent-from-database look alike but mean opposite things.

    The first is enrichment working as intended; the second means the two searches used different
    FASTAs, which no adjustment policy should paper over.
    """
    ptm = _make_ptm({"Q99|S10": "Q99"})
    global_mdata = _make_global(
        quantified_groups=["P1"],
        accession_to_group={"P1": "P1"},
        fasta_accessions=["P1", "P8"],
    )

    assert _statuses(ptm, global_mdata) == {"Q99|S10": ADJUSTMENT_STATUS_NOT_IN_GLOBAL_FASTA}


def test_without_a_global_fasta_the_two_absences_are_reported_together():
    ptm = _make_ptm({"Q99|S10": "Q99"})
    global_mdata = _make_global(quantified_groups=["P1"], accession_to_group={"P1": "P1"})

    assert _statuses(ptm, global_mdata) == {"Q99|S10": ADJUSTMENT_STATUS_NO_GLOBAL_GROUP}


def test_unadjusted_sites_survive_and_x_keeps_the_unadjusted_intensities():
    """Adjustment must not truncate features or overwrite .X.

    Residuals and raw abundances are different quantities; leaving both in one matrix would let
    downstream testing compare them along the same axis with no way to tell them apart. Dropping the
    unadjustable sites instead would destroy the unadjusted analysis entirely.
    """
    ptm = _make_ptm({"P1|S30": "P1", "P3|S30;P9|S45": "P3;P9"})
    original_x = ptm["phospho_site"].X.copy()
    global_mdata = _make_global(
        quantified_groups=["P1", "P3", "P9"],
        accession_to_group={"P1": "P1", "P3": "P3", "P9": "P9"},
    )

    adjusted = mm.pp.adjust_ptm_by_protein(ptm, global_mdata, modality="phospho_site", rescale=False)
    site_adata = adjusted["phospho_site"]

    assert list(site_adata.var_names) == ["P1|S30", "P3|S30;P9|S45"]
    assert np.allclose(site_adata.X, original_x)

    adjusted_layer = site_adata.layers["protein_adjusted"]
    assert not np.isnan(adjusted_layer[:, 0]).any()
    assert np.isnan(adjusted_layer[:, 1]).all()
    assert site_adata.var["is_protein_adjusted"].tolist() == [True, False]
    assert site_adata.var["adjustment_status"].tolist() == [
        ADJUSTMENT_STATUS_ADJUSTED,
        ADJUSTMENT_STATUS_SHARED_GROUPS,
    ]


def test_ratio_is_the_default_estimator():
    """ridge fits a slope from as many points as there are samples and shrinks it toward zero;
    subtraction assumes the slope-one relationship mass action predicts and needs no fitting."""
    import inspect

    assert inspect.signature(mm.pp.adjust_ptm_by_protein).parameters["method"].default == "ratio"


def test_ridge_alpha_is_reachable_from_the_public_api():
    """A hidden penalty decides whether ridge regresses at all, so it must be settable."""
    ptm = _make_ptm({"P1|S30": "P1"})
    global_mdata = _make_global(quantified_groups=["P1"], accession_to_group={"P1": "P1"})

    adjusted = mm.pp.adjust_ptm_by_protein(
        ptm, global_mdata, modality="phospho_site", method="ridge", ridge_alpha=1e6, rescale=False
    )

    # With an enormous penalty the slope collapses to zero, so the residual is the site centred on
    # its own mean -- the protein axis is not removed at all.
    residuals = adjusted["phospho_site"].layers["protein_adjusted"][:, 0]
    assert residuals == pytest.approx(residuals - np.nanmean(residuals), abs=1e-6)


def test_unknown_method_is_rejected():
    ptm = _make_ptm({"P1|S30": "P1"})
    global_mdata = _make_global(quantified_groups=["P1"], accession_to_group={"P1": "P1"})

    with pytest.raises(ValueError, match="Unknown PTM adjustment method"):
        mm.pp.adjust_ptm_by_protein(ptm, global_mdata, modality="phospho_site", method="_rescale")


def test_missing_global_sample_is_named_in_the_error():
    ptm = _make_ptm({"P1|S30": "P1"})
    global_mdata = _make_global(quantified_groups=["P1"], accession_to_group={"P1": "P1"})
    global_mdata = MuData({"protein": global_mdata["protein"][:3].copy()})
    global_mdata.uns["protein_map"] = pd.DataFrame({"initial_protein": ["P1"], "protein_group": ["P1"]})

    with pytest.raises(ValueError, match="missing samples"):
        PTMProteinAdjuster(ptm, global_mdata, ptm_mod="phospho_site", global_mod="protein")


def test_global_without_protein_map_is_rejected_with_a_usable_message():
    ptm = _make_ptm({"P1|S30": "P1"})
    global_mdata = _make_global(quantified_groups=["P1"], accession_to_group={"P1": "P1"})
    del global_mdata.uns["protein_map"]

    with pytest.raises(ValueError, match="protein_map"):
        PTMProteinAdjuster(ptm, global_mdata, ptm_mod="phospho_site", global_mod="protein")


def test_end_to_end_from_real_inference_output_through_site_adjustment():
    """Full flow on inference output: infer_protein's protein_map feeds the denominator search.

    Covers the three outcomes that matter together, on one container, with the global groups produced
    by the real parsimony rather than hand-written:

    * a phospho peptide the global run never observed, whose protein it did quantify -> adjusted
    * a peptide on two paralogs the global run cannot tell apart -> one group, so still adjusted
    * a peptide on two proteins the global run *can* tell apart -> no valid denominator
    """
    from msmu._preprocessing._infer_protein import get_protein_mapping

    # Global evidence: P2/P7 are indistinguishable; P3 and P9 are separable but share one peptide.
    global_peptides = pd.Series(["AAAK", "BBBK", "AAAK", "BBBK", "CCCK", "SHAREDK", "EEEK", "SHAREDK", "DDDK"])
    global_proteins = pd.Series(["P2", "P2", "P7", "P7", "P3", "P3", "P9", "P9", "P1"])
    peptide_map, protein_map = get_protein_mapping(global_peptides, global_proteins)

    # to_protein quantifies unique peptides only, so a group backed solely by a shared peptide never
    # reaches the protein matrix.
    unique_groups = sorted({group for group in peptide_map["protein_group"] if ";" not in group})
    global_values = np.arange(len(SAMPLES) * len(unique_groups), dtype=float).reshape(len(SAMPLES), len(unique_groups))
    global_mdata = MuData(
        {
            "protein": AnnData(
                X=global_values,
                obs=pd.DataFrame(index=SAMPLES),
                var=pd.DataFrame(index=unique_groups),
            )
        }
    )
    global_mdata.uns["protein_map"] = protein_map

    ptm = _make_ptm(
        {
            # never in global's peptide_map -- only reachable through its accession
            "P1|S30": "P1",
            "P2|S30;P7|S30": "P2;P7",
            "P3|S40;P9|S40": "P3;P9",
        }
    )

    adjusted = mm.pp.adjust_ptm_by_protein(ptm, global_mdata, modality="phospho_site", rescale=False)
    site_var = adjusted["phospho_site"].var

    assert site_var["adjustment_status"].tolist() == [
        ADJUSTMENT_STATUS_ADJUSTED,
        ADJUSTMENT_STATUS_ADJUSTED,
        ADJUSTMENT_STATUS_SHARED_GROUPS,
    ]
    # The paralog pair resolves to the single group the global parsimony merged them into.
    assert site_var["denominator_group"].tolist()[1] == "P2,P7"
    assert site_var["is_protein_adjusted"].tolist() == [True, True, False]


def test_status_records_when_the_estimator_declined_to_produce_a_value():
    """ridge drops a site it cannot fit, which must not leave the status column saying 'adjusted'.

    The denominator search succeeded; it is the estimator that refused, and the two are different
    reasons a site ends up without a value.
    """
    from msmu._preprocessing._normalisation import ADJUSTMENT_STATUS_NO_ESTIMATE

    ptm = _make_ptm({"P1|S30": "P1"})
    # Only two samples carry a paired observation, one short of what ridge requires.
    ptm["phospho_site"].X[2:, 0] = np.nan
    global_mdata = _make_global(quantified_groups=["P1"], accession_to_group={"P1": "P1"})

    adjusted = mm.pp.adjust_ptm_by_protein(ptm, global_mdata, modality="phospho_site", method="ridge", rescale=False)
    site_var = adjusted["phospho_site"].var

    assert site_var["adjustment_status"].tolist() == [ADJUSTMENT_STATUS_NO_ESTIMATE]
    assert site_var["is_protein_adjusted"].tolist() == [False]
    assert site_var["denominator_group"].tolist() == ["P1"]
