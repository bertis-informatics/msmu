from msmu._provenance import get_log
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData

from msmu._preprocessing._infer_protein import (
    _make_peptide_map,
    _make_protein_map,
    infer_protein,
)
from msmu._utils.protein import select_representative


def _make_peptide_mdata(*, modality: str = "peptide") -> MuData:
    obs = pd.DataFrame(index=["s1", "s2"])
    var = pd.DataFrame(
        {
            "stripped_peptide": ["pep1", "pep2"],
            "proteins": ["P1", "P2"],
        },
        index=["f1", "f2"],
    )
    adata = AnnData(X=np.array([[1.0, 2.0], [3.0, 4.0]]), obs=obs, var=var)

    return MuData({modality: adata})


def _make_target_and_decoy_peptide_mdata(peptides: list[str], proteins: list[str], decoy: list[int]) -> MuData:
    var = pd.DataFrame(
        {"stripped_peptide": peptides, "proteins": proteins, "decoy": decoy},
        index=[f"f{position}" for position in range(len(peptides))],
    )
    adata = AnnData(X=np.ones((2, len(peptides))), obs=pd.DataFrame(index=["s1", "s2"]), var=var)
    return MuData({"peptide": adata})


def test_select_representative_prefers_swissprot_canonical():
    protein_group = "P1,P2;P3"
    protein_info = {"P1": "sp_P1", "P2": "sp_P2-2", "P3": "tr_P3"}
    assert select_representative(protein_group, protein_info) == "P1"


def test_select_representative_uses_contaminant_behavior():
    protein_group = "C1"
    protein_info = {"C1": "contam_sp_Cont_C1"}
    assert select_representative(protein_group, protein_info) == "Cont_C1"


def test_select_representative_ranks_contaminant_below_swissprot():
    """A Hao-style contaminant carries the ``sp`` database tag, so it must not outrank a target."""
    protein_group = "C1,P1"
    protein_info = {"C1": "sp_Cont_C1", "P1": "sp_P1"}
    assert select_representative(protein_group, protein_info) == "P1"


def test_make_peptide_map_groups():
    df = pd.DataFrame({"peptide": ["p1", "p1", "p2"], "protein": ["A", "B", "C"]})
    peptide_map = _make_peptide_map(df)
    assert peptide_map.loc[peptide_map["peptide"] == "p1", "protein_group"].iloc[0] == "A;B"


def test_make_protein_map_flags():
    initial = pd.DataFrame({"protein": ["A", "B", "C"]})
    subset_map = {"B": "A"}
    indist_map = {}
    subsum_map = {"C": "A"}
    protein_map = _make_protein_map(initial, subset_map, indist_map, subsum_map)
    assert bool(protein_map.loc[protein_map["initial_protein"] == "B", "subsetted"].iloc[0]) is True


def test_infer_protein_supports_custom_modality_name():
    mdata = _make_peptide_mdata(modality="phospho_site")

    out = infer_protein(mdata, modality="phospho_site")

    assert out.mod["phospho_site"].var["protein_group"].tolist() == ["P1", "P2"]
    assert out.mod["phospho_site"].var["peptide_type"].tolist() == ["unique", "unique"]
    assert set(out.uns) >= {"peptide_map", "protein_map"}


def test_infer_protein_logs_protein_inference_stats(caplog):
    caplog.set_level("INFO", logger="msmu")
    mdata = _make_peptide_mdata()

    out = infer_protein(mdata)

    entry = get_log(out)["events"][0]
    assert entry["function"] == "infer_protein"
    assert "stdout" not in entry
    stdout = caplog.text
    assert "Initial proteins: 2" in stdout
    assert "Removed indistinguishable: 0" in stdout
    assert "Removed subsettable: 0" in stdout
    assert "Removed subsumable: 0" in stdout


def test_infer_protein_annotates_decoy_rows_and_maps_only_the_targets():
    mdata = _make_target_and_decoy_peptide_mdata(
        peptides=["pep1", "pep2", "dpep1"], proteins=["P1", "P2", "decoy_P1"], decoy=[0, 0, 1]
    )

    out = infer_protein(mdata)

    var = out.mod["peptide"].var
    assert var["protein_group"].tolist() == ["P1", "P2", "decoy_P1"]
    assert var["peptide_type"].tolist() == ["unique", "unique", "unique"]
    assert sorted(out.uns["peptide_map"]["peptide"]) == ["pep1", "pep2"]
    assert sorted(out.uns["protein_map"]["initial_protein"]) == ["P1", "P2"]


def test_infer_protein_keeps_decoy_groups_apart_from_target_groups():
    """The decoy flag, not the accession, separates the inferences.

    Here the decoy carries a target accession, as DIA-NN reports it. Inferred together, the decoy
    peptide would make P2 a subset of P1 and shrink the target group to P1.
    """
    mdata = _make_target_and_decoy_peptide_mdata(peptides=["pep1", "dpep1"], proteins=["P1;P2", "P1"], decoy=[0, 1])

    var = infer_protein(mdata).mod["peptide"].var

    assert var["protein_group"].tolist() == ["P1,P2", "P1"]
    assert var["peptide_type"].tolist() == ["unique", "unique"]


def test_infer_protein_refuses_a_decoy_group_named_like_a_target_group():
    """to_protein groups by name, so the two would become one protein row flagged as a decoy.

    msmu's readers prefix decoy accessions; a modality built elsewhere may carry the target's.
    """
    mdata = _make_target_and_decoy_peptide_mdata(peptides=["pep1", "dpep1"], proteins=["P1", "P1"], decoy=[0, 1])

    with pytest.raises(ValueError, match="share their name with a target group"):
        infer_protein(mdata)


def test_infer_protein_missing_modality_raises():
    mdata = _make_peptide_mdata()

    with pytest.raises(ValueError, match="Modality 'protein' not found"):
        infer_protein(mdata, modality="protein")


def test_infer_protein_missing_required_columns_raises():
    mdata = _make_peptide_mdata()
    mdata.mod["peptide"].var = pd.DataFrame({"proteins": ["P1", "P2"]}, index=["f1", "f2"])

    with pytest.raises(ValueError, match="Required columns missing"):
        infer_protein(mdata)
