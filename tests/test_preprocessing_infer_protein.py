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


def _make_peptide_mdata(
    *,
    modality: str = "peptide",
    include_decoy: bool = False,
) -> MuData:
    obs = pd.DataFrame(index=["s1", "s2"])
    var = pd.DataFrame(
        {
            "stripped_peptide": ["pep1", "pep2"],
            "proteins": ["P1", "P2"],
        },
        index=["f1", "f2"],
    )
    adata = AnnData(X=np.array([[1.0, 2.0], [3.0, 4.0]]), obs=obs, var=var)
    if include_decoy:
        adata.uns["decoy"] = pd.DataFrame(
            {
                "stripped_peptide": ["dpep1"],
                "proteins": ["DP1"],
            },
            index=["d1"],
        )

    return MuData({modality: adata})


def test_select_representative_prefers_swissprot_canonical():
    protein_group = "P1,P2;P3"
    protein_info = {"P1": "sp_P1", "P2": "sp_P2-2", "P3": "tr_P3"}
    assert select_representative(protein_group, protein_info) == "P1"


def test_select_representative_uses_contaminant_behavior():
    protein_group = "C1"
    protein_info = {"C1": "contam_sp_contam_C1"}
    assert select_representative(protein_group, protein_info) == "contam_C1"


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


def test_infer_protein_cmd_stdout_records_protein_inference_stats():
    mdata = _make_peptide_mdata()

    out = infer_protein(mdata)

    entry = out.uns["_cmd"]["0"]
    assert entry["function"] == "infer_protein"
    assert "stdout" in entry
    stdout = entry["stdout"]
    assert "INFO - Initial proteins: 2" in stdout
    assert "INFO - Removed indistinguishable: 0" in stdout
    assert "INFO - Removed subsettable: 0" in stdout
    assert "INFO - Removed subsumable: 0" in stdout


def test_infer_protein_annotates_decoys():
    mdata = _make_peptide_mdata(include_decoy=True)

    out = infer_protein(mdata)

    decoy_df = out.mod["peptide"].uns["decoy"]
    assert decoy_df["protein_group"].tolist() == ["DP1"]
    assert decoy_df["peptide_type"].tolist() == ["unique"]


def test_infer_protein_uses_propagated_mapping_from_path(monkeypatch):
    mdata = _make_peptide_mdata()
    propagated = MuData(
        {
            "peptide": AnnData(
                X=np.array([[1.0]]),
                obs=pd.DataFrame(index=["s1"]),
                var=pd.DataFrame(index=["f1"]),
            )
        }
    )
    propagated.uns["peptide_map"] = pd.DataFrame(
        {
            "peptide": ["pep1", "pep2"],
            "protein_group": ["G1", "G1;G2"],
        }
    )
    propagated.uns["protein_map"] = pd.DataFrame(
        {
            "initial_protein": ["P1", "P2"],
            "protein_group": ["G1", "G1;G2"],
            "indistinguishable": [False, False],
            "subsetted": [False, False],
            "subsumable": [False, False],
        }
    )

    monkeypatch.setattr("msmu._preprocessing._infer_protein.read_h5mu", lambda _: propagated)

    out = infer_protein(mdata, propagated_from="mapping.h5mu")

    assert out.mod["peptide"].var["protein_group"].tolist() == ["G1", "G1;G2"]
    assert out.mod["peptide"].var["peptide_type"].tolist() == ["unique", "shared"]
    pd.testing.assert_frame_equal(out.uns["protein_map"], propagated.uns["protein_map"])


def test_infer_protein_missing_modality_raises():
    mdata = _make_peptide_mdata()

    with pytest.raises(ValueError, match="Modality 'protein' not found"):
        infer_protein(mdata, modality="protein")


def test_infer_protein_missing_required_columns_raises():
    mdata = _make_peptide_mdata()
    mdata.mod["peptide"].var = pd.DataFrame({"proteins": ["P1", "P2"]}, index=["f1", "f2"])

    with pytest.raises(ValueError, match="Required columns missing"):
        infer_protein(mdata)
