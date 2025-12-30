import pandas as pd

from msmu._preprocessing._infer_protein import _make_peptide_map, _make_protein_map, select_representative


def test_select_representative_prefers_swissprot_canonical():
    protein_group = "P1,P2;P3"
    protein_info = {"P1": "sp_P1", "P2": "sp_P2-2", "P3": "tr_P3"}
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
