"""Preprocessing must behave identically on a MuData read back from .h5mu.

``write_h5mu`` stores var string columns as categorical, so every column of a MuData loaded from
disk is categorical rather than ``str``. Two behaviours are pinned here.

1. Value destruction (a real failure). On pandas 3, ``.str.split`` applied to a categorical returns
   the *repr* of the split list (``"['P0C7N9', 'D3Z016']"``) instead of the list, so the following
   ``explode`` is a no-op and the accessions are silently fused into one bogus protein. (pandas 2
   returned an object column of real lists, so this only surfaced on the pandas 3 stack.) The PTM
   path explodes twice, on ``;`` then ``,``, which shredded the repr string into fragments like
   ``"['A"``.
2. Ghost rows (hardening). A group-by key whose categories include values absent from the rows used
   to emit a group per unused category -- an empty entry in ``uns['peptide_map']`` and a phantom
   feature row quantified as 0 rather than missing. No route to that state via the public API was
   found: anndata drops unused categories when features are subset, and the frames kept in ``uns``
   come back from disk as ``str``, never categorical. These tests construct the state directly, so
   they guard the invariant rather than reproduce a reported failure.

Note that pandas 3 already defaults ``groupby(observed=True)``; the explicit ``observed=False`` this
replaced was an opt-in to the older behaviour.
"""

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData

import msmu as mm
from msmu._preprocessing._infer_protein import get_protein_mapping, infer_protein
from msmu._preprocessing._summarisation import Aggregator, PtmSummarisationPrep
from msmu._utils._pandas import split_delimited_strings

# pep1/pep4 are shared between two distinguishable proteins, pep2/pep3 are unique to one each -- so
# the peptides land in different groups and carry different peptide_type values. A fixture of only
# indistinguishable proteins would collapse to a single group and hide a broken split. The repeated
# "P1;P2" also matters: anndata only stores a string column as categorical when it has fewer
# categories than rows, so an all-distinct column would round-trip as ``str`` and prove nothing.
PEPTIDES = ["pep1", "pep2", "pep3", "pep4"]
PROTEINS = ["P1;P2", "P1", "P2", "P1;P2"]


def _make_peptide_mdata() -> MuData:
    var = pd.DataFrame(
        {"stripped_peptide": PEPTIDES, "proteins": PROTEINS},
        index=[f"f{feature_number}" for feature_number in range(len(PEPTIDES))],
    )
    adata = AnnData(
        X=np.ones((2, len(PEPTIDES)), dtype="float32"),
        obs=pd.DataFrame(index=["s1", "s2"]),
        var=var,
    )

    return MuData({"peptide": adata})


def test_split_delimited_strings_splits_categorical_into_lists():
    values = pd.Series(["A;B", "C", None], dtype="category")

    split_values = split_delimited_strings(values, ";")

    assert split_values.tolist()[:2] == [["A", "B"], ["C"]]
    assert split_values.isna().iloc[2]


def test_infer_protein_is_unchanged_by_an_h5mu_roundtrip(tmp_path):
    mdata = _make_peptide_mdata()
    in_memory_var = infer_protein(mdata).mod["peptide"].var

    path = tmp_path / "peptide.h5mu"
    mdata.write(path)
    roundtripped = mm.read_h5mu(path)
    assert isinstance(roundtripped.mod["peptide"].var["proteins"].dtype, pd.CategoricalDtype)

    roundtripped_var = infer_protein(roundtripped).mod["peptide"].var

    assert roundtripped_var["protein_group"].tolist() == in_memory_var["protein_group"].tolist()
    assert roundtripped_var["peptide_type"].tolist() == in_memory_var["peptide_type"].tolist()
    # Guard the values themselves, not just their agreement: a split that fails on both sides
    # would keep them equal while destroying the accessions.
    assert roundtripped_var["protein_group"].tolist() == ["P1;P2", "P1", "P2", "P1;P2"]
    assert roundtripped_var["peptide_type"].tolist() == ["shared", "unique", "unique", "shared"]


def test_infer_protein_ignores_stale_peptide_categories():
    """Categories left behind by filtering must not become empty peptide_map rows."""
    peptides = pd.Series(pd.Categorical(PEPTIDES, categories=[*PEPTIDES, "pep_filtered_out"]))

    peptide_map, _protein_map = get_protein_mapping(peptides, pd.Series(PROTEINS))

    assert peptide_map["peptide"].tolist() == PEPTIDES


def test_ptm_explode_splits_categorical_protein_groups():
    ptm_info = pd.DataFrame(
        {"protein_group": pd.Series(["P1,P2;P3", "P4"], dtype="category"), "peptide": ["pep1", "pep2"]},
    )

    exploded_groups = PtmSummarisationPrep._explode_protein_groups(None, ptm_info)
    exploded_proteins = PtmSummarisationPrep._explode_protein_group(None, exploded_groups)

    assert exploded_groups["_prot_gr"].tolist() == ["P1,P2", "P3", "P4"]
    assert exploded_proteins["_prots"].tolist() == ["P1", "P2", "P3", "P4"]


def test_aggregator_ignores_stale_feature_categories():
    """A stale category must not become a phantom feature quantified as 0."""
    identification_df = pd.DataFrame(
        {
            "peptide": pd.Categorical(["pep1", "pep2"], categories=["pep1", "pep2", "pep_filtered_out"]),
            "proteins": ["P1", "P2"],
            "stripped_peptide": ["pep1", "pep2"],
            "PEP": [0.01, 0.02],
        }
    )
    quantification_df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["s1", "s2"])

    aggregator = Aggregator.peptide(
        identification_df,
        quantification_df,
        None,
        "sum",
        "best_pep",
        "proteins",
        "peptide",
    )

    assert aggregator.aggregate_identification().index.tolist() == ["pep1", "pep2"]
    assert aggregator.aggregate_quantification().index.tolist() == ["pep1", "pep2"]
