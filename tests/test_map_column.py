from msmu._core._provenance import _event_inputs
import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

import msmu as mm


def data():
    psm = ad.AnnData(X=np.ones((1, 3)))
    psm.varm["search_result"] = pd.DataFrame(
        {"sequence": ["a", "a", "b"], "group": ["G1", "G1", "G2"],
         "unique": pd.array([True, True, False], dtype="boolean")},
        index=psm.var_names,
    )
    peptide = ad.AnnData(X=np.ones((1, 3)), var=pd.DataFrame(
        {"match": ["b", "a", "missing"], "protein_group": ["old"] * 3}, index=["b", "a", "missing"]
    ))
    return md.MuData({"psm": psm, "peptide": peptide})


@pytest.mark.parametrize("target_index", [None, "match"])
def test_mapping_and_saved_replay(tmp_path, target_index):
    original = tmp_path / "input.h5mu"
    data().write_h5mu(original)
    with mm.pv.options(hashing=True):
        m = mm.read_h5mu(original)
        result = mm.dt.map(
            m, source="psm.varm.search_result", target="peptide.var",
            source_index="sequence", target_index=target_index,
            columns={"group": "protein_group", "unique": "peptide_type"},
        )
    assert result is m
    assert m["peptide"].var["protein_group"].iloc[:2].tolist() == ["G2", "G1"]
    assert pd.isna(m["peptide"].var["protein_group"].iloc[2])
    assert m["peptide"].var["peptide_type"].iloc[:2].tolist() == [False, True]
    expected = m["peptide"].var.copy()
    saved = tmp_path / "result.h5mu"
    m.write_h5mu(saved)
    history = mm.pv.get_log(md.read_h5mu(saved))
    event = history["events"][-1]
    assert set(event["parameters"]) == {"mdata", "source", "target", "source_index", "target_index", "columns"}
    assert len(_event_inputs(event)) == 1  # Only MuData, no copied column payload.
    pd.testing.assert_frame_equal(mm.pv.replay(history)["peptide"].var, expected)
    namespace = {}
    exec(mm.pv.to_script(history), namespace)
    pd.testing.assert_frame_equal(namespace["mdata"]["peptide"].var, expected)


def test_map_onto_decoy_table_and_saved_replay(tmp_path):
    # The tutorial case: a sample annotation keyed by filename, carried onto the decoy table so
    # that a variable filter on it can also be applied to decoys.
    psm = ad.AnnData(X=np.ones((2, 2)), obs=pd.DataFrame({"cell": ["C10", "SVEC"]}, index=["f1", "f2"]))
    psm.var["filename"] = ["f1", "f2"]
    psm.uns["decoy"] = pd.DataFrame({"filename": ["f2", "f1", "f3"]}, index=["d1", "d2", "d3"])
    original = tmp_path / "input.h5mu"
    md.MuData({"psm": psm}).write_h5mu(original)
    with mm.pv.options(hashing=True):
        m = mm.read_h5mu(original)
        mm.dt.map(m, source="psm.obs", target="psm.decoy", source_index=None, target_index="filename",
                  columns={"cell": "cell"})
    decoy = m["psm"].uns["decoy"]
    assert decoy["cell"].iloc[:2].tolist() == ["SVEC", "C10"]
    assert pd.isna(decoy["cell"].iloc[2])
    expected = decoy.copy()
    saved = tmp_path / "result.h5mu"
    m.write_h5mu(saved)
    history = mm.pv.get_log(md.read_h5mu(saved))
    pd.testing.assert_frame_equal(mm.pv.replay(history)["psm"].uns["decoy"], expected)
    namespace = {}
    exec(mm.pv.to_script(history), namespace)
    pd.testing.assert_frame_equal(namespace["mdata"]["psm"].uns["decoy"], expected)


def test_default_source_index_and_overlapping_columns():
    m = data()
    mm.dt.map(m, source="peptide.var", target="peptide.var",
                     columns={"match": "protein_group", "protein_group": "match"})
    assert m["peptide"].var["protein_group"].tolist() == ["b", "a", "missing"]
    assert m["peptide"].var["match"].tolist() == ["old"] * 3


@pytest.mark.parametrize("problem", ["conflict", "missing_column", "invalid_path", "duplicate_destination"])
def test_invalid_mapping_does_not_mutate(problem):
    m = data()
    kwargs = dict(source="psm.varm.search_result", target="peptide.var", source_index="sequence",
                  columns={"group": "protein_group", "unique": "peptide_type"})
    if problem == "conflict":
        m["psm"].varm["search_result"].iloc[1, 1] = "different"
    elif problem == "missing_column":
        kwargs["columns"]["absent"] = "absent"
    elif problem == "invalid_path":
        kwargs["target"] = "peptide.uns"
    else:
        kwargs["columns"]["unique"] = "protein_group"
    before = mm.pv.compute_hash(m)
    with pytest.raises((ValueError, KeyError)):
        mm.dt.map(m, **kwargs)
    assert mm.pv.compute_hash(m) == before
