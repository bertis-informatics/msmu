"""copy_mudata, add_modality, replace_modality and refresh_global_obs must leave a MuData in exactly the
state mudata's own copy() and update() produce.

The reference in every test is mudata itself, so these tests also detect a change in mudata's
bookkeeping layout: if mudata starts storing varmap or varm differently, the comparisons here fail
before any provenance hash does.
"""

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
from _mudata_bookkeeping import assert_bookkeeping_is_final
from _mudata_bookkeeping import global_bookkeeping as _global_bookkeeping

import msmu as mm
from msmu._core._hashing import compute_hash
from msmu._utils._mudata import (
    MODALITY_MAP_DTYPE,
    _rebuild_global_var,
    add_modality,
    copy_mudata,
    refresh_global_obs,
    replace_modality,
)


def _swap_psm_for_its_first_two_features_by_hand(mdata: md.MuData) -> None:
    """A raw modality swap, as a user might do it: the bookkeeping is left behind until update()."""
    mdata.mod["psm"] = mdata["psm"][:, :2].copy()


def _make_unrebuildable(container: md.MuData, how: str) -> None:
    """Give the container a global var layout _rebuild_global_var cannot build directly."""
    if how == "global_var_column":
        container.var["pulled"] = np.arange(container.n_vars)
    elif how == "user_varm":
        container.varm["custom"] = np.zeros((container.n_vars, 2))
    else:
        container.varp["distance"] = np.zeros((container.n_vars, container.n_vars))


def _new_modality(mdata: md.MuData, var_names: list[str]) -> ad.AnnData:
    return ad.AnnData(
        X=np.arange(mdata.n_obs * len(var_names), dtype=float).reshape(mdata.n_obs, len(var_names)),
        obs=pd.DataFrame(index=mdata.obs_names),
        var=pd.DataFrame(index=var_names),
    )


# --- copy_mudata -------------------------------------------------------------------------------------


def test_copy_mudata_matches_mudata_copy(mdata: md.MuData) -> None:
    reference = mdata.copy()

    copied = copy_mudata(mdata)

    assert copied is not mdata
    assert _global_bookkeeping(copied) == _global_bookkeeping(reference)
    # A faithful copy keeps the source's obs as it is; MuData.copy() may re-derive a pulled column with
    # the modality's dtype, which the provenance hash does not distinguish.
    assert copied.obs.equals(mdata.obs)
    assert compute_hash(copied) == compute_hash(reference) == compute_hash(mdata)
    for modality_name in mdata.mod:
        np.testing.assert_array_equal(copied[modality_name].X, reference[modality_name].X)
        assert copied[modality_name].var.equals(reference[modality_name].var)
    assert list(copied["protein"].obsm.keys()) == list(reference["protein"].obsm.keys())


def test_copy_mudata_does_not_alias_the_source(mdata: md.MuData) -> None:
    source_hash = compute_hash(mdata)

    copied = copy_mudata(mdata)
    copied["psm"].X[0, 0] = -1.0
    copied["psm"].var["score"] = 0.0
    copied.obs["group"] = "B"
    copied.uns["plotting"] = {"changed": True}
    copied["protein"].obsm["X_pca"].iloc[0, 0] = 99.0

    assert compute_hash(mdata) == source_hash
    assert mdata["psm"].X[0, 0] == 1.0
    assert mdata["protein"].obsm["X_pca"].iloc[0, 0] == 1.0


def test_copy_mudata_keeps_the_input_hash_when_branching(mdata: md.MuData) -> None:
    input_hash = compute_hash(mdata)

    first_branch = mm.pp.log2_transform(mdata, modality="psm")
    second_branch = mm.pp.log2_transform(mdata, modality="psm")

    assert compute_hash(mdata) == input_hash
    assert compute_hash(first_branch) == compute_hash(second_branch)


def test_copy_mudata_of_a_view_materialises_it(mdata: md.MuData) -> None:
    view = mdata[:2, :]
    assert view.is_view

    copied = copy_mudata(view)

    assert not copied.is_view
    assert _global_bookkeeping(copied) == _global_bookkeeping(view.copy())
    assert copied.n_obs == 2


def test_copy_mudata_does_not_rebuild_the_bookkeeping(mdata: md.MuData) -> None:
    """The contract: a MuData edited by hand needs mudata's update(), as it does without msmu."""
    edited = copy_mudata(mdata)
    _swap_psm_for_its_first_two_features_by_hand(edited)

    copied = copy_mudata(edited)

    assert copied.var_names.tolist() == edited.var_names.tolist()  # still the old global var
    copied.update()
    assert _global_bookkeeping(copied) == _global_bookkeeping(edited.copy())
    assert compute_hash(copied) == compute_hash(edited.copy())


# --- _rebuild_global_var -----------------------------------------------------------------------------


def test_rebuild_global_var_matches_mudata_update_var(mdata: md.MuData) -> None:
    rebuilt = copy_mudata(mdata)
    reference = copy_mudata(mdata)
    _swap_psm_for_its_first_two_features_by_hand(rebuilt)
    _swap_psm_for_its_first_two_features_by_hand(reference)

    _rebuild_global_var(rebuilt)
    reference.update_var()

    assert _global_bookkeeping(rebuilt) == _global_bookkeeping(reference)
    assert compute_hash(rebuilt) == compute_hash(reference)
    # mudata's change cache is keyed by the current modalities after an update; ours must be too.
    assert set(rebuilt._varhash) == set(reference._varhash) == set(rebuilt.mod)


def test_rebuild_global_var_uses_mudata_layout(mdata: md.MuData) -> None:
    rebuilt = copy_mudata(mdata)
    _swap_psm_for_its_first_two_features_by_hand(rebuilt)

    _rebuild_global_var(rebuilt)

    assert rebuilt.var.index.tolist() == rebuilt["psm"].var_names.tolist() + rebuilt["protein"].var_names.tolist()
    assert rebuilt.var.shape[1] == 0
    assert rebuilt.var.columns.dtype == mdata.copy().var.columns.dtype
    for modality_name, modality in rebuilt.mod.items():
        varmap = rebuilt.varmap[modality_name]
        assert varmap.dtype == MODALITY_MAP_DTYPE
        assert varmap[varmap > 0].tolist() == list(range(1, modality.n_vars + 1))
        mask = rebuilt.varm[modality_name]
        assert mask.dtype == bool and mask.shape == (rebuilt.n_vars, 1)
        np.testing.assert_array_equal(mask.ravel(), varmap > 0)


@pytest.mark.parametrize("make_unrebuildable", ["global_var_column", "user_varm", "varp"])
def test_rebuild_global_var_falls_back_to_mudata_update_var(mdata: md.MuData, make_unrebuildable: str) -> None:
    rebuilt = copy_mudata(mdata)
    reference = copy_mudata(mdata)
    for container in (rebuilt, reference):
        _make_unrebuildable(container, make_unrebuildable)
        _swap_psm_for_its_first_two_features_by_hand(container)

    _rebuild_global_var(rebuilt)
    reference.update_var()

    assert _global_bookkeeping(rebuilt) == _global_bookkeeping(reference)
    assert rebuilt.var.columns.tolist() == reference.var.columns.tolist()


# --- refresh_global_obs ------------------------------------------------------------------------------


def test_refresh_global_obs_finishes_relabelled_observations(mdata: md.MuData) -> None:
    relabelled = copy_mudata(mdata)
    for modality in relabelled.mod.values():
        modality.obs_names = [f"renamed_{name}" for name in modality.obs_names]

    refresh_global_obs(relabelled)

    assert relabelled.obs_names.tolist() == [f"renamed_{name}" for name in mdata.obs_names]
    assert_bookkeeping_is_final(relabelled, "refresh_global_obs")


# --- add_modality / replace_modality -----------------------------------------------------------------


def test_add_modality_returns_finished_bookkeeping(mdata: md.MuData) -> None:
    added = add_modality(copy_mudata(mdata), _new_modality(mdata, ["peptide_1", "peptide_2"]), "peptide")

    assert added.obsmap["peptide"].dtype == MODALITY_MAP_DTYPE
    assert "peptide" in added.obsm
    assert added.var.index.tolist()[-2:] == ["peptide_1", "peptide_2"]
    assert_bookkeeping_is_final(added, "add_modality")


def test_add_modality_pushes_the_global_obs_columns_onto_the_new_modality(mdata: md.MuData) -> None:
    container = copy_mudata(mdata)
    assert "group" in container.obs.columns

    added = add_modality(container, _new_modality(mdata, ["peptide_1"]), "peptide")

    assert added["peptide"].obs["group"].tolist() == added.obs["group"].tolist()


def test_replace_modality_returns_finished_bookkeeping(mdata: md.MuData) -> None:
    replaced = replace_modality(copy_mudata(mdata), "psm", mdata["psm"][:, :2].copy())

    assert replaced["psm"].n_vars == 2
    assert replaced.var.index.tolist() == replaced["psm"].var_names.tolist() + replaced["protein"].var_names.tolist()
    assert_bookkeeping_is_final(replaced, "replace_modality")


def test_replace_modality_pushes_the_global_obs_columns_onto_the_new_modality(mdata: md.MuData) -> None:
    container = copy_mudata(mdata)
    assert "group" in container.obs.columns

    replaced = replace_modality(container, "psm", _new_modality(mdata, ["new_1"]))

    assert replaced["psm"].obs["group"].tolist() == replaced.obs["group"].tolist()


def test_replace_modality_requires_an_existing_modality(mdata: md.MuData) -> None:
    with pytest.raises(KeyError, match="peptide"):
        replace_modality(copy_mudata(mdata), "peptide", _new_modality(mdata, ["peptide_1"]))


@pytest.mark.parametrize("insert", [add_modality, replace_modality])
def test_modality_insertion_rejects_observations_outside_the_container(mdata: md.MuData, insert) -> None:
    outside = ad.AnnData(
        X=np.zeros((1, 1)), obs=pd.DataFrame(index=["not_a_sample"]), var=pd.DataFrame(index=["feature"])
    )
    mod_name = "psm" if insert is replace_modality else "peptide"

    with pytest.raises(ValueError, match="not present in mdata.obs.index"):
        insert(copy_mudata(mdata), *((mod_name, outside) if insert is replace_modality else (outside, mod_name)))


@pytest.mark.parametrize("make_unrebuildable", ["global_var_column", "user_varm", "varp"])
def test_replace_modality_handles_renamed_and_filtered_features_where_mudata_refuses(
    mdata: md.MuData, make_unrebuildable: str
) -> None:
    container = copy_mudata(mdata)
    _make_unrebuildable(container, make_unrebuildable)
    new_psm = _new_modality(mdata, ["new_1", "new_2"])

    # mudata's own update cannot restore the order after a modality is renamed and filtered at once.
    refused = copy_mudata(container)
    refused.mod["psm"] = new_psm.copy()
    with pytest.raises(NotImplementedError):
        refused.update_var()

    replaced = replace_modality(container, "psm", new_psm)

    assert replaced["psm"].var_names.tolist() == ["new_1", "new_2"]
    assert set(replaced.var_names) == {"new_1", "new_2", *replaced["protein"].var_names}
    if make_unrebuildable == "global_var_column":
        assert replaced.var.columns.tolist() == ["pulled"]
    assert_bookkeeping_is_final(replaced, make_unrebuildable)


def test_replace_modality_handles_a_var_name_shared_with_another_modality(mdata: md.MuData) -> None:
    # The plain layout alone does not qualify for the direct rebuild: a shared name sends the rebuild to
    # update_var(), which would refuse the renamed and filtered modality after the swap had happened.
    container = copy_mudata(mdata)
    input_hash = compute_hash(container)
    shared_name = mdata["protein"].var_names[0]

    replaced = replace_modality(container, "psm", _new_modality(mdata, [shared_name, "new_1"]))

    assert replaced["psm"].var_names.tolist() == [shared_name, "new_1"]
    assert replaced.var_names.tolist().count(shared_name) == 2
    assert compute_hash(container) == input_hash  # the input is not left half rewritten
    assert_bookkeeping_is_final(replaced, "shared var name")


def test_replaced_modality_round_trips_through_h5mu(mdata: md.MuData, tmp_path) -> None:
    replaced = replace_modality(copy_mudata(mdata), "psm", mdata["psm"][:, :2].copy())

    path = tmp_path / "replaced.h5mu"
    replaced.write_h5mu(path)
    reread = md.read_h5mu(path)

    assert reread["psm"].shape == (4, 2)
    assert _global_bookkeeping(reread) == _global_bookkeeping(replaced)
