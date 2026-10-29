"""copy_mudata / realign_global_var must leave a MuData in exactly the state mudata's own copy() produces.

The reference in every test is mudata itself (``MuData.copy()`` or ``update_var()``), so these tests also
detect a change in mudata's bookkeeping layout: if mudata starts storing varmap or varm differently, the
comparisons here fail before any provenance hash does.
"""

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
from _mudata_bookkeeping import global_bookkeeping as _global_bookkeeping

import msmu as mm
from msmu._core._hashing import compute_hash
from msmu._utils._mudata import MODALITY_MAP_DTYPE, add_modality, copy_mudata, realign_global_var


def _replace_psm_with_first_two_features(mdata: md.MuData) -> None:
    """What apply_filter does: swap the modality in place and leave the global var stale."""
    psm_subset = mdata["psm"][:, :2].copy()
    mdata.mod["psm"] = psm_subset


def test_copy_mudata_matches_mudata_copy(mdata: md.MuData) -> None:
    reference = mdata.copy()

    copied = copy_mudata(mdata)

    assert copied is not mdata
    assert _global_bookkeeping(copied) == _global_bookkeeping(reference)
    assert copied.obs.equals(reference.obs)
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


def test_copy_mudata_heals_a_replaced_modality_on_the_copy_only(mdata: md.MuData) -> None:
    stale = copy_mudata(mdata)
    _replace_psm_with_first_two_features(stale)
    stale_var_names = stale.var.index.tolist()
    assert len(stale_var_names) == mdata.n_vars  # global var not yet realigned

    healed = copy_mudata(stale)

    assert _global_bookkeeping(healed) == _global_bookkeeping(stale.copy())
    assert compute_hash(healed) == compute_hash(stale.copy())
    assert healed.n_vars == 2 + mdata["protein"].n_vars
    # The caller's object is untouched: healing it would change the hash a branched input carries.
    assert stale.var.index.tolist() == stale_var_names


def test_realign_global_var_matches_mudata_update_var(mdata: md.MuData) -> None:
    realigned = copy_mudata(mdata)
    reference = copy_mudata(mdata)
    _replace_psm_with_first_two_features(realigned)
    _replace_psm_with_first_two_features(reference)

    realign_global_var(realigned)
    reference.update_var()

    assert _global_bookkeeping(realigned) == _global_bookkeeping(reference)
    assert compute_hash(realigned) == compute_hash(reference)
    # mudata's change cache is keyed by the current modalities after an update; ours must be too.
    assert set(realigned._varhash) == set(reference._varhash) == set(realigned.mod)


def test_realign_global_var_uses_mudata_layout(mdata: md.MuData) -> None:
    realigned = copy_mudata(mdata)
    _replace_psm_with_first_two_features(realigned)

    realign_global_var(realigned)

    assert realigned.var.index.tolist() == realigned["psm"].var_names.tolist() + realigned["protein"].var_names.tolist()
    assert realigned.var.shape[1] == 0
    assert realigned.var.columns.dtype == mdata.copy().var.columns.dtype
    for modality_name, modality in realigned.mod.items():
        varmap = realigned.varmap[modality_name]
        assert varmap.dtype == MODALITY_MAP_DTYPE
        assert varmap[varmap > 0].tolist() == list(range(1, modality.n_vars + 1))
        mask = realigned.varm[modality_name]
        assert mask.dtype == bool and mask.shape == (realigned.n_vars, 1)
        np.testing.assert_array_equal(mask.ravel(), varmap > 0)


@pytest.mark.parametrize("make_unrebuildable", ["global_var_column", "user_varm", "varp"])
def test_realign_global_var_falls_back_to_mudata_update_var(mdata: md.MuData, make_unrebuildable: str) -> None:
    realigned = copy_mudata(mdata)
    reference = copy_mudata(mdata)
    for container in (realigned, reference):
        if make_unrebuildable == "global_var_column":
            container.var["pulled"] = np.arange(container.n_vars)
        elif make_unrebuildable == "user_varm":
            container.varm["custom"] = np.zeros((container.n_vars, 2))
        else:
            container.varp["distance"] = np.zeros((container.n_vars, container.n_vars))
        _replace_psm_with_first_two_features(container)

    realign_global_var(realigned)
    reference.update_var()

    assert _global_bookkeeping(realigned) == _global_bookkeeping(reference)
    assert realigned.var.columns.tolist() == reference.var.columns.tolist()


def test_copy_mudata_after_add_modality_matches_mudata_copy(mdata: md.MuData) -> None:
    peptide_adata = ad.AnnData(
        X=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        obs=pd.DataFrame(index=mdata.obs_names),
        var=pd.DataFrame(index=["peptide_1", "peptide_2"]),
    )
    added = add_modality(copy_mudata(mdata), peptide_adata, "peptide")
    assert added.obsmap["peptide"].dtype != MODALITY_MAP_DTYPE  # add_modality leaves the int64 obsmap

    copied = copy_mudata(added)

    assert _global_bookkeeping(copied) == _global_bookkeeping(added.copy())
    assert compute_hash(copied) == compute_hash(added.copy())
    assert copied.obsmap["peptide"].dtype == MODALITY_MAP_DTYPE
    assert "peptide" in copied.obsm


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


def test_copy_mudata_result_round_trips_through_h5mu(mdata: md.MuData, tmp_path) -> None:
    copied = copy_mudata(mdata)
    _replace_psm_with_first_two_features(copied)
    healed = copy_mudata(copied)

    path = tmp_path / "copied.h5mu"
    healed.write_h5mu(path)
    reread = md.read_h5mu(path)

    assert reread["psm"].shape == (4, 2)
    assert _global_bookkeeping(reread) == _global_bookkeeping(healed)
