import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData

from msmu._read_write._reader_utils import _decompose_data, add_modality, merge_mudata, to_categorical


def _make_mdata(name: str) -> MuData:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    obs = pd.DataFrame({"group": ["A", "B"]}, index=[f"{name}_s1", f"{name}_s2"])
    var = pd.DataFrame(index=["f1", "f2"])
    adata = AnnData(x, obs=obs, var=var)
    adata.uns["level"] = "psm"
    return MuData({"psm": adata})


def test_decompose_data_requires_modality_for_anndata():
    adata = AnnData(np.array([[1.0]]))
    with pytest.raises(ValueError, match="mod should be specified"):
        _decompose_data(data=adata, name="x", parent_dict={}, modality=None)


def test_decompose_data_rejects_modality_for_mudata():
    mdata = _make_mdata("x")
    with pytest.raises(ValueError, match="mod should be None"):
        _decompose_data(data=mdata, name="x", parent_dict={}, modality="psm")


def test_merge_mudata_adds_dataset_column():
    mdata_a = _make_mdata("a")
    mdata_b = _make_mdata("b")
    merged = merge_mudata({"a": mdata_a, "b": mdata_b})
    assert "psm" in merged.mod_names
    assert "dataset" in merged.obs.columns
    assert set(merged.obs["dataset"].cat.categories) == {"a", "b"}


def test_add_modality_requires_parent_mods():
    mdata = _make_mdata("a")
    new_adata = AnnData(np.array([[1.0]]))
    with pytest.raises(ValueError, match="parent_mods should not be empty"):
        add_modality(mdata=mdata, adata=new_adata, mod_name="peptide", parent_mods=[])


def test_add_modality_inserts_modality():
    mdata = _make_mdata("a")
    new_adata = AnnData(np.array([[1.0]]), obs=pd.DataFrame(index=["a_s1"]), var=pd.DataFrame(index=["p1"]))
    out = add_modality(mdata=mdata, adata=new_adata, mod_name="peptide", parent_mods=["psm"])
    assert "peptide" in out.mod_names


def test_to_categorical_casts_object_columns():
    df = pd.DataFrame({"group": ["A", "B"]})
    out = to_categorical(df)
    assert out["group"].dtype.name == "category"
