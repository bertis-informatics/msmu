import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData
from scipy import sparse
import mudata as md

from msmu._provenance import compute_hash

from msmu._read_write._reader_utils import (
    add_modality,
    concat,
    to_categorical,
)


def test_concat_rejects_invalid_inputs(mdata_factory):
    with pytest.raises(ValueError, match="At least two"):
        concat({})
    original = mdata_factory("a")
    before = original.copy()
    with pytest.raises(ValueError, match="At least two"):
        concat({"a": original})
    assert "dataset" not in original.obs
    assert original.uns == before.uns
    with pytest.raises(TypeError, match="Expected MuData"):
        concat({"invalid": AnnData(np.array([[1.0]])), "a": original})


def test_concat_adds_dataset_column(mdata_factory):
    mdata_a = mdata_factory("a")
    mdata_b = mdata_factory("b")
    merged = concat({"a": mdata_a, "b": mdata_b})
    assert "psm" in merged.mod
    assert "dataset" in merged.obs.columns
    assert set(merged.obs["dataset"].cat.categories) == {"a", "b"}


def test_concat_uses_first_metadata_and_preserves_native_arrays():
    inputs = {}
    for name, features, values in [("a", ["v2", "v1"], [2.0, 1.0]), ("b", ["v3", "v2"], [3.0, 20.0])]:
        adata = AnnData(np.array([values]), obs=pd.DataFrame(index=[name]), var=pd.DataFrame(index=features))
        adata.var["value"] = values
        adata.varm["table"] = pd.DataFrame({"value": values}, index=features)
        adata.varm["array"] = np.array(values)[:, None]
        adata.obsp["graph"] = sparse.csr_matrix([[1.0]])
        adata.layers["counts"] = adata.X.copy()
        adata.uns.update(common="same", different=name, items=[name], table=pd.DataFrame({"v": [1]}))
        inputs[name] = MuData({"protein": adata})
    merged = concat(inputs)["protein"]
    expected = pd.Series({"v1": 1.0, "v2": 2.0, "v3": np.nan}, name="value")
    pd.testing.assert_series_equal(merged.var["value"], expected.reindex(merged.var_names))
    pd.testing.assert_series_equal(merged.varm["table"]["value"], expected.reindex(merged.var_names))
    np.testing.assert_array_equal(merged.varm["array"][merged.var_names.get_indexer(["v2", "v1"]), 0], [2.0, 1.0])
    assert np.isnan(merged.varm["array"][merged.var_names.get_loc("v3"), 0])
    np.testing.assert_array_equal(merged.layers["counts"], merged.X)
    np.testing.assert_array_equal(merged.obsp["graph"].toarray(), np.eye(2))
    assert merged.uns["common"] == "same"
    assert merged.uns["different"] == "a"
    assert merged.uns["items"] == ["a"]
    assert merged.uns["table"].shape == (1, 1)


def test_concat_preserves_sample_alignment_in_obsm():
    inputs = {}
    for group, samples, values in [("G1", ["s1", "s3"], [1.0, 3.0]), ("G2", ["s2", "s4"], [2.0, 4.0])]:
        adata = AnnData(np.array(values)[:, None], obs=pd.DataFrame(index=samples))
        adata.obsm["filter"] = pd.DataFrame({group: [True, False]}, index=samples)
        adata.obsm["embedding"] = np.array(values)[:, None]
        if group == "G1":
            adata.obsm["partial"] = pd.DataFrame({"value": values}, index=samples)
        inputs[group] = MuData({"protein": adata})
        inputs[group].obs["label"] = pd.Categorical(pd.Series(["a", "b"], dtype="string"))
        adata.obs["label"] = pd.Categorical(pd.Index(["a", "b"], dtype=object))

    merged = concat(inputs)["protein"]

    assert merged.obs_names.tolist() == ["s1", "s3", "s2", "s4"]
    np.testing.assert_array_equal(merged.X[:, 0], [1.0, 3.0, 2.0, 4.0])
    np.testing.assert_array_equal(merged.obsm["embedding"], merged.X)
    assert merged.obs["label"].tolist() == ["a", "b", "a", "b"]
    for group, original in inputs.items():
        samples = original["protein"].obs_names
        assert merged.obsm["filter"].loc[samples, group].tolist() == [True, False]
        assert merged.obsm["filter"].drop(index=samples)[group].isna().all()
        assert original["protein"].obsm["filter"].columns.tolist() == [group]
    assert merged.obsm["partial"].loc[["s2", "s4"]].isna().all().all()
    assert merged.obsm["partial"].loc[["s1", "s3"], "value"].tolist() == [1.0, 3.0]


def test_concat_result_does_not_share_mutable_data(mdata_factory):
    inputs = {name: mdata_factory(name) for name in ("a", "b")}
    first = inputs["a"]["psm"]
    first.uns["metadata"] = {"values": [1]}
    first.varm["annotation"] = pd.DataFrame({"value": [1, 2]}, index=first.var_names)
    original_x = first.X.copy()

    merged = concat(inputs)["psm"]
    merged.X[0, 0] = -99
    merged.varm["annotation"].iloc[0, 0] = -99
    merged.uns["metadata"]["values"].append(2)

    np.testing.assert_array_equal(first.X, original_x)
    assert first.varm["annotation"].iloc[0, 0] == 1
    assert first.uns["metadata"] == {"values": [1]}


@pytest.mark.parametrize("shared", [True, False])
def test_concat_preserves_modality_union_order_and_global_annotations(shared):
    def modality(sample, feature):
        return AnnData(np.array([[2.0]]), obs=pd.DataFrame(index=[sample]), var=pd.DataFrame(index=[feature]))

    left = MuData({"psm": modality("s1", "p1"), "protein": modality("s1", "r1")})
    right_mods = {"peptide": modality("s2", "e1")}
    if shared:
        right_mods["psm"] = modality("s2", "p2")
    right = MuData(right_mods)
    for data in (left, right):
        data.obsm["embedding"] = np.array([[5.0]])
        data.obsp["graph"] = sparse.eye(1, format="csr")
    before = [compute_hash(data) for data in (left, right)]
    merged = concat({"a": left, "b": right})
    assert list(merged.mod) == ["psm", "protein", "peptide"]
    assert merged["peptide"].obs_names.tolist() == ["s2"]
    assert merged["peptide"].X[0, 0] == 2
    assert merged["psm"].obs_names.tolist() == (["s1", "s2"] if shared else ["s1"])
    assert set(merged.var_names) == ({"p1", "p2", "r1", "e1"} if shared else {"p1", "r1", "e1"})
    for mod in merged.mod:
        mask = np.asarray(merged.varm[mod]).ravel()
        assert merged.var_names[mask].tolist() == merged[mod].var_names.tolist()
    np.testing.assert_array_equal(merged.obsm["embedding"], [[5.0], [5.0]])
    np.testing.assert_array_equal(merged.obsp["graph"].toarray(), np.eye(2))
    assert [compute_hash(data) for data in (left, right)] == before


def test_concat_failure_does_not_remove_input_masks(mdata_factory, monkeypatch):
    inputs = {name: mdata_factory(name) for name in ("a", "b")}
    before = {name: compute_hash(data) for name, data in inputs.items()}

    def fail_after_removing_masks(mdatas, **kwargs):
        for data in mdatas.values():
            del data.obsm["psm"]
            del data.varm["psm"]
        raise RuntimeError("concat failed")

    monkeypatch.setattr(md, "concat", fail_after_removing_masks)
    with pytest.raises(RuntimeError, match="concat failed"):
        concat(inputs)
    assert {name: compute_hash(data) for name, data in inputs.items()} == before


def test_add_modality_requires_parent_mods(mdata_factory):
    mdata = mdata_factory("a")
    new_adata = AnnData(np.array([[1.0]]))
    with pytest.raises(ValueError, match="parent_mods should not be empty"):
        add_modality(mdata=mdata, adata=new_adata, mod_name="peptide", parent_mods=[])


def test_add_modality_inserts_modality(mdata_factory):
    mdata = mdata_factory("a")
    new_adata = AnnData(
        np.array([[1.0]]),
        obs=pd.DataFrame(index=["a_s1"]),
        var=pd.DataFrame(index=["p1"]),
    )
    out = add_modality(mdata=mdata, adata=new_adata, mod_name="peptide", parent_mods=["psm"])
    assert "peptide" in out.mod


def test_to_categorical_casts_string_columns_and_preserves_missing_values():
    df = pd.DataFrame(
        {
            "object": pd.Series(["A", None], dtype=object),
            "string": pd.Series(["A", pd.NA], dtype="string"),
        }
    )
    out = to_categorical(df)
    assert out["object"].dtype.name == "category"
    assert out["string"].dtype.name == "category"
    assert out.iloc[1].isna().all()


def test_deprecated_merge_mudata_records_concat(mdata_factory):
    import msmu as mm

    with pytest.warns(DeprecationWarning, match="use msmu.concat"):
        result = mm.merge_mudata({name: mdata_factory(name) for name in ("a", "b")})
    events = mm.pv.get_log(result)["events"]
    assert len(events) == 1
    assert events[0]["function"] == "concat"
