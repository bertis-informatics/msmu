"""Concat replay from independent files, without tutorial data or filtering."""

from copy import deepcopy

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import msmu as mm


@pytest.mark.parametrize("sparse_input", [False, True])
def test_concat_independent_sources_and_chained_merge(tmp_path, sparse_input):
    def modality(sample, features, values, label):
        x = np.array([values], dtype=float)
        if sparse_input:
            x = sparse.csr_matrix(x)
        data = ad.AnnData(
            x,
            obs=pd.DataFrame(index=[sample]),
            var=pd.DataFrame({"label": label}, index=features),
        )
        data.layers["counts"] = x.copy()
        data.obsm["qc"] = pd.DataFrame({"score": [1.]}, index=data.obs_names)
        data.obsp["graph"] = sparse.eye(1, format="csr")
        return data

    sources = {
        "batch/z": md.MuData({
            "protein": modality("z", ["p1", "p2"], [2., 4.], "first"),
            "peptide": modality("z", ["pep1"], [5.], "first"),
        }),
        "batch'a": md.MuData({
            "protein": modality("a", ["p3", "p2"], [8., 16.], "second"),
        }),
        "third": md.MuData({
            "psm": modality("m", ["psm1"], [32.], "third"),
        }),
    }
    inputs = {}
    for index, (name, data) in enumerate(sources.items()):
        path = tmp_path / f"source-{index}.h5mu"
        data.write_h5mu(path)
        inputs[name] = mm.read_h5mu(path)
    before = {name: (mm.pv.compute_hash(data), deepcopy(mm.pv.get_log(data))) for name, data in inputs.items()}

    first = mm.dt.concat({name: inputs[name] for name in ("batch/z", "batch'a")})
    first_hash = mm.pv.compute_hash(first)
    merged = mm.dt.concat({"combined": first, "third": inputs["third"]})
    original = mm.dt.assign(merged, "checked", True, modality="protein")
    history = mm.pv.get_log(original)
    assert len(history["events"]) == 6  # Three readers, two concats, one assignment.
    concat_events = [event for event in history["events"] if event["function"] == "concat"]
    assert len(concat_events) == 2
    assert all(len(event["parents"]) == 2 for event in concat_events)
    assert concat_events[0]["id"] in concat_events[1]["parents"]
    assert list(concat_events[0]["parameters"]["mdatas"]) == ["batch/z", "batch'a"]

    script = mm.pv.to_script(original, verify=True)
    assert script.count(" = mm.dt.concat(") == 2
    namespace = {}
    exec(compile(script, "concat_workflow.py", "exec"), namespace)
    replayed = mm.pv.replay(original, verify=True)
    for result in (original, replayed, namespace["mdata"]):
        assert mm.pv.compute_hash(result) == mm.pv.compute_hash(original)
        assert list(result.mod) == ["protein", "peptide", "psm"]
        assert result.obs_names.tolist() == ["z", "a", "m"]
        assert result.obs["dataset"].tolist() == ["combined", "combined", "third"]
        protein = result["protein"]
        assert protein.obs_names.tolist() == ["z", "a"]
        fill = 0. if sparse_input else np.nan
        expected = np.array([[2., 4., fill], [fill, 16., 8.]])
        for layer in (None, "counts"):
            actual = protein.to_df(layer=layer).reindex(index=["z", "a"], columns=["p1", "p2", "p3"])
            np.testing.assert_allclose(actual.to_numpy(), expected, equal_nan=True)
        assert protein.var.loc["p2", "label"] == "first"
        assert protein.var["checked"].all()
        assert protein.obsm["qc"].index.equals(protein.obs_names)
        np.testing.assert_array_equal(protein.obsp["graph"].toarray(), np.eye(2))
        assert result["peptide"].obs_names.tolist() == ["z"]
        assert result["psm"].obs_names.tolist() == ["m"]
        assert len(mm.pv.get_log(result)["events"]) == 6
    assert mm.pv.compute_hash(first) == first_hash
    for name, data in inputs.items():
        assert (mm.pv.compute_hash(data), mm.pv.get_log(data)) == before[name]
