from msmu._core._provenance import _event_inputs
import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

import msmu as mm


@pytest.mark.parametrize("target", ["uns", "psm.uns"])
def test_drop_key_saved_replay_and_script(tmp_path, target):
    m = md.MuData({"psm": ad.AnnData(X=np.ones((1, 2)))})
    container = m if target == "uns" else m["psm"]
    container.uns["decoy"] = {"values": np.arange(100)}
    container.uns["keep"] = "untouched"
    source = tmp_path / "source.h5mu"
    m.write_h5mu(source)
    with mm.pv.options(hashing=True):
        m = mm.read_h5mu(source)
        assert mm.dt.drop(m, target=target, key="decoy") is m
    container = m if target == "uns" else m["psm"]
    assert "decoy" not in container.uns
    assert container.uns["keep"] == "untouched"
    saved = tmp_path / "result.h5mu"
    m.write_h5mu(saved)
    history = mm.pv.get_log(md.read_h5mu(saved))
    event = history["events"][-1]
    assert {k: v for k, v in event["parameters"].items() if k != "mdata"} == {"target": target, "key": "decoy"}
    assert event["parameters"]["mdata"]["type"] == "MuData"
    assert len(_event_inputs(event)) == 1
    expected = mm.pv.compute_hash(m)
    assert mm.pv.compute_hash(mm.pv.replay(history)) == expected
    namespace = {}
    exec(mm.pv.to_script(history), namespace)
    assert mm.pv.compute_hash(namespace["mdata"]) == expected


@pytest.mark.parametrize("target", [
    "obs", "var", "psm.obs", "psm.var",
    "obsm.metrics", "varm.metrics", "psm.obsm.metrics", "psm.varm.metrics",
])
def test_drop_column_saved_replay_and_script(tmp_path, target):
    m = md.MuData({"psm": ad.AnnData(X=np.ones((2, 2)))})
    for table in (m.obs, m.var, m["psm"].obs, m["psm"].var):
        table["remove"] = [1, 2]
        table["keep"] = [3, 4]
    for owner in (m, m["psm"]):
        for axis in ("obsm", "varm"):
            index = owner.obs_names if axis == "obsm" else owner.var_names
            getattr(owner, axis)["metrics"] = pd.DataFrame({"remove": [1, 2], "keep": [3, 4]}, index=index)
    source = tmp_path / "source.h5mu"
    m.write_h5mu(source)
    with mm.pv.options(hashing=True):
        m = mm.read_h5mu(source)
        assert mm.dt.drop(m, target=target, key="remove") is m
    owner = m["psm"] if target.startswith("psm.") else m
    parts = target.split(".")
    table = (getattr(owner, parts[-2])[parts[-1]]
             if len(parts) > 1 and parts[-2] in ("obsm", "varm") else getattr(owner, parts[-1]))
    assert "remove" not in table
    assert table["keep"].tolist() == [3, 4]
    saved = tmp_path / "result.h5mu"
    m.write_h5mu(saved)
    history = mm.pv.get_log(md.read_h5mu(saved))
    event = history["events"][-1]
    assert {k: v for k, v in event["parameters"].items() if k != "mdata"} == {"target": target, "key": "remove"}
    assert len(_event_inputs(event)) == 1
    expected = mm.pv.compute_hash(m)
    assert mm.pv.compute_hash(mm.pv.replay(history)) == expected
    namespace = {}
    exec(mm.pv.to_script(history), namespace)
    assert mm.pv.compute_hash(namespace["mdata"]) == expected


@pytest.mark.parametrize("target", ["obsm", "varm", "psm.obsm", "psm.varm"])
def test_drop_mapping_entry_saved_replay(tmp_path, target):
    m = md.MuData({"psm": ad.AnnData(X=np.ones((2, 2)))})
    owner = m["psm"] if target.startswith("psm.") else m
    mapping = getattr(owner, target.split(".")[-1])
    mapping["remove"] = np.ones((2, 1))
    mapping["keep"] = np.full((2, 1), 2)
    source = tmp_path / "source.h5mu"
    m.write_h5mu(source)
    with mm.pv.options(hashing=True):
        m = mm.read_h5mu(source)
        assert mm.dt.drop(m, target=target, key="remove") is m
    owner = m["psm"] if target.startswith("psm.") else m
    mapping = getattr(owner, target.split(".")[-1])
    assert "remove" not in mapping
    assert mapping["keep"].tolist() == [[2], [2]]
    saved = tmp_path / "result.h5mu"
    m.write_h5mu(saved)
    history = mm.pv.get_log(md.read_h5mu(saved))
    expected = mm.pv.compute_hash(m)
    assert mm.pv.compute_hash(mm.pv.replay(history)) == expected
    namespace = {}
    exec(mm.pv.to_script(history), namespace)
    assert mm.pv.compute_hash(namespace["mdata"]) == expected


@pytest.mark.parametrize("target,key,error", [
    ("psm.uns", "missing", KeyError),
    ("psm.var", "missing", KeyError),
    ("psm.obsm", "missing", KeyError),
    ("psm.obsm.embedding", "missing", TypeError),
    ("psm.varm.search_result", "missing", KeyError),
    ("psm.obsp", "missing", ValueError),
    ("obsm", "psm", ValueError),
    ("varm", "psm", ValueError),
    ("uns", "_log", ValueError),
])
def test_drop_key_failure_leaves_data_unchanged(target, key, error):
    m = md.MuData({"psm": ad.AnnData(X=np.ones((1, 2)))})
    m["psm"].obsm["embedding"] = np.ones((1, 2))
    m["psm"].varm["search_result"] = pd.DataFrame({"keep": [1, 2]}, index=m["psm"].var_names)
    before = mm.pv.compute_hash(m)
    history = mm.pv.get_log(m)
    with pytest.raises(error):
        mm.dt.drop(m, target=target, key=key)
    assert mm.pv.compute_hash(m) == before
    assert mm.pv.get_log(m) == history
