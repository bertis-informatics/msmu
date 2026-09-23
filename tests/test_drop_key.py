from msmu._core._provenance import _event_inputs
import anndata as ad
import mudata as md
import numpy as np
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


@pytest.mark.parametrize("target,key,error", [
    ("psm.uns", "missing", KeyError),
    ("psm.var", "decoy", ValueError),
    ("uns", "_log", ValueError),
])
def test_drop_key_failure_leaves_data_unchanged(target, key, error):
    m = md.MuData({"psm": ad.AnnData(X=np.ones((1, 2)))})
    before = mm.pv.compute_hash(m)
    history = mm.pv.get_log(m)
    with pytest.raises(error):
        mm.dt.drop(m, target=target, key=key)
    assert mm.pv.compute_hash(m) == before
    assert mm.pv.get_log(m) == history
