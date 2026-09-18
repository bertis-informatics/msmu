import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

import msmu as mm


@pytest.mark.parametrize("values", [
    "shared",
    ["group1", "group2"],
    np.array([1, 2], dtype="int32"),
    np.float32(1.5),
    [0.12345678901234567, float("nan")],
    pd.Index(["group1", "group2"]),
    pd.Series(["group2", "group1"], index=["p2", "p1"]),
    pd.Series([1, pd.NA], index=["p2", "p1"], dtype="Int64"),
    pd.Series(pd.Categorical(["unique", "shared"], categories=["shared", "unique"], ordered=True),
              index=["p2", "p1"]),
    pd.Series(["group1"], index=["p1"]),
])
@pytest.mark.parametrize("on", ["var", "obs"])
def test_set_column_matches_pandas_and_replays_after_save(tmp_path, values, on):
    source = tmp_path / "source.h5mu"
    md.MuData({"peptide": ad.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame(index=["p1", "p2"]),
        var=pd.DataFrame(index=["p1", "p2"]),
    )}).write_h5mu(source)
    with mm.pv.options(hashing=True):
        mdata = mm.read_h5mu(source)
        expected = getattr(mdata["peptide"], on).copy()
        expected["annotation"] = values
        result = mm.dt.assign(mdata, "annotation", values, modality="peptide", on=on)
    assert result is mdata
    pd.testing.assert_frame_equal(getattr(result["peptide"], on), expected)
    saved = tmp_path / "result.h5mu"
    result.write_h5mu(saved)
    history = mm.pv.get_log(md.read_h5mu(saved))
    replayed = mm.pv.replay(history)
    pd.testing.assert_series_equal(getattr(replayed["peptide"], on)["annotation"], expected["annotation"])
    namespace = {}
    exec(mm.pv.to_script(history), namespace)
    pd.testing.assert_series_equal(getattr(namespace["mdata"]["peptide"], on)["annotation"], expected["annotation"])


def test_set_column_errors_leave_data_unchanged():
    mdata = md.MuData({"peptide": ad.AnnData(X=np.ones((2, 2)))})
    before = mm.pv.compute_hash(mdata)
    for kwargs, error in [
        ({"values": [1], "on": "var"}, ValueError),
        ({"values": 1, "on": "uns"}, ValueError),
        ({"values": object(), "on": "var"}, TypeError),
    ]:
        with pytest.raises(error):
            mm.dt.assign(mdata, "annotation", modality="peptide", **kwargs)
        assert mm.pv.compute_hash(mdata) == before


def test_captured_values_are_complete_and_detached():
    values = np.arange(2000)
    mdata = md.MuData({"peptide": ad.AnnData(X=np.ones((1, len(values))))})
    mm.dt.assign(mdata, "annotation", values, modality="peptide")
    values[:] = -1
    captured = mm.pv.get_log(mdata)["events"][-1]["parameters"]["values"]
    assert captured["values"] == list(range(2000))
