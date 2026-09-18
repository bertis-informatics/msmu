import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

import msmu as mm


@pytest.mark.parametrize("dtype", ["boolean", "category"])
def test_replace_rules_preserve_missing_and_replay(tmp_path, dtype):
    source = tmp_path / "input.h5mu"
    var = pd.DataFrame({"peptide_type": pd.Series([True, False, None], dtype=dtype)})
    var.index = ["p1", "p2", "p3"]
    md.MuData({"peptide": ad.AnnData(X=np.ones((1, 3)), var=var)}).write_h5mu(source)
    with mm.pv.options(hashing=True):
        m = mm.read_h5mu(source)
        result = mm.dt.replace(m, target="peptide.var",
                                     columns={"peptide_type": {True: "unique", False: "shared"}})
    assert result is m
    values = m["peptide"].var["peptide_type"].copy()
    assert values.iloc[:2].tolist() == ["unique", "shared"]
    assert pd.isna(values.iloc[2])
    saved = tmp_path / "result.h5mu"
    m.write_h5mu(saved)
    history = mm.pv.get_log(md.read_h5mu(saved))
    event = history["events"][-1]
    assert len(event["inputs"]) == 1
    assert set(event["parameters"]) == {"target", "columns"}
    pd.testing.assert_series_equal(mm.pv.replay(history)["peptide"].var["peptide_type"], values)
    namespace = {}
    exec(mm.pv.to_script(history), namespace)
    pd.testing.assert_series_equal(namespace["mdata"]["peptide"].var["peptide_type"], values)


def test_unmatched_values_and_failed_multi_column_assignment():
    m = md.MuData({"peptide": ad.AnnData(X=np.ones((1, 4)))})
    m["peptide"].var["kind"] = [True, False, "other", None]
    mm.dt.replace(m, target="peptide.var", columns={"kind": {True: "unique", False: "shared"}})
    assert m["peptide"].var["kind"].tolist() == ["unique", "shared", "other", None]
    before = mm.pv.compute_hash(m)
    with pytest.raises(KeyError):
        mm.dt.replace(m, target="peptide.var", columns={"kind": {"other": "new"}, "absent": {0: 1}})
    assert mm.pv.compute_hash(m) == before
