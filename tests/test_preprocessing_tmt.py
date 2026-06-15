import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData

import msmu as mm
import msmu._utils as msmu_utils


def _make_tmt_mdata() -> MuData:
    obs = pd.DataFrame(index=["126", "127"])
    var = pd.DataFrame(
        {"filename": ["runA.raw", "runB.raw", "runA.raw", "runB.raw"]},
        index=["psm1", "psm2", "psm3", "psm4"],
    )
    adata = AnnData(
        X=np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        obs=obs,
        var=var,
        uns={"level": "psm", "label": "tmt"},
    )
    return MuData({"psm": adata})


def test_split_tmt_is_preprocessing_api():
    assert hasattr(mm.pp, "split_tmt")
    assert "split_tmt" not in msmu_utils.__all__
    assert not hasattr(msmu_utils, "split_tmt")


def test_split_tmt_splits_channels_by_run_set():
    mdata = _make_tmt_mdata()

    out = mm.pp.split_tmt(mdata, {"runA": "set1", "runB": "set2"})

    assert list(out.mod["psm"].obs_names) == ["126_set1", "127_set1", "126_set2", "127_set2"]
    assert list(out.mod["psm"].var_names) == ["psm1", "psm2", "psm3", "psm4"]
    assert out.mod["psm"].uns["label"] == "tmt"
    assert out.mod["psm"].to_df().loc["126_set1", "psm1"] == 1.0
    assert out.mod["psm"].to_df().loc["126_set2", "psm2"] == 2.0
    assert np.isnan(out.mod["psm"].to_df().loc["126_set1", "psm2"])
