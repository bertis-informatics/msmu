import os
from pathlib import Path
import sys
import types

import mudata
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData


mudata.set_options(pull_on_update=False)
os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

if "msmu" not in sys.modules:
    msmu_pkg = types.ModuleType("msmu")
    msmu_pkg.__path__ = [str(PACKAGE_ROOT)]
    sys.modules["msmu"] = msmu_pkg


def _make_adata(x, obs, var, *, uns=None, obsm=None) -> AnnData:
    adata = AnnData(X=x, obs=obs, var=var)
    if uns:
        adata.uns.update(uns)
    if obsm:
        for key, value in obsm.items():
            adata.obsm[key] = value
    return adata


def _make_mdata(mods: dict[str, AnnData]) -> MuData:
    return MuData(mods)


@pytest.fixture
def obs_df() -> pd.DataFrame:
    obs = pd.DataFrame(
        {
            "sample": ["s1", "s2", "s3", "s4"],
            "group": pd.Categorical(["A", "A", "B", "B"], categories=["A", "B"]),
            "batch": pd.Categorical(["x", "y", "x", "y"], categories=["x", "y"]),
            "filename": ["f1", "f2", "f3", "f4"],
        },
        index=["s1", "s2", "s3", "s4"],
    )
    return obs


@pytest.fixture
def var_df() -> pd.DataFrame:
    var = pd.DataFrame(
        {
            "class": pd.Categorical(["x", "y", "x"], categories=["x", "y"]),
            "score": [10.0, 20.0, 30.0],
        },
        index=["v1", "v2", "v3"],
    )
    return var


@pytest.fixture
def psm_adata(obs_df: pd.DataFrame, var_df: pd.DataFrame) -> AnnData:
    x = np.array(
        [
            [1.0, 2.0, np.nan],
            [2.0, 3.0, 4.0],
            [np.nan, 1.0, 2.0],
            [3.0, np.nan, 5.0],
        ]
    )
    return _make_adata(x, obs_df.copy(), var_df.copy(), uns={"search_engine": "Diann"})


@pytest.fixture
def protein_adata(obs_df: pd.DataFrame, var_df: pd.DataFrame) -> AnnData:
    x = np.array(
        [
            [1.0, 1.5, np.nan],
            [2.0, 3.0, 4.5],
            [1.2, 1.0, 2.2],
            [3.1, 2.5, 5.0],
        ]
    )
    obsm = {
        "X_pca": pd.DataFrame(
            [[1.0, 0.2], [0.5, -0.1], [-0.3, 0.4], [0.1, -0.2]],
            index=obs_df.index,
            columns=["PC_1", "PC_2"],
        ),
        "X_umap": pd.DataFrame(
            [[-1.0, 2.0], [0.5, 1.5], [1.2, -0.5], [-0.8, -1.1]],
            index=obs_df.index,
            columns=["UMAP_1", "UMAP_2"],
        ),
    }
    return _make_adata(
        x, obs_df.copy(), var_df.copy(), uns={"pca": {"variance_ratio": np.array([0.6, 0.3])}}, obsm=obsm
    )


@pytest.fixture
def mdata(psm_adata: AnnData, protein_adata: AnnData) -> MuData:
    mdata = _make_mdata({"psm": psm_adata, "protein": protein_adata})
    mdata.uns["plotting"] = {"default_obs_column": "group"}
    mdata.obs["sample"] = psm_adata.obs["sample"].astype(str)
    mdata.obs["group"] = pd.Categorical(psm_adata.obs["group"], categories=["A", "B"])
    mdata.obs["batch"] = pd.Categorical(psm_adata.obs["batch"], categories=["x", "y"])
    mdata.obs["filename"] = psm_adata.obs["filename"].astype(str)
    return mdata


@pytest.fixture
def filter_mdata() -> MuData:
    obs = pd.DataFrame(index=["s1", "s2"])
    var = pd.DataFrame({"score": [10.0, 20.0, 30.0]}, index=["v1", "v2", "v3"])
    x = pd.DataFrame([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], index=obs.index, columns=var.index)
    adata = _make_adata(x, obs, var, uns={"decoy": pd.DataFrame({"score": [5.0, 25.0, 30.0]}, index=var.index)})
    return _make_mdata({"psm": adata})


@pytest.fixture
def simple_mdata() -> MuData:
    obs = pd.DataFrame(index=["s1", "s2", "gis1"])
    obs["gis"] = [False, False, True]
    var = pd.DataFrame(index=["v1", "v2"])
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    adata = _make_adata(x, obs, var)
    return _make_mdata({"psm": adata})


@pytest.fixture
def simple_adata() -> AnnData:
    obs = pd.DataFrame(index=["s1", "s2"])
    var = pd.DataFrame(
        {"peptide": ["p1", "p1", "p2"], "score": [0.1, 0.9, 0.2]},
        index=["f1", "f2", "f3"],
    )
    x = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    return _make_adata(x, obs, var)


@pytest.fixture
def ptm_mdata() -> MuData:
    obs = pd.DataFrame(index=["s1", "s2"])
    ptm_var = pd.DataFrame({"protein_group": ["P1", "P1"]}, index=["site1", "site2"])
    ptm_x = np.array([[1.0, 2.0], [3.0, 4.0]])
    ptm_adata = _make_adata(ptm_x, obs.copy(), ptm_var)
    return _make_mdata({"phospho_site": ptm_adata})


@pytest.fixture
def global_mdata() -> MuData:
    obs = pd.DataFrame(index=["s1", "s2"])
    global_var = pd.DataFrame(index=["P1"])
    global_x = np.array([[0.5], [1.5]])
    global_adata = _make_adata(global_x, obs.copy(), global_var)
    return _make_mdata({"protein": global_adata})


@pytest.fixture
def mdata_factory():
    def _factory(name: str) -> MuData:
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        obs = pd.DataFrame({"group": ["A", "B"]}, index=[f"{name}_s1", f"{name}_s2"])
        var = pd.DataFrame(index=["f1", "f2"])
        adata = _make_adata(x, obs, var, uns={"level": "psm"})
        return _make_mdata({"psm": adata})

    return _factory


@pytest.fixture
def labeled_mdata() -> MuData:
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    obs = pd.DataFrame(index=["s1", "s2", "gis1"])
    var = pd.DataFrame(index=["f1", "f2"])
    adata = _make_adata(x, obs, var, uns={"level": "psm", "label": "tmt"})
    return _make_mdata({"psm": adata})


@pytest.fixture
def psm_mdata_export() -> MuData:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    obs = pd.DataFrame(index=["s1", "s2"])
    var = pd.DataFrame(
        {
            "filename": ["f1.raw", "f2.raw"],
            "rt": [10.0, 20.0],
            "charge": [2, 3],
            "stripped_peptide": ["AA", "BB"],
            "peptide": ["AA", "BB"],
            "calcmass": [100.0, 200.0],
            "proteins": ["P1", "P2"],
            "extra": [1, 2],
        },
        index=["f1", "f2"],
    )
    adata = _make_adata(x, obs, var)
    return _make_mdata({"psm": adata})
