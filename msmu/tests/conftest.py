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
    adata = AnnData(X=x, obs=obs_df.copy(), var=var_df.copy())
    adata.uns["search_engine"] = "Diann"
    return adata


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
    adata = AnnData(X=x, obs=obs_df.copy(), var=var_df.copy())
    adata.uns["pca"] = {"variance_ratio": np.array([0.6, 0.3])}
    adata.obsm["X_pca"] = pd.DataFrame(
        [[1.0, 0.2], [0.5, -0.1], [-0.3, 0.4], [0.1, -0.2]],
        index=obs_df.index,
        columns=["PC_1", "PC_2"],
    )
    adata.obsm["X_umap"] = pd.DataFrame(
        [[-1.0, 2.0], [0.5, 1.5], [1.2, -0.5], [-0.8, -1.1]],
        index=obs_df.index,
        columns=["UMAP_1", "UMAP_2"],
    )
    return adata


@pytest.fixture
def mdata(psm_adata: AnnData, protein_adata: AnnData) -> MuData:
    mdata = MuData({"psm": psm_adata, "protein": protein_adata})
    mdata.uns["plotting"] = {"default_obs_column": "group"}
    mdata.obs["sample"] = psm_adata.obs["sample"].astype(str)
    mdata.obs["group"] = pd.Categorical(psm_adata.obs["group"], categories=["A", "B"])
    mdata.obs["batch"] = pd.Categorical(psm_adata.obs["batch"], categories=["x", "y"])
    mdata.obs["filename"] = psm_adata.obs["filename"].astype(str)
    return mdata
