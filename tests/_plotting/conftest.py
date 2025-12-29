import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData

from msmu._plotting._pdata import PlotData


def _build_expression_matrix(n_obs: int, n_var: int, nan_fraction: float = 0.2) -> np.ndarray:
    """Create a reproducible matrix with one-decimal floats and a fixed fraction of NaNs."""
    base = np.mod(np.arange(1, n_obs * n_var + 1) * 0.7, 10.0)
    matrix = np.round(base.reshape(n_obs, n_var), 1)

    # deterministically sprinkle NaNs for roughly the requested fraction
    total_entries = n_obs * n_var
    n_nans = int(total_entries * nan_fraction)
    nan_positions = [
        divmod(idx * 7 % total_entries, n_var)  # pseudo-random but deterministic spread
        for idx in range(n_nans)
    ]
    for row, col in nan_positions:
        matrix[row, col] = np.nan

    return matrix


@pytest.fixture
def plotting_mudata() -> MuData:
    """MuData object enriched with metadata for exercising plotting utilities."""
    obs_index = [f"s{i:02d}" for i in range(1, 11)]
    condition_values = ["control"] * 5 + ["experimental"] * 5
    batch_values = ["b01", "b02", "b03", "b01", "b02", "b03", "b04", "b01", "b02", "b03"]
    obs = pd.DataFrame(
        {
            "sample": pd.Categorical(obs_index, categories=obs_index, ordered=True),
            "condition": pd.Categorical(condition_values, categories=["control", "experimental"]),
            "batch": pd.Categorical(batch_values, categories=sorted(set(batch_values))),
        },
        index=obs_index,
    )

    var_index = [f"p{i:02d}" for i in range(1, 13)]
    protein_classes = [
        "kinase",
        "kinase",
        "enzyme",
        "enzyme",
        "tf",
        "tf",
        "enzyme",
        "enzyme",
        "kinase",
        "kinase",
        "tf",
        "tf",
    ]
    lengths = [float(val) for val in range(110, 230, 10)]
    q_values = [round(val, 1) for val in np.linspace(0.1, 1.2, num=len(var_index))]
    var = pd.DataFrame(
        {
            "protein_class": pd.Categorical(protein_classes, categories=["kinase", "enzyme", "tf"]),
            "length": [round(val, 1) for val in lengths],
            "q_value": q_values,
        },
        index=var_index,
    )

    X = _build_expression_matrix(len(obs_index), len(var_index), nan_fraction=0.2)

    adata = AnnData(X=X, obs=obs.copy(), var=var.copy())
    loadings = np.round(
        np.linspace(0.1, 2.4, num=len(var_index) * 2).reshape(len(var_index), 2),
        1,
    )
    adata.varm["loadings"] = loadings

    pca_values = np.round(np.linspace(-4.5, 4.5, num=len(obs_index)), 1)
    pca_df = pd.DataFrame(
        {
            "PC_1": pca_values,
            "PC_2": np.round(pca_values[::-1], 1),
            "PC_3": np.round(np.linspace(2.0, -2.0, num=len(obs_index)), 1),
        },
        index=obs_index,
    )
    umap_df = pd.DataFrame(
        {
            "UMAP_1": np.round(np.linspace(-3.0, 3.0, num=len(obs_index)), 1),
            "UMAP_2": np.round(np.linspace(3.0, -3.0, num=len(obs_index)), 1),
        },
        index=obs_index,
    )
    adata.obsm["X_pca"] = pca_df
    adata.obsm["X_umap"] = umap_df
    adata.uns["pca"] = {"variance_ratio": np.array([0.5, 0.3, 0.2])}

    mdata = MuData({"protein": adata})
    mdata.obs = obs.copy()
    mdata.uns["plotting"] = {"default_obs_column": "sample"}

    return mdata


@pytest.fixture
def plot_data(plotting_mudata) -> PlotData:
    return PlotData(plotting_mudata, modality="protein")
