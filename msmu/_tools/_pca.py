import pandas as pd
from mudata import MuData
from sklearn.decomposition import PCA

from .._utils import uns_logger


@uns_logger
def pca(
    mdata: MuData,
    modality: str,
    n_comps: int = None,
    # zero_center: bool = True,
    svd_solver: str = "auto",
    random_state: int | None = 0,
) -> MuData:
    # Drop columns with NaN values
    data = mdata[modality].to_df().dropna(axis=1)

    # Calculate PCA
    pca = PCA(n_components=n_comps, svd_solver=svd_solver, random_state=random_state)
    pca.fit(data)

    # Save PCA results - dimensions
    dimensions = pca.transform(data)
    mdata[modality].obsm["X_pca"] = pd.DataFrame(
        dimensions, index=mdata[modality].obs_names, columns=[f"PC_{i + 1}" for i in range(dimensions.shape[1])]
    )

    # Save PCA results - loadings
    pcs = pd.DataFrame(pca.components_, columns=pca.feature_names_in_, index=pca.get_feature_names_out())
    mdata[modality].varm["PCs"] = pd.DataFrame(index=mdata[modality].var_names)
    mdata[modality].varm["PCs"] = mdata[modality].varm["PCs"].join(pcs.T)

    # Save PCA results - explained variance
    mdata[modality].uns["pca"] = {
        "variance": pca.explained_variance_,
        "variance_ratio": pca.explained_variance_ratio_,
    }

    # Save PCA results - number of components
    mdata[modality].uns["n_pca"] = pca.n_components_

    return mdata
