import pandas as pd
from mudata import MuData
from umap import UMAP

from .._utils import uns_logger


@uns_logger
def umap(
    mdata: MuData,
    modality: str,
    n_comps: int = 2,
    n_neighbors: int = None,
    metric: str = "euclidean",
    init: str = "random",
    min_dist: float = 0.1,
    random_state: int | None = 0,
) -> MuData:
    # Drop columns with NaN values
    data = mdata[modality].to_df().dropna(axis=1)

    # Set n_neighbors
    if n_neighbors is None:
        n_neighbors = data.shape[0] - 1

    # Calculate UMAP
    umap = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_comps,
        metric=metric,
        init=init,
        min_dist=min_dist,
        random_state=random_state,
    )
    umap.fit(data)

    # Save PCA results - dimensions
    dimensions = umap.transform(data)
    mdata[modality].obsm["X_umap"] = pd.DataFrame(
        dimensions, index=mdata[modality].obs_names, columns=[f"UMAP_{i + 1}" for i in range(dimensions.shape[1])]
    )

    # Save UMAP results - number of components
    mdata[modality].uns["n_umap"] = umap.n_components

    return mdata
