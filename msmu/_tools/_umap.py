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
    """Calculate UMAP embedding for a given modality in MuData object.
    
    Parameters:
        mdata (MuData): MuData object containing the data.
        modality (str): The modality to perform UMAP on.
        n_comps (int): Number of UMAP components to compute. Default is 2.
        n_neighbors (int): Number of neighbors to use. If None, set to number of samples - 1. Default is None.
        metric (str): Metric to use for UMAP. Default is 'euclidean'.
        init (str): Initialization method for UMAP. Default is 'random'.
        min_dist (float): Minimum distance parameter for UMAP. Default is 0.1.
        random_state (int | None): Random state for reproducibility. Default is 0.

    Returns:
        MuData: Updated MuData object with UMAP results.
    """
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
