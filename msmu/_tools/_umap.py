"""
Module for performing UMAP dimensionality reduction on MuData objects, based on umap-learn library.
"""

import numpy as np
import pandas as pd
from mudata import MuData
from umap import UMAP
from typing import Any

from .._utils import uns_logger
from .._utils.get import get_adata


@uns_logger
def umap(
    mdata: MuData,
    modality: str,
    layer: str | None = None,
    n_components: int = 2,
    n_neighbors: int | None = 15,
    metric: str = "euclidean",
    init: str = "random",
    min_dist: float = 0.1,
    random_state: int | None = None,
    **kwargs: Any,
) -> MuData:
    """Calculate UMAP embedding for a given modality in MuData object.

    - [Repository](https://github.com/lmcinnes/umap)
    - [Documentation](https://umap-learn.readthedocs.io/en/latest/)

    References:
        McInnes, L., Healy, J., & Melville, J. (2018).
        UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
        arXiv preprint arXiv:1802.03426.

    Parameters:
        mdata: MuData object containing the data.
        modality: The modality to perform UMAP on.
        n_components:
            The dimension of the space to embed into. This defaults to 2 to
            provide easy visualization, but can reasonably be set to any
            integer value in the range 2 to 100.
        n_neighbors:
            The size of local neighborhood (in terms of number of neighboring
            sample points) used for manifold approximation. Larger values
            result in more global views of the manifold, while smaller
            values result in more local data being preserved. In general
            values should be in the range 2 to 100.
        layer:
            Layer to use for quantification aggregation. If None, the default layer (.X) will be used. Defaults to None.
        metric:
            The metric to use to compute distances in high dimensional space.
            If a string is passed it must match a valid predefined metric. If
            a general metric is required a function that takes two 1d arrays and
            returns a float can be provided. For performance purposes it is
            required that this be a numba jit'd function. Valid string metrics
            include:

            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * ll_dirichlet
            * hellinger
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule

            Metrics that take arguments (such as minkowski, mahalanobis etc.)
            can have arguments passed via the metric_kwds dictionary. At this
            time care must be taken and dictionary elements must be ordered
            appropriately; this will hopefully be fixed in the future.
        init:
            How to initialize the low dimensional embedding. Options are:

            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * 'pca': use the first n_components from PCA applied to the
                input data.
            * 'tswspectral': use a spectral embedding of the fuzzy
                1-skeleton, using a truncated singular value decomposition to
                "warm" up the eigensolver. This is intended as an alternative
                to the 'spectral' method, if that takes an  excessively long
                time to complete initialization (or fails to complete).
            * A numpy array of initial embedding positions.
        min_dist:
            The effective minimum distance between embedded points. Smaller values
            will result in a more clustered/clumped embedding where nearby points
            on the manifold are drawn closer together, while larger values will
            result on a more even dispersal of points. The value should be set
            relative to the ``spread`` value, which determines the scale at which
            embedded points will be spread out.
        random_state:
            RandomState instance or None, optional (default: None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
        **kwargs:
            Additional keyword arguments passed to UMAP constructor.

    Returns:
        Updated MuData object with UMAP results.
    """
    mdata = mdata.copy()

    # Drop columns with NaN values
    adata = get_adata(mdata, modality)
    if layer is not None:
        data = pd.DataFrame(data=adata.layers[layer], index=adata.obs_names, columns=adata.var_names)
    else:
        data = adata.to_df()
    data = data.dropna(axis=1)

    # Set n_neighbors
    if n_neighbors is None:
        n_neighbors = data.shape[0] - 1

    # Calculate UMAP
    umap = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric=metric,
        init=init,
        min_dist=min_dist,
        random_state=random_state,
        **kwargs,
    )
    umap.fit(data)

    # Save PCA results - dimensions
    dimensions = np.asarray(umap.transform(data))
    mdata[modality].obsm["X_umap"] = pd.DataFrame(
        dimensions,
        index=mdata[modality].obs_names,
        columns=[f"UMAP_{i + 1}" for i in range(dimensions.shape[1])],
    )

    # Save UMAP results - number of components
    mdata[modality].uns["n_umap"] = umap.n_components

    return mdata
