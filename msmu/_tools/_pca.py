"""
Module for performing Principal Component Analysis (PCA) on MuData objects, using scikit-learn's PCA implementation.
"""

import pandas as pd
from mudata import MuData
from sklearn.decomposition import PCA
from typing import Literal, Any

from .._utils import uns_logger
from .._utils.get import get_adata


@uns_logger
def pca(
    mdata: MuData,
    modality: str,
    layer: str | None = None,
    n_components: int | None = None,
    svd_solver: Literal["auto", "full", "arpack", "randomized"] = "auto",
    random_state: int | None = 0,
    **kwargs: Any,
) -> MuData:
    """
    Perform Principal Component Analysis (PCA) on the specified modality of the MuData object.

    - [Repository](https://github.com/scikit-learn/scikit-learn)
    - [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

    References:
        Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011).
        Scikit-learn: Machine learning in Python.
        Journal of machine learning research, 12(Oct), 2825-2830.

        Andrzej M., Waldemar R. (1993). Principal Component Analysis (PCA).
        Computers & Geosciences, 19(3), 303-342.

    Parameters:
        mdata: MuData object containing the data.
        modality: The modality to perform PCA on.
        layer: Layer to use for quantification aggregation. If None, the default layer (.X) will be used. Defaults to "scaled".
        n_components:
            Number of components to keep.
            if n_components is not set all components are kept::

                n_components == min(n_samples, n_features)

            If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
            MLE is used to guess the dimension. Use of ``n_components == 'mle'``
            will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.

            If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
            number of components such that the amount of variance that needs to be
            explained is greater than the percentage specified by n_components.

            If ``svd_solver == 'arpack'``, the number of components must be
            strictly less than the minimum of n_features and n_samples.

            Hence, the None case results in:

                n_components == min(n_samples, n_features) - 1
        svd_solver:

            "auto":
                The solver is selected by a default 'auto' policy is based on `X.shape` and
                `n_components`: if the input data has fewer than 1000 features and
                more than 10 times as many samples, then the "covariance_eigh"
                solver is used. Otherwise, if the input data is larger than 500x500
                and the number of components to extract is lower than 80% of the
                smallest dimension of the data, then the more efficient
                "randomized" method is selected. Otherwise the exact "full" SVD is
                computed and optionally truncated afterwards.

            "full" :
                Run exact full SVD calling the standard LAPACK solver via
                `scipy.linalg.svd` and select the components by postprocessing

            "arpack" :
                Run SVD truncated to `n_components` calling ARPACK solver via
                `scipy.sparse.linalg.svds`. It requires strictly
                `0 < n_components < min(X.shape)`

            "randomized" :
                Run randomized SVD by the method of Halko et al.
        random_state:
            Used when the 'arpack' or 'randomized' solvers are used. Pass an int
            for reproducible results across multiple function calls.
        **kwargs:
            Additional keyword arguments passed to PCA constructor.

    Returns:
        Updated MuData object with PCA results.
    """
    mdata = mdata.copy()

    # Drop columns with NaN values
    adata = get_adata(mdata, modality)
    if layer is not None:
        data = pd.DataFrame(data=adata.layers[layer], index=adata.obs_names, columns=adata.var_names)
    else:
        data = adata.to_df()
    data = data.dropna(axis=1)

    # Calculate PCA
    pca = PCA(n_components=n_components, svd_solver=svd_solver, random_state=random_state, **kwargs)
    pca.fit(data)

    # Save PCA results - dimensions
    dimensions = pca.transform(data)
    mdata[modality].obsm["X_pca"] = pd.DataFrame(
        dimensions,
        index=mdata[modality].obs_names,
        columns=[f"PC_{i + 1}" for i in range(dimensions.shape[1])],
    )

    # Save PCA results - loadings
    pcs = pd.DataFrame(pca.components_, columns=pca.feature_names_in_, index=pca.get_feature_names_out())
    pcs_df = pd.DataFrame(index=mdata[modality].var_names)
    pcs_df = pcs_df.join(pcs.T)
    mdata[modality].varm["PCs"] = pcs_df

    # Save PCA results - explained variance
    mdata[modality].uns["pca"] = {
        "variance": pca.explained_variance_,
        "variance_ratio": pca.explained_variance_ratio_,
    }

    # Save PCA results - number of components
    mdata[modality].uns["n_pca"] = pca.n_components_

    return mdata
