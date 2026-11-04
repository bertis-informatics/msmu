"""
Module for performing principal component analysis (PCA) on MuData objects, using scikit-learn's PCA implementation.
"""

import pandas as pd
from mudata import MuData
from sklearn.decomposition import PCA
from typing import Literal, Any

from .._utils._mudata import copy_mudata, get_anndata_mod
from .._core._blockdiag import to_dense_df
from .._core._provenance import log_provenance


@log_provenance
def pca(
    mdata: MuData,
    modality: str,
    layer: str | None = None,
    n_components: int | None = None,
    svd_solver: Literal["auto", "full", "arpack", "randomized"] = "auto",
    random_state: int | None = 0,
    key_added: str = "X_pca",
    **kwargs: Any,
) -> MuData:
    """
    Perform principal component analysis (PCA) on the specified modality of the `MuData` object.

    This uses the `scikit-learn` implementation described by
    <a href="#ref-pedregosa-2011">Pedregosa et al. (2011)</a>. For background on PCA, see
    <a href="#ref-mackiewicz-1993">Maćkiewicz and Ratajczak (1993)</a>.

    - [Repository](https://github.com/scikit-learn/scikit-learn)
    - [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

    Parameters:
        mdata: `MuData` object containing the data.
        modality: The modality to perform PCA on.
        layer: Layer to use for quantification aggregation. If `None`, the default layer (`.X`) will be used. Defaults to `"scaled"`.
        n_components:
            Number of components to keep.
            if `n_components` is not set all components are kept::

                n_components == min(n_samples, n_features)

            If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
            MLE is used to guess the dimension. Use of ``n_components == 'mle'``
            will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.

            If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
            number of components such that the amount of variance that needs to be
            explained is greater than the percentage specified by `n_components`.

            If ``svd_solver == 'arpack'``, the number of components must be
            strictly less than the minimum of `n_features` and `n_samples`.

            Hence, the `None` case results in:

                n_components == min(n_samples, n_features) - 1
        svd_solver:

            `"auto"`:
                The solver is selected by a default `'auto'` policy is based on `X.shape` and
                `n_components`: if the input data has fewer than 1000 features and
                more than 10 times as many samples, then the `"covariance_eigh"`
                solver is used. Otherwise, if the input data is larger than 500x500
                and the number of components to extract is lower than 80% of the
                smallest dimension of the data, then the more efficient
                `"randomized"` method is selected. Otherwise the exact `"full"` SVD is
                computed and optionally truncated afterward.

            `"full"` :
                Run exact full SVD calling the standard LAPACK solver via
                `scipy.linalg.svd` and select the components by postprocessing

            `"arpack"` :
                Run SVD truncated to `n_components` calling ARPACK solver via
                `scipy.sparse.linalg.svds`. It requires strictly
                `0 < n_components < min(X.shape)`

            `"randomized"` :
                Run randomized SVD by the method of <a href="#ref-halko-2011">Halko et al. (2011)</a>.
        random_state:
            Used when the `'arpack'` or `'randomized'` solvers are used. Pass an `int`
            for reproducible results across multiple function calls.
        key_added:
            Base key used for PCA outputs. Results are stored in:
            - `.obsm[key_added]` for component scores
            - `.varm[key_added]` for loadings
            - `.uns[key_added]` for explained variance metadata
            Defaults to `"X_pca"`.
        **kwargs:
            Additional keyword arguments passed to the `PCA` constructor.

    Returns:
        Updated `MuData` object with PCA results.

    References:
        <a id="ref-pedregosa-2011"></a>

        > Pedregosa, F. et al. Scikit-learn: machine learning in Python.
        > *Journal of Machine Learning Research* 12, 2825–2830 (2011).
        > [Full text](https://www.jmlr.org/papers/v12/pedregosa11a.html).

        <a id="ref-mackiewicz-1993"></a>

        > Maćkiewicz, A. and Ratajczak, W. Principal components analysis (PCA).
        > *Computers & Geosciences* 19, 303–342 (1993).
        > [https://doi.org/10.1016/0098-3004(93)90090-r](https://doi.org/10.1016/0098-3004(93)90090-r).

        <a id="ref-halko-2011"></a>

        > Halko, N., Martinsson, P. G., and Tropp, J. A. Finding structure with randomness:
        > probabilistic algorithms for constructing approximate matrix decompositions.
        > *SIAM Review* 53, 217–288 (2011). [https://doi.org/10.1137/090771806](https://doi.org/10.1137/090771806).
    """
    mdata = copy_mudata(mdata)

    # Drop columns with NaN values
    adata = get_anndata_mod(mdata, modality)
    # to_dense_df restores absent cells as NaN for a sparse .X/layer (a plain DataFrame over
    # the raw sparse matrix crashes, and a densify would poison absent cells with 0).
    data = to_dense_df(adata, layer=layer)
    data = data.dropna(axis=1)

    # Calculate PCA
    pca = PCA(
        n_components=n_components,
        svd_solver=svd_solver,
        random_state=random_state,
        **kwargs,
    )
    pca.fit(data)

    # Save PCA results - dimensions
    dimensions = pca.transform(data)
    adata.obsm[key_added] = pd.DataFrame(
        dimensions,
        index=adata.obs_names,
        columns=pd.Index([f"PC_{i + 1}" for i in range(dimensions.shape[1])]),
    )

    # Save PCA results - loadings
    pcs = pd.DataFrame(
        pca.components_,
        columns=data.columns,
        index=pca.get_feature_names_out(),
    )
    pcs_df = pd.DataFrame(index=adata.var_names)
    pcs_df = pcs_df.join(pcs.T)
    adata.varm[key_added] = pcs_df

    # Save PCA results - explained variance and metadata
    adata.uns[key_added] = {
        "variance": pca.explained_variance_,
        "variance_ratio": pca.explained_variance_ratio_,
        "n_components": pca.n_components_,
    }

    return mdata
