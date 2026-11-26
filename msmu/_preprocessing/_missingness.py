from __future__ import annotations

import mudata as md
import numpy as np
import pandas as pd

from .._core._blockdiag import is_sparse
from .._core._provenance import log_provenance
from .._utils._anndata import _get_target_feature_mask
from .._utils._mudata import copy_mudata, get_anndata_mod
from ..logging_utils import get_logger

logger = get_logger(__name__)


def _count_observed_per_feature(matrix) -> np.ndarray:
    """Count the samples holding a value for each feature.

    In a sparse ``.X`` a cell is missing when it is not stored *or* when a NaN is stored, so both
    are excluded; a dense ``.X`` marks missing cells with NaN only.
    """
    if is_sparse(matrix):
        observed = matrix.copy()
        observed.data = (~np.isnan(observed.data)).astype(np.int64)
        return np.asarray(observed.sum(axis=0)).ravel()
    return (~np.isnan(np.asarray(matrix))).sum(axis=0)


def _observed_levels(groups: pd.Series) -> list:
    """Levels present in ``groups``, in category order when categorical; missing labels are dropped."""
    if isinstance(groups.dtype, pd.CategoricalDtype):
        return [level for level in groups.cat.categories if (groups == level).any()]
    return list(pd.unique(groups.dropna()))


@log_provenance
def calculate_missingness(
    mdata: md.MuData,
    modality: str,
    group_obs: str | None = None,
) -> md.MuData:
    """Count missing values per feature and record the counts in ``.var``.

    Writes ``.var["n_missing"]``, the number of samples without a value for the feature, and
    ``.var["missing_ratio"]``, that count divided by the number of samples. With ``group_obs``, an
    `.obs` column of the modality, ``.var["missing_ratio_<level>"]`` is added for every level present,
    so a filter can be set within one group. Existing columns are overwritten: recompute after
    removing samples, because [`filter`][msmu.pp.filter] does not update them.

    Decoy features (`.var["decoy"] == 1`) are counted from their own quantification, as the search
    engine reports it. An engine that does not quantify decoys (for example Sage label-free) leaves
    them missing in every sample, so a missingness filter removes them all (a warning says so); to
    keep them, filter with `include_decoy=False` or after the q-values are calculated.

    Parameters:
        mdata: `MuData` with the modality to annotate; a copy is returned.
        modality: Modality whose ``.X`` is counted along the sample axis.
        group_obs: Optional `.obs` column of the modality. One ``missing_ratio_<level>`` column is
            written per level present; samples with a missing label contribute only to the overall
            columns, with a warning.

    Returns:
        A copy of ``mdata`` with the columns added to ``.var``.

    Examples:
        ```python
        import msmu as mm
        mdata = mm.pp.calculate_missingness(mdata, modality="protein", group_obs="condition")
        mdata = mm.pp.filter(mdata, modality="protein", on="var", column="missing_ratio", keep="lt", value=0.5)
        ```
    """
    mdata = copy_mudata(mdata)
    adata = get_anndata_mod(mdata, modality)
    if adata.n_obs == 0:
        raise ValueError(f"{modality!r} has no observations to count missing values over")

    n_missing = adata.n_obs - _count_observed_per_feature(adata.X)
    columns: dict[str, np.ndarray] = {
        "n_missing": n_missing,
        "missing_ratio": n_missing / adata.n_obs,
    }

    if group_obs is not None:
        groups = adata.obs[group_obs]
        if groups.isna().any():
            logger.warning(
                "%s.obs[%r] has %d missing labels; those samples count only towards the overall columns.",
                modality,
                group_obs,
                int(groups.isna().sum()),
            )
        for level in _observed_levels(groups):
            # var columns become HDF5 dataset keys on write, where "/" is not allowed.
            if "/" in str(level):
                raise ValueError(f"group_obs level {level!r} contains '/', which cannot name a var column")
            in_group = np.flatnonzero((groups == level).to_numpy())
            n_observed_in_group = _count_observed_per_feature(adata.X[in_group])
            columns[f"missing_ratio_{level}"] = 1.0 - n_observed_in_group / len(in_group)

    for name, values in columns.items():
        adata.var[name] = values

    is_decoy_feature = ~_get_target_feature_mask(adata.var)
    if is_decoy_feature.any() and (n_missing[is_decoy_feature] == adata.n_obs).all():
        logger.warning(
            "%s: no decoy has a value in any sample, so a missingness filter would remove every decoy and "
            "leave none to calculate q-values from. Filter with include_decoy=False or after the q-values.",
            modality,
        )

    logger.info(
        "%s: %d of %d features have no value in any sample",
        modality,
        int((n_missing == adata.n_obs).sum()),
        adata.n_vars,
    )
    return mdata
