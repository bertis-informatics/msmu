from __future__ import annotations

import mudata as md
import numpy as np
import pandas as pd

from .._core._blockdiag import is_sparse
from .._core._provenance import log_provenance
from .._utils._mudata import get_anndata_mod
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
    """Count missing values per feature and record the counts in ``.var`` (and on the decoys).

    Writes ``.var["n_missing"]``, the number of samples without a value for the feature, and
    ``.var["missing_ratio"]``, that count divided by the number of samples. With ``group_obs``, an
    `.obs` column of the modality, ``.var["missing_ratio_<level>"]`` is added for every level present,
    so a filter can be set within one group. Existing columns are overwritten: recompute after
    removing samples, because [`apply_filter`][msmu.pp.apply_filter] does not update them.

    Decoys have identifications but no quantification, so a missingness filter has nothing to
    judge them on. They receive the same columns with ``0`` / ``0.0``, which the usual
    "missing at most" filters (``le``/``lt``) keep; a filter in the other direction (``ge``/``gt``
    on missingness) would drop every decoy.

    Parameters:
        mdata: `MuData` with the modality to annotate; a copy is returned.
        modality: Modality whose ``.X`` is counted along the sample axis.
        group_obs: Optional `.obs` column of the modality. One ``missing_ratio_<level>`` column is
            written per level present; samples with a missing label contribute only to the overall
            columns, with a warning.

    Returns:
        A copy of ``mdata`` with the columns added to ``.var`` and to ``.uns["decoy"]`` when present.

    Examples:
        ```python
        import msmu as mm
        mdata = mm.pp.calculate_missingness(mdata, modality="protein", group_obs="condition")
        mdata = mm.pp.add_filter(mdata, modality="protein", column="missing_ratio", keep="lt", value=0.5)
        mdata = mm.pp.apply_filter(mdata, modality="protein")
        ```
    """
    mdata = mdata.copy()
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
    if "decoy" in adata.uns:
        # Decoys carry identifications, not quantification, so a missingness filter has nothing to
        # judge them on. Recording "nothing missing" lets the usual "at most k missing" / "ratio
        # below p" filters keep every decoy, which keeps the target/decoy counts behind q-values intact.
        for name in columns:
            adata.uns["decoy"][name] = 0 if name == "n_missing" else 0.0

    logger.info(
        "%s: %d of %d features have no value in any sample",
        modality,
        int((n_missing == adata.n_obs).sum()),
        adata.n_vars,
    )
    return mdata
