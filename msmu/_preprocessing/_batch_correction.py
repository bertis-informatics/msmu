import warnings
from typing import Literal
import mudata as md
import numpy as np
import pandas as pd
import statsmodels.api as sm
from inmoose.pycombat import pycombat_norm

from .._utils._mudata import get_anndata_mod, get_mudata
from .._core._provenance import uns_logger
from .._core._blockdiag import dense_block, is_sparse
from ..logging_utils import get_logger

logger = get_logger(__name__)

# A matrix has cross-batch structure once at least one feature is observed in this many batches (a
# cross-batch reference then exists). If NO feature reaches it the matrix is fully block-diagonal
# (e.g. a per-plex-split PSM matrix) and the scale is not restored -- restoring a single-batch
# feature against its own level would merely reconstruct the original and undo the correction (see
# ``BatchCorrector._restore_mask``).
MIN_BATCHES_FOR_SCALE_RESTORE: int = 2


@uns_logger
def correct_batch_effect(
    mdata: md.MuData,
    modality: str,
    method: Literal["gis", "median_center", "combat", "continuous"],
    category: str,
    layer: str | None = None,
    gis_samples: list[str] | None = None,
    drop_gis: bool = True,
    log_transformed: bool = True,
) -> md.MuData:
    """
    Batch correction methods for MuData object.
    GIS-based normalization, median centering, ComBat, and continuous batch correction (with lowess) are supported.

    For 'gis', 'median_center', and 'continuous' the per-feature abundance scale is restored after
    correction (gis: the geometric mean of the per-batch GIS levels, i.e. classic IRS [Plubell et
    al. 2017]; median_center/continuous: the per-feature overall median). Restoration is decided for
    the modality as a whole: when the matrix has cross-batch structure (at least one feature observed
    in >=2 batches, as at the peptide/protein level) every observed feature is restored -- a
    single-batch feature to its own level, so it stays on the same abundance scale as the rest. When
    the matrix is fully block-diagonal (no feature spans >=2 batches, e.g. a per-plex-split PSM
    matrix) nothing is restored and the output stays reference-relative, ready to roll up. ComBat
    scales itself.

    Parameters:
        mdata: MuData object to batch correct.
        method: Batch correction method to use. Options are 'gis', 'median_center', 'combat', 'continuous'.
        category: Category in .obs to use for batch correction.
        modality: Modality to batch correct.
        layer: Layer to batch correct. If None, the default layer (.X) will be used.
        gis_samples: List of GIS samples.
        drop_gis: If True, GIS samples will be dropped after correction. Default is True.
        log_transformed: If True, data is assumed to be log-transformed. Default is True.

    Returns:
        Batch corrected MuData object.
    """
    mdata = mdata.copy()

    batch_corrector: BatchCorrector = BatchCorrector(
        mdata=mdata,
        modality=modality,
        layer=layer,
        category=category,
        log_transformed=log_transformed,
    )

    if method == "gis":
        if gis_samples is None:
            raise ValueError("gis_samples must be provided when method is 'gis'.")
        corrected_arr: np.ndarray = batch_corrector.gis(gis_samples=gis_samples)
    elif method == "median_center":
        corrected_arr: np.ndarray = batch_corrector.median_center()
    elif method == "combat":
        corrected_arr: np.ndarray = batch_corrector.combat()
    elif method == "continuous":
        corrected_arr: np.ndarray = batch_corrector.continuous()
    else:
        logger.error(
            f"Method {method}. not recognised. Please choose from 'gis', 'median_center', 'combat', 'continuous'"
        )
        raise ValueError(
            f"Method {method}. not recognised. Please choose from 'gis', 'median_center', 'combat', 'continuous'"
        )

    logger.debug(
        "Batch correction '%s' produced array with shape %s.",
        method,
        corrected_arr.shape,
    )

    adata = get_anndata_mod(mdata, modality)
    if layer is None:
        adata.X = corrected_arr
    else:
        adata.layers[layer] = corrected_arr

    if drop_gis and method == "gis":
        mdata = get_mudata(mdata[adata.obs_names.difference(gis_samples), :].copy())

    return mdata


class BatchCorrector:
    def __init__(
        self,
        mdata: md.MuData,
        modality: str,
        layer: str | None = None,
        category: str | None = None,
        log_transformed: bool = True,
    ):
        self.mdata = mdata
        self.modality = modality
        self.layer = layer
        self.category = category
        self.log_transformed = log_transformed
        self.adata = get_anndata_mod(self.mdata, self.modality)

        self.original_arr = self.adata.X if self.layer is None else self.adata.layers[self.layer]
        # Batch correction (median/ComBat/lowess) works on a dense array; densify a sparse
        # block-diagonal with NaN for absent cells (0-fill would corrupt the corrections).
        if is_sparse(self.original_arr):
            self.original_arr = dense_block(self.original_arr).astype(self.original_arr.dtype)
        self.corrected_arr: np.ndarray | None = None  # placeholder for corrected array

    def gis(self, gis_samples: list[str]):
        self.corrected_arr = self.original_arr.copy()
        batches, batch_idx = self._make_batch_matrix()
        logger.debug("Applying GIS batch correction across %d batches.", len(batches))

        n_batches = len(batches)

        gis_idx = self._make_gis_index(gis_samples=gis_samples)

        # Per-batch reference level, per feature (GIS channels within a batch are averaged). NaN for
        # a (batch, feature) with no observed GIS channel, so a batch missing the reference drops out.
        # The IRS reference target (Plubell et al. 2017) is the geometric mean of these per-batch
        # levels across batches (the arithmetic mean in log space); both come straight from the GIS
        # levels, so they are computed here before the correction mutates ``corrected_arr``.
        gis_avg_arr = np.full((n_batches, self.corrected_arr.shape[1]), np.nan, dtype=float)
        with warnings.catch_warnings():
            # An all-NaN slice (a batch with no observed GIS channel for a feature) is the expected
            # "no reference here" sentinel -> NaN; silence the empty-slice warning (common on a
            # block-diagonal matrix where a feature exists in only one batch).
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            for i in range(n_batches):
                gis_avg_arr[i, :] = np.nanmean(self.corrected_arr[gis_idx & (batch_idx == i), :], axis=0)
            if self.log_transformed:
                reference_target = np.nanmean(gis_avg_arr, axis=0)
            else:
                reference_target = np.exp(np.nanmean(np.log(gis_avg_arr), axis=0))
        reference_batch_count = np.sum(~np.isnan(gis_avg_arr), axis=0)
        n_missing_reference = int((reference_batch_count == 0).sum())
        if n_missing_reference:
            # A feature with no GIS channel in any batch has no reference to normalise against; the
            # correction below subtracts NaN and it becomes NaN (dropped). Reported here (at the
            # correction) rather than at the restore, where it would look like a scale choice.
            logger.warning(
                "%d/%d features have no GIS reference in any batch; they become NaN after correction.",
                n_missing_reference,
                reference_batch_count.size,
            )
        restore_mask = self._restore_mask(reference_batch_count)

        correction_factor = gis_avg_arr[batch_idx, :]
        self.corrected_arr = self._correct(correction_factor=correction_factor)

        # Restore the per-feature reference scale so the abundance axis is preserved instead of
        # collapsed to a reference ratio.
        self._restore_scale(reference_target=reference_target, restore_mask=restore_mask)

        return self.corrected_arr

    def _make_gis_index(self, gis_samples: list[str]) -> np.ndarray:
        obs = self.mdata.obs
        gis_idx = np.full((obs.shape[0],), False)

        for c in gis_samples:
            if c in obs.index:
                gis_idx = gis_idx | (obs.index == c)
            else:
                logger.error(f"{c} as GIS not found in obs.")
                raise ValueError(f"{c} as GIS not found in obs.")

        return gis_idx

    def median_center(self):
        self.corrected_arr = self.original_arr.copy()
        _, batch_idx = self._make_batch_matrix()
        logger.debug(
            "Applying median-centering batch correction across %d batches.",
            len(np.unique(batch_idx)),
        )

        median_arr = pd.DataFrame(self.corrected_arr).groupby(batch_idx).median().values

        correction_factor = median_arr[batch_idx, :]
        self.corrected_arr = self._correct(correction_factor=correction_factor)

        # Restore the per-feature overall median so the abundance axis is preserved instead of every
        # batch collapsing to a common centre (analogous to the gis IRS restore).
        reference_target = np.nanmedian(self.original_arr, axis=0)
        restore_mask = self._restore_mask(np.sum(~np.isnan(median_arr), axis=0))
        self._restore_scale(reference_target=reference_target, restore_mask=restore_mask)

        return self.corrected_arr

    def combat(self):
        """
        ComBat batch correction using pycombat.
        https://epigenelabs.github.io/pyComBat/
        """
        _, batch_idx = self._make_batch_matrix()
        logger.debug(
            "Applying ComBat batch correction across %d batches.",
            len(np.unique(batch_idx)),
        )
        sorted_idx = np.argsort(batch_idx)

        df = pd.DataFrame(
            self.original_arr,
            columns=self.adata.var_names,
            index=self.adata.obs_names,
        ).T

        df_sorted = df.iloc[:, sorted_idx]
        batch_idx_sorted = batch_idx[sorted_idx]

        df_corrected_sorted = pycombat_norm(counts=df_sorted, batch=batch_idx_sorted)

        rev_indices = np.argsort(sorted_idx)
        df_corrected = df_corrected_sorted.iloc[:, rev_indices]

        self.corrected_arr = df_corrected.T.values

        return self.corrected_arr

    def continuous(self):
        """
        Continuous batch correction using lowess.

        Treats ``category`` as an ORDERED continuous covariate (e.g. acquisition / run order): the
        per-feature trend against the batch index is fitted with lowess and removed. It is not meant
        for an unordered/nominal batch label. Note the batch index orders ``category`` by its
        sorted-unique values, so a numeric run order stored as strings sorts lexicographically.

        reference: Diagnostics and correction of batch effects in large-scale proteomic studies: a tutorial
        https://pmc.ncbi.nlm.nih.gov/articles/PMC8447595/
        """
        self.corrected_arr = self.original_arr.copy()
        batches, batch_idx = self._make_batch_matrix()
        n_batches = len(batches)
        logger.debug("Applying continuous batch correction across %d batches.", n_batches)

        res_lowess = np.full_like(self.corrected_arr, np.nan)
        for i in range(self.corrected_arr.shape[1]):
            y = self.corrected_arr[:, i]
            res = sm.nonparametric.lowess(
                endog=y,
                exog=batch_idx,
                xvals=batch_idx,
                missing="drop",
                frac=0.8,
                is_sorted=False,
                return_sorted=False,
            )
            res_lowess[:, i] = res

        self.corrected_arr = self._correct(correction_factor=res_lowess)

        # lowess removes each feature's level along with its batch-order trend; restore the
        # per-feature overall median (like median_center).
        reference_target = np.nanmedian(self.original_arr, axis=0)
        restore_mask = self._restore_mask(self._feature_batch_count(batch_idx, n_batches))
        self._restore_scale(reference_target=reference_target, restore_mask=restore_mask)

        return self.corrected_arr

    def _make_batch_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        obs_category = self.mdata.obs[self.category]
        batches, batch_idx = np.unique(obs_category, return_inverse=True)
        return batches, batch_idx

    def _feature_batch_count(self, batch_idx: np.ndarray, n_batches: int) -> np.ndarray:
        """Per feature, the number of batches with at least one observed (non-NaN) value."""
        batch_count = np.zeros(self.original_arr.shape[1], dtype=int)
        for batch in range(n_batches):
            batch_count += np.any(~np.isnan(self.original_arr[batch_idx == batch]), axis=0)
        return batch_count

    def _restore_mask(self, feature_batch_count: np.ndarray) -> np.ndarray:
        """Which features to restore, decided for the matrix as a whole (not per feature).

        When any feature is observed in >=``MIN_BATCHES_FOR_SCALE_RESTORE`` batches the matrix has
        cross-batch structure (e.g. peptide/protein level), so every observed feature is restored --
        a lone single-batch feature to its own level, keeping it on the same abundance scale as the
        rest rather than stranded at the reference-relative origin. When no feature spans that many
        batches the matrix is fully block-diagonal (e.g. a per-plex-split PSM matrix): restoring a
        single-batch feature against its own level would just reconstruct the original and undo the
        correction, so nothing is restored and the output stays reference-relative (ready to roll up).
        """
        if np.any(feature_batch_count >= MIN_BATCHES_FOR_SCALE_RESTORE):
            return feature_batch_count >= 1
        return np.zeros_like(feature_batch_count, dtype=bool)

    def _restore_scale(self, reference_target: np.ndarray, restore_mask: np.ndarray) -> None:
        """Add the per-feature reference target back (for the features in ``restore_mask``) so the
        abundance scale is preserved. Additive in log space, multiplicative otherwise. Which features
        are restored is decided by :meth:`_restore_mask`.
        """
        n_restored = int(restore_mask.sum())
        if n_restored == 0:
            logger.info(
                "Fully block-diagonal (no feature spans >=%d batches): abundance scale not restored, "
                "output stays reference-relative -- roll up before restoring the scale.",
                MIN_BATCHES_FOR_SCALE_RESTORE,
            )
            return
        logger.info("Restored the abundance scale for %d/%d features.", n_restored, restore_mask.size)
        if self.log_transformed:
            self.corrected_arr[:, restore_mask] += reference_target[restore_mask]
        else:
            self.corrected_arr[:, restore_mask] *= reference_target[restore_mask]

    def _correct(self, correction_factor: np.ndarray) -> np.ndarray:
        if self.log_transformed:
            self.corrected_arr -= correction_factor
        else:
            self.corrected_arr /= correction_factor

        return self.corrected_arr
