from typing import Literal
import mudata as md
import numpy as np
import pandas as pd
import logging
import statsmodels.api as sm
from inmoose.pycombat import pycombat_norm

from .._utils import uns_logger


logger = logging.getLogger(__name__)


@uns_logger
def correct_batch_effect(
    mdata: md.MuData,
    modality: str,
    method: Literal["gis", "median_center", "combat", "continuous"],
    category: str,
    layer: str | None = None,
    gis_samples: list[str] | None = None,
    drop_gis: bool = True,
    rescale: bool = True,
    log_transformed: bool = True,
) -> md.MuData:
    """
    Batch correction methods for MuData object.
    GIS-based normalization, median centering, ComBat, and continuous batch correction (with lowess) are supported.

    Parameters:
        mdata: MuData object to batch correct.
        method: Batch correction method to use. Options are 'gis', 'median_center', 'combat', 'continuous'.
        category: Category in .obs to use for batch correction.
        modality: Modality to batch correct.
        layer: Layer to batch correct. If None, the default layer (.X) will be used.
        gis_samples: List of GIS samples.
        drop_gis: If True, GIS samples will be dropped after correction. Default is True.
        rescale: If True, rescale the data after batch correction with median value across dataset (except combat).
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
        corrected_arr: np.ndarray = batch_corrector.gis(gis_samples=gis_samples)
    elif method == "median_center":
        corrected_arr: np.ndarray = batch_corrector.median_center()
    elif method == "combat":
        corrected_arr: np.ndarray = batch_corrector.combat()
        if rescale:
            logger.warning("Rescaling is not supported after ComBat correction.")
            logger.warning("Setting rescale to False.")
            rescale = False
    elif method == "continuous":
        corrected_arr: np.ndarray = batch_corrector.continuous()
    else:
        logger.error(
            f"Method {method}. not recognised. Please choose from 'gis', 'median_center', 'combat', 'continuous'"
        )
        raise ValueError(
            f"Method {method}. not recognised. Please choose from 'gis', 'median_center', 'combat', 'continuous'"
        )

    if rescale:
        logger.info("Rescaling data after batch correction.")
        corrected_arr: np.ndarray = batch_corrector.rescale()

    if layer is None:
        mdata.mod[modality].X = corrected_arr
    else:
        mdata.mod[modality].layers[layer] = corrected_arr

    if drop_gis and method == "gis" and gis_samples is not None:
        mdata = mdata[mdata.mod[modality].obs_names.difference(gis_samples), :].copy()

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

        self.original_arr = (
            self.mdata[self.modality].X if self.layer is None else self.mdata[self.modality].layers[self.layer]
        )
        self.corrected_arr: np.ndarray | None = None  # placeholder for corrected array

    def gis(self, gis_samples: list[str]):
        self.corrected_arr = self.original_arr.copy()
        batches, batch_idx, _ = self._make_batch_matrix()

        n_batches = len(batches)

        gis_idx = self._make_gis_index(gis_samples=gis_samples)

        gis_avg_arr = np.ndarray((n_batches, self.corrected_arr.shape[1]), dtype=float)
        for i in range(n_batches):
            gis_avg_arr[i, :] = np.nanmean(self.corrected_arr[gis_idx & (batch_idx == i), :], axis=0)
        correction_factor = gis_avg_arr[batch_idx, :]
        self.corrected_arr = self._correct(correction_factor=correction_factor)

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
        _, batch_idx, _ = self._make_batch_matrix()

        median_arr = pd.DataFrame(self.corrected_arr).groupby(batch_idx).median().values

        correction_factor = median_arr[batch_idx, :]
        self.corrected_arr = self._correct(correction_factor=correction_factor)

        return self.corrected_arr

    def combat(self):
        """
        ComBat batch correction using pycombat.
        https://epigenelabs.github.io/pyComBat/
        """
        _, batch_idx, _ = self._make_batch_matrix()
        sorted_idx = np.argsort(batch_idx)

        df = pd.DataFrame(
            self.original_arr,
            columns=self.mdata[self.modality].var_names,
            index=self.mdata[self.modality].obs_names,
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
        reference: Diagnostics and correction of batch effects in large‐scale proteomic studies: a tutorial
        https://pmc.ncbi.nlm.nih.gov/articles/PMC8447595/
        """
        self.corrected_arr = self.original_arr.copy()
        _, batch_idx, _ = self._make_batch_matrix()

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

        # res_lowess = res_lowess.replace(np.nan, 0)
        self.corrected_arr = self._correct(correction_factor=res_lowess)

        return self.corrected_arr

    def rescale(self):
        total_median = np.nanmedian(self.original_arr.flatten())
        self.corrected_arr += total_median

        return self.corrected_arr

    def _make_batch_matrix(self) -> np.ndarray:
        obs_category = self.mdata.obs[self.category]
        batches, batch_idx = np.unique(obs_category, return_inverse=True)
        n_batches = len(batches)

        batch_idx_arr = np.eye(n_batches)[batch_idx]

        return batches, batch_idx, batch_idx_arr

    def _correct(self, correction_factor: np.ndarray) -> np.ndarray:
        if self.log_transformed:
            self.corrected_arr -= correction_factor
        else:
            self.corrected_arr /= correction_factor

        return self.corrected_arr
