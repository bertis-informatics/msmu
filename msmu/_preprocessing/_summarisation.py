import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..logging_utils import get_logger
from .._core._blockdiag import aggregate_features_by_group, dense_block, is_sparse, to_dense_df
from ._filter import _mask_boolean_filter

# for type checking only
import anndata as ad
from typing import Literal


logger = get_logger(__name__)


@dataclass
class SparseQuant:
    """Carrier for a sparse block-diagonal quantification matrix through the summarisation path.

    Wraps a SciPy sparse ``(samples, features)`` matrix together with its sample names so the
    aggregator can reduce features per group without ever densifying the full matrix (see
    :mod:`msmu._core._blockdiag`). Exposes ``shape``/``copy`` so it can stand in for the dense
    quantification DataFrame in the existing flow.
    """

    matrix: sp.spmatrix  # (n_samples, n_features)
    sample_names: list[str]

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    def copy(self) -> "SparseQuant":
        return SparseQuant(matrix=self.matrix.copy(), sample_names=list(self.sample_names))

MEDIAN_POLISH_MAX_ITERATIONS: int = 10
# Relative convergence on the sum of absolute residuals, matching R's ``stats::medpolish``
# (the summarisation used by MSstats). Median polish does not always converge to a unique
# fixed point on data with missing values, so this stops at the standard residual criterion
# rather than iterating to machine precision.
MEDIAN_POLISH_CONVERGENCE_TOLERANCE: float = 1e-4

# directlfq's within-protein normalisation caps how many samples it uses to build the
# pairwise-shift graph (directlfq's ``number_of_quadratic_samples`` default) and requires at
# least this many observed ions to emit a protein estimate.
DIRECTLFQ_NUM_SAMPLES_QUADRATIC: int = 10
DIRECTLFQ_MIN_NON_MISSING_IONS: int = 1

# directlfq's low-level per-protein worker (unlike its top-level ``run_lfq``) never runs the
# copy-flag probe itself. On pandas>=3 / numpy>=2, ``DataFrame.to_numpy()`` returns a read-only
# (copy-on-write) array that directlfq's in-place sample shift cannot mutate, so the flag has to
# be set before the first call. Guarded so the one-off configuration runs at most once per process.
_directlfq_runtime_configured: bool = False


def _directlfq_rollup(feature_by_sample_matrix: np.ndarray) -> np.ndarray:
    """Summarise a feature-by-sample matrix to per-sample estimates via directlfq (DirectLFQ).

    DirectLFQ (Ammar et al. 2023) aligns each feature's intensity trace onto a common scale
    (a within-group "peptide shift") and takes the per-sample median of the aligned traces.
    It is a fast, MaxLFQ-inspired label-free quantification method, but a *distinct* algorithm
    from the classical MaxLFQ pairwise-ratio least-squares — the results are correlated, not
    identical.

    This calls directlfq's per-protein worker on a single group's submatrix, so the estimate for
    each group depends only on that group's own features (directlfq's cross-group step is its
    between-sample normalisation, which is skipped here — msmu handles normalisation upstream).
    That keeps this rollup a drop-in per-group aggregation, symmetric with ``_median_polish``.

    IMPORTANT: like median polish this operates in log space. directlfq's per-protein worker
    consumes log2 intensities and returns a log2-space profile, so the input must be log2
    (e.g. apply log2_transform first) and the output is log2.

    NaN handling: missing values propagate as directlfq's own missingness. Features (rows) with
    no observed values are dropped before the call; a sample column with no observed values
    across every feature yields ``NaN``.

    Args:
        feature_by_sample_matrix (np.ndarray): 2D array of log-space intensities with
            shape ``(n_features, n_samples)``. May contain NaN for missing values.

    Returns:
        np.ndarray: 1D array of length ``n_samples`` with the per-sample rollup estimate.
    """
    global _directlfq_runtime_configured

    import directlfq.config as directlfq_config
    import directlfq.protein_intensity_estimation as directlfq_estimation

    if not _directlfq_runtime_configured:
        directlfq_config.check_wether_to_copy_numpy_arrays_derived_from_pandas()
        # The worker is called once per group with idx=0, and directlfq logs an INFO line
        # whenever idx % 100 == 0 (always true at idx=0). Left on, that emits one identical
        # "lfq-object 0" line per protein group — thousands of lines on a real run.
        directlfq_config.set_log_processed_proteins(log_processed_proteins=False)
        _directlfq_runtime_configured = True

    matrix = np.asarray(feature_by_sample_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("directlfq rollup expects a 2D feature-by-sample matrix.")

    _number_of_features, number_of_samples = matrix.shape
    sample_estimates = np.full(number_of_samples, np.nan, dtype=float)

    # Drop features with no observed values; they contribute nothing and match directlfq's own
    # all-NaN-row removal. A fully-missing sample column is preserved and returns NaN.
    observed_feature_mask = ~np.all(np.isnan(matrix), axis=1)
    if not observed_feature_mask.any():
        return sample_estimates

    observed_matrix = matrix[observed_feature_mask]
    peptide_by_sample_df = pd.DataFrame(
        observed_matrix,
        index=pd.MultiIndex.from_arrays(
            [
                np.zeros(observed_matrix.shape[0], dtype=int),  # single protein group
                np.arange(observed_matrix.shape[0]),  # ion (feature) identifiers
            ],
            names=[directlfq_config.PROTEIN_ID, directlfq_config.QUANT_ID],
        ),
    )

    protein_profile, _shifted_peptides = directlfq_estimation.calculate_peptide_and_protein_intensities(
        0,
        peptide_by_sample_df,
        DIRECTLFQ_NUM_SAMPLES_QUADRATIC,
        DIRECTLFQ_MIN_NON_MISSING_IONS,
    )
    if protein_profile is None:
        return sample_estimates

    return np.asarray(protein_profile, dtype=float)


def _median_polish(feature_by_sample_matrix: np.ndarray) -> np.ndarray:
    """Summarise a feature-by-sample matrix to per-sample estimates via Tukey's median polish.

    Median polish fits the additive model ``value[feature, sample] = overall +
    row_effect[feature] + col_effect[sample] + residual`` by iteratively sweeping row
    and column medians out of the matrix. The rollup estimate for each sample is
    ``overall + col_effect[sample]``.

    Convergence follows R's ``stats::medpolish`` (MSstats' summarisation): iterate until the
    relative change in the sum of absolute residuals falls below
    ``MEDIAN_POLISH_CONVERGENCE_TOLERANCE`` or ``MEDIAN_POLISH_MAX_ITERATIONS`` is reached.
    Convergence on ``overall`` alone is NOT sufficient — it can stabilise while the row/column
    effects are still moving, stopping early with a wrong estimate.

    IMPORTANT: this is an *additive* model, so it must be applied to log-space
    intensities (e.g. log2). Applying it to linear intensities is not meaningful.

    NaN handling: fully-missing features (all-NaN rows) are dropped before polishing --
    they carry no information, and keeping them would bias the row-effect alignment median
    that recovers the overall protein level toward zero (collapsing groups whose features
    are mostly all-NaN, e.g. proteins quantified by a few unique peptides among many masked
    shared ones). Remaining missing values are ignored via ``nanmedian``. A sample column
    with no observed values across every feature yields ``NaN`` (there is nothing to summarise).

    Args:
        feature_by_sample_matrix (np.ndarray): 2D array of log-space intensities with
            shape ``(n_features, n_samples)``. May contain NaN for missing values.

    Returns:
        np.ndarray: 1D array of length ``n_samples`` with the per-sample rollup estimate.
    """
    input_matrix = np.asarray(feature_by_sample_matrix, dtype=float)

    if input_matrix.ndim != 2:
        raise ValueError("median polish expects a 2D feature-by-sample matrix.")

    number_of_samples = input_matrix.shape[1]
    fully_missing_sample_mask = np.all(np.isnan(input_matrix), axis=0)

    # Drop fully-missing features before polishing (symmetric with ``_directlfq_rollup``). An
    # all-NaN row carries no information, yet ``nanmedian`` still yields a placeholder row effect
    # that, folded back through the row-effect alignment median, drags ``overall`` -- and therefore
    # every sample estimate -- toward zero. When such rows are the majority of a protein group
    # (e.g. a few unique peptides among many masked shared ones) they collapse the whole protein
    # to ~0. Boolean indexing returns a fresh writable copy, so the sweeps mutate in place safely.
    observed_feature_mask = ~np.all(np.isnan(input_matrix), axis=1)
    if not observed_feature_mask.any():
        return np.full(number_of_samples, np.nan, dtype=float)
    residual_matrix = input_matrix[observed_feature_mask]

    number_of_features = residual_matrix.shape[0]

    overall_effect: float = 0.0
    row_effects = np.zeros(number_of_features, dtype=float)
    col_effects = np.zeros(number_of_samples, dtype=float)
    previous_residual_sum: float = np.inf

    with warnings.catch_warnings():
        # nanmedian legitimately hits all-NaN columns for fully-missing samples (all-NaN rows
        # were dropped above), so silence the resulting empty-slice warning.
        warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
        for iteration in range(MEDIAN_POLISH_MAX_ITERATIONS):
            # Row sweep: remove the median of each feature (row) across samples.
            row_medians = np.nan_to_num(np.nanmedian(residual_matrix, axis=1))
            residual_matrix -= row_medians[:, np.newaxis]
            row_effects += row_medians
            col_alignment = np.nan_to_num(np.nanmedian(col_effects))
            col_effects -= col_alignment
            overall_effect += col_alignment

            # Column sweep: remove the median of each sample (column) across features.
            col_medians = np.nan_to_num(np.nanmedian(residual_matrix, axis=0))
            residual_matrix -= col_medians[np.newaxis, :]
            col_effects += col_medians
            row_alignment = np.nan_to_num(np.nanmedian(row_effects))
            row_effects -= row_alignment
            overall_effect += row_alignment

            # Convergence on the sum of absolute residuals (R medpolish criterion).
            current_residual_sum = float(np.nansum(np.abs(residual_matrix)))
            if current_residual_sum == 0.0:
                break
            if iteration > 0 and abs(previous_residual_sum - current_residual_sum) < (
                MEDIAN_POLISH_CONVERGENCE_TOLERANCE * current_residual_sum
            ):
                break
            previous_residual_sum = current_residual_sum

    sample_estimates = overall_effect + col_effects
    sample_estimates[fully_missing_sample_mask] = np.nan

    return sample_estimates


class FeatureRanker:
    """Ranking methods for selecting top features based on quantification data."""

    @staticmethod
    def total_intensity(identification_df, quantification_df, col_to_groupby):
        """
        Rank features based on total intensity across all samples.

        Args:
            identification_df (pd.DataFrame): DataFrame containing feature identifications.
            quantification_df (pd.DataFrame): DataFrame containing feature quantifications.
            col_to_groupby (str): Column name to group by for ranking.

        Returns:
            pd.DataFrame: DataFrame with added 'rank_score' and 'rank' columns.
        """
        sum_intensity = quantification_df.sum(axis=1)
        identification_df.loc[:, "rank_score"] = sum_intensity
        identification_df.loc[:, "rank"] = identification_df.groupby(col_to_groupby)["rank_score"].rank(ascending=False)

        return identification_df

    @staticmethod
    def max_intensity(identification_df, quantification_df, col_to_groupby):
        """
        Rank features based on maximum intensity across all samples.

        Args:
            identification_df (pd.DataFrame): DataFrame containing feature identifications.
            quantification_df (pd.DataFrame): DataFrame containing feature quantifications.
            col_to_groupby (str): Column name to group by for ranking.

        Returns:
            pd.DataFrame: DataFrame with added 'rank_score' and 'rank' columns.
        """
        max_intensity = quantification_df.max(axis=1)
        identification_df.loc[:, "rank_score"] = max_intensity
        identification_df.loc[:, "rank"] = identification_df.groupby(col_to_groupby)["rank_score"].rank(ascending=False)

        return identification_df

    @staticmethod
    def median_intensity(identification_df, quantification_df, col_to_groupby):
        """
        Rank features based on median intensity across all samples.

        Args:
            identification_df (pd.DataFrame): DataFrame containing feature identifications.
            quantification_df (pd.DataFrame): DataFrame containing feature quantifications.
            col_to_groupby (str): Column name to group by for ranking.

        Returns:
            pd.DataFrame: DataFrame with added 'rank_score' and 'rank' columns.
        """
        median_intensity = quantification_df.median(axis=1)
        identification_df.loc[:, "rank_score"] = median_intensity
        identification_df.loc[:, "rank"] = identification_df.groupby(col_to_groupby)["rank_score"].rank(ascending=False)

        return identification_df

    @staticmethod
    def mean_intensity(identification_df, quantification_df, col_to_groupby):
        """
        Rank features based on mean intensity across all samples.

        Args:
            identification_df (pd.DataFrame): DataFrame containing feature identifications.
            quantification_df (pd.DataFrame): DataFrame containing feature quantifications.
            col_to_groupby (str): Column name to group by for ranking.

        Returns:
            pd.DataFrame: DataFrame with added 'rank_score' and 'rank' columns.
        """
        mean_intensity = quantification_df.mean(axis=1)
        identification_df.loc[:, "rank_score"] = mean_intensity
        identification_df.loc[:, "rank"] = identification_df.groupby(col_to_groupby)["rank_score"].rank(ascending=False)

        return identification_df


class Scorer:
    """Scoring methods for aggregating PSM scores to peptide/protein scores."""

    EPS = 1e-10

    def __init__(self, pep: float | np.ndarray | list[float]):
        self._raw_pep = np.asarray(pep, dtype=float)
        self._picked_pep: float | None = None

    @classmethod
    def best_pep(cls, values):
        """Factory for best PEP aggregation."""
        scorer = cls(values)
        scorer._picked_pep = scorer._best_pep()
        return scorer

    def _best_pep(self) -> float:
        """Return the minimum PEP (best evidence)."""
        arr = np.asarray(self._raw_pep, dtype=float)
        if arr.size == 0:
            return np.nan
        return np.nanmin(arr)

    @property
    def picked_pep(self) -> float:
        """The aggregated PEP value."""
        return self._picked_pep

    @property
    def picked_score(self) -> float:
        """The −log10 transformed score."""
        if self._picked_pep is None or np.isnan(self._picked_pep):
            return np.nan
        return -np.log10(self._picked_pep + self.EPS)

    @classmethod
    def func(cls, method: str):
        """Return a pure function that returns numeric PEPs (for pandas .agg)."""
        if method == "best_pep":
            return lambda x: cls.best_pep(x).picked_pep
        elif method == "combined":
            return lambda x: cls.combined(x).picked_pep
        else:
            raise ValueError(f"Scoring method '{method}' not recognized.")


class Aggregator:
    """
    Base class for aggregating identification and quantification data.
    """

    def __init__(
        self,
        identification_df: pd.DataFrame,
        quantification_df: pd.DataFrame,
        decoy_df: pd.DataFrame | None,
        agg_method: Literal["median", "mean", "sum", "median_polish", "directlfq"],
        score_method: Literal["best_pep"],
    ) -> None:
        self._id_df: pd.DataFrame = identification_df.copy()
        self._quant_df: pd.DataFrame = quantification_df.copy()
        self._decoy_id_df: pd.DataFrame = decoy_df.copy() if decoy_df is not None else pd.DataFrame()
        self._agg_method: Literal["median", "mean", "sum", "median_polish", "directlfq"] = agg_method
        self._score_method: Literal["best_pep"] = score_method

        self._id_agg_dict: dict = dict()  # placeholder
        self._col_to_groupby: str = ""  # placeholder
        self._decoy_agg_dict: dict = dict()  # placeholder

    @classmethod
    def peptide(
        cls,
        identification_df,
        quantification_df,
        decoy_df,
        agg_method,
        score_method,
        protein_col,
        peptide_col,
    ):
        """
        Create a peptide-level aggregator.
        """
        aggregator = cls(
            identification_df,
            quantification_df,
            decoy_df,
            agg_method,
            score_method,
        )
        aggregator._col_to_groupby = peptide_col
        aggregator._protein_col = protein_col
        aggregator._id_agg_dict = {
            aggregator._col_to_groupby: (aggregator._col_to_groupby, "first"),
            aggregator._protein_col: (aggregator._protein_col, "first"),
            "stripped_peptide": ("stripped_peptide", "first"),
            "count_psm": ("peptide", "count"),
            "PEP": ("PEP", Scorer.func(score_method)),
        }

        aggregator._decoy_agg_dict = {
            aggregator._protein_col: (aggregator._protein_col, "first"),
            "stripped_peptide": ("stripped_peptide", "first"),
            "PEP": ("PEP", Scorer.func(score_method)),
        }

        return aggregator

    @classmethod
    def protein(
        cls,
        identification_df,
        quantification_df,
        decoy_df,
        agg_method,
        score_method,
        protein_col,
    ):
        """
        Create a protein-level aggregator.
        """
        aggregator = cls(identification_df, quantification_df, decoy_df, agg_method, score_method)
        aggregator._col_to_groupby = protein_col
        aggregator._id_agg_dict = {
            # "total_psm": "sum",
            "count_psm": ("count_psm", "sum"),
            "count_stripped_peptide": ("stripped_peptide", "nunique"),
            "PEP": ("PEP", Scorer.func(score_method)),
        }

        aggregator._decoy_agg_dict = {"PEP": ("PEP", Scorer.func(score_method))}

        return aggregator

    @classmethod
    def ptm_site(
        cls,
        identification_df,
        quantification_df,
        agg_method,
    ):
        """
        Create a PTM site-level aggregator.
        """
        aggregator = cls(identification_df, quantification_df, None, agg_method, None)
        aggregator._col_to_groupby = "protein_site"
        aggregator._id_agg_dict = {
            "count_psm": ("count_psm", "sum"),
            "peptide": ("peptide", lambda x: ";".join(sorted(x.unique()))),
            "count_peptide": ("peptide", "nunique"),
            "count_stripped_peptide": ("stripped_peptide", "nunique"),
            "modified_protein": ("modified_protein", "first"),
            "protein_group": ("protein_group", "first"),
        }

        return aggregator

    def aggregate_identification(self) -> pd.DataFrame:
        agg_id_df: pd.DataFrame = self._id_df.copy()
        col_to_groupby = self._col_to_groupby

        agg_id_df = agg_id_df.groupby(col_to_groupby, observed=False).agg(**self._id_agg_dict)

        agg_id_df = agg_id_df.rename_axis(index=None)

        return agg_id_df

    def aggregate_quantification(self) -> pd.DataFrame:
        if isinstance(self._quant_df, SparseQuant):
            return self._aggregate_quantification_sparse()

        sample_columns = self._quant_df.columns
        agg_quant_df: pd.DataFrame = self._quant_df.copy()
        agg_quant_df[self._col_to_groupby] = self._id_df[self._col_to_groupby]
        grouped_quant = agg_quant_df.groupby(self._col_to_groupby, observed=False)

        # Matrix rollups operate on each group's full feature-by-sample submatrix, so they cannot
        # be expressed as a column-wise pandas aggregation and are applied per group instead.
        matrix_rollups = {
            "median_polish": _median_polish,
            "directlfq": _directlfq_rollup,
        }
        if self._agg_method in matrix_rollups:
            rollup_function = matrix_rollups[self._agg_method]
            agg_quant_df = grouped_quant[sample_columns].apply(
                lambda group_quant: pd.Series(
                    rollup_function(group_quant.to_numpy(dtype=float)),
                    index=sample_columns,
                )
            )
        else:
            agg_quant_df = grouped_quant.agg(self._agg_method)

        agg_quant_df = agg_quant_df.rename_axis(index=None)

        return agg_quant_df

    def _aggregate_quantification_sparse(self) -> pd.DataFrame:
        """Aggregate a sparse block-diagonal quantification per group without densifying it.

        Reduces feature columns within each group (median/mean/sum) directly on the sparse
        matrix, one small group-block at a time. The group order matches
        :meth:`aggregate_identification` (same pandas ``groupby`` order) so the returned frame
        aligns positionally with the aggregated identifications downstream.
        """
        if self._agg_method not in ("median", "mean", "sum"):
            # median_polish / directlfq are peptide->protein matrix rollups and are never applied
            # to the sparse PSM level; they run on the (dense) peptide modality in to_protein.
            raise NotImplementedError(
                f"Sparse quantification supports agg_method in ('median', 'mean', 'sum'); "
                f"got {self._agg_method!r}."
            )
        feature_groups = self._id_df[self._col_to_groupby].to_numpy()
        group_order = self._id_df.groupby(self._col_to_groupby, observed=False).size().index.to_numpy()
        groups, aggregated = aggregate_features_by_group(
            self._quant_df.matrix,
            feature_groups,
            self._agg_method,
            group_order=group_order,
        )
        agg_quant_df = pd.DataFrame(aggregated, index=groups, columns=self._quant_df.sample_names)
        return agg_quant_df.rename_axis(index=None)

    def aggregate_decoy(self) -> pd.DataFrame:
        agg_decoy_df: pd.DataFrame = self._decoy_id_df.copy()
        agg_decoy_df = agg_decoy_df.groupby(self._col_to_groupby, observed=False).agg(**self._decoy_agg_dict)

        agg_decoy_df = agg_decoy_df.rename_axis(index=None)

        return agg_decoy_df


class SummarisationPrep:
    """
    Preparation steps for summarisation.

    Attributes:
        mdata (MuData): MuData object containing feature-level data.
        filter_dict (dict): Dictionary specifying filtering criteria.
        rank_dict (dict): Dictionary specifying ranking criteria.
    """

    def __init__(self, adata: ad.AnnData, col_to_groupby: str, has_decoy: bool) -> None:
        self.adata: ad.AnnData = adata.copy()
        self._col_to_groupby = col_to_groupby

        self._filter_dict: dict = {}  # {"column_name": (keep, value)} | {"purity": ("gt", 0.7)}
        self._rank_tuple: tuple = ()  # ("method", num_top) | ("max_intensity", 3)
        self._has_decoy: bool = has_decoy

    @property
    def filter_dict(self) -> dict:
        return self._filter_dict

    @filter_dict.setter
    def filter_dict(self, new_filter_dict: dict) -> None:
        logger.debug("Applying filter criteria: %s", new_filter_dict)
        self._filter_dict = new_filter_dict

    @property
    def rank_tuple(self) -> tuple:
        return self._rank_tuple

    @rank_tuple.setter
    def rank_tuple(self, new_rank_tuple: tuple) -> None:
        logger.debug(
            "Ranking features by '%s' to select top %s features.",
            new_rank_tuple[0],
            new_rank_tuple[1],
        )
        self._rank_tuple = new_rank_tuple

    def prepare_data_to_summarise(self) -> pd.DataFrame:
        identification_df: pd.DataFrame = self.adata.var.copy()
        if is_sparse(self.adata.X):
            # Keep the block-diagonal quantification sparse; the aggregator reduces it per group
            # without materialising the full (samples x features) matrix. Values are aligned to
            # var order (== identification_df order) so no dense pivot is needed here.
            quantification_df = SparseQuant(
                matrix=self.adata.X.tocsc(),
                sample_names=list(self.adata.obs_names),
            )
        else:
            quantification_df = self.adata.to_df().transpose().copy()
        if self._has_decoy:
            decoy_df: pd.DataFrame = self.adata.uns["decoy"].copy()

        return (
            identification_df,
            quantification_df,
            decoy_df if self._has_decoy else None,
        )

    def _make_filter_mask(self, id_df: pd.DataFrame):
        filter_indices = pd.Series(False, index=id_df.index)

        for column, (keep, value) in self._filter_dict.items():
            column_mask = _mask_boolean_filter(series_to_mask=id_df[column], keep=keep, value=value)
            filter_indices = filter_indices | column_mask

        return filter_indices

    def _make_rank_mask(self) -> pd.Series:
        rank_method, top_n = self.rank_tuple

        ranked_id_df = FeatureRanker().__getattribute__(rank_method)(
            identification_df=self.adata.var,
            quantification_df=to_dense_df(self.adata).transpose(),
            col_to_groupby=self._col_to_groupby,
        )

        rank_mask = _mask_boolean_filter(series_to_mask=ranked_id_df["rank"], keep="le", value=top_n)

        return rank_mask

    def _mask_quantification(self, quant_df, mask_indices: pd.Series):
        if isinstance(quant_df, SparseQuant):
            # Drop stored entries in the masked-out feature columns so those features become
            # all-absent (contribute nothing to the group aggregation) -- no densification. The mask
            # is over features, whose order (id_df.index == var == matrix columns) matches the CSC.
            keep = np.asarray(mask_indices, dtype=bool)
            coo = quant_df.matrix.tocoo()
            keep_entry = keep[coo.col]
            masked = sp.coo_matrix(
                (coo.data[keep_entry], (coo.row[keep_entry], coo.col[keep_entry])),
                shape=quant_df.matrix.shape,
                dtype=quant_df.matrix.dtype,
            ).tocsc()
            return SparseQuant(matrix=masked, sample_names=quant_df.sample_names)

        mask_with_nan_quant = quant_df.copy()
        mask_with_nan_quant.loc[~mask_indices, :] = np.nan

        return mask_with_nan_quant

    def prep(self):
        identification_df, quantification_df, decoy_df = self.prepare_data_to_summarise()

        # Only the rank mask needs the quant densely (FeatureRanker ranks features by intensity); the
        # column/purity filter is computed from the identification frame and applied sparse-natively
        # (drop feature columns) below. So a sparse block-diagonal is densified only when a rank is
        # requested -- the common TMT to_peptide (purity filter, no rank) stays sparse end-to-end,
        # which is the whole point of the block-diagonal representation.
        if isinstance(quantification_df, SparseQuant) and self.rank_tuple:
            logger.debug("Densifying sparse quantification for rank-masked summarisation.")
            quantification_df = pd.DataFrame(
                dense_block(quantification_df.matrix).T,
                index=self.adata.var_names,
                columns=quantification_df.sample_names,
            )

        # make filter mask
        if self._filter_dict:
            filter_mask = self._make_filter_mask(identification_df)
            quantification_df = self._mask_quantification(quantification_df, filter_mask)

        # make rank mask
        if self.rank_tuple:
            rank_mask = self._make_rank_mask()
            quantification_df = self._mask_quantification(quantification_df, rank_mask)

        return (
            identification_df,
            quantification_df,
            decoy_df if self._has_decoy else None,
        )


class PtmSummarisationPrep(SummarisationPrep):
    """
    Preparation steps for PTM site summarisation.
        1. Filter data with only modified peptides with modi_identifier
        2. Get modified sites from peptide
        3. Label peptide site
        4. Explode data to single protein for labeling protein site
        5. Label protein site to each single protein
        6. Wrap up single protein to single protein group
        7. Group by modified peptide and its peptide site
        8. Merge data with peptide value indexed by peptide
    """

    def __init__(self, adata: ad.AnnData, modi_identifier: str, fasta: pd.DataFrame) -> None:
        self._modi_identifier = modi_identifier
        self._fasta_dict: dict = fasta["Sequence"].to_dict()
        self._col_to_groupby = "ptm_site"

        super().__init__(adata, self._col_to_groupby, has_decoy=False)

    def prep(self):
        identification_df, quantification_df, _ = self.prepare_data_to_summarise()
        # PTM sites are aggregated from the peptide modality, which is dense (peptides span samples,
        # so it is not block-diagonal). Densify defensively if a sparse .X is ever passed -- the
        # pd.merge below cannot operate on a SparseQuant.
        if isinstance(quantification_df, SparseQuant):
            quantification_df = pd.DataFrame(
                dense_block(quantification_df.matrix).T,
                index=self.adata.var_names,
                columns=quantification_df.sample_names,
            )
        identification_df["peptide"] = identification_df.index
        modi_df = self._extract_modi_peptide_df(data=identification_df)

        labelled_ptm_df = self.label_ptm_site(
            data=modi_df,
        )

        quantification_df = pd.merge(
            labelled_ptm_df[["peptide", "protein_site"]],
            quantification_df,
            how="left",
            left_on="peptide",
            right_index=True,
        ).drop(columns="peptide")

        # make rank mask
        if self.rank_tuple:
            rank_mask = self._make_rank_mask()
            quantification_df = self._mask_quantification(quantification_df, rank_mask)

        return labelled_ptm_df, quantification_df

    def _extract_modi_peptide_df(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        extracted_df: pd.DataFrame = data.copy()
        extracted_df = extracted_df.loc[extracted_df["peptide"].str.contains(self._modi_identifier, regex=False)].copy()
        logger.debug("Extracted modified peptides: %d / %d", len(extracted_df), len(data))

        return extracted_df

    def label_ptm_site(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Label PTM site to each single protein and get data arranged by peptide - peptide site

        Parameters:
            data (pd.DataFrame): Peptide data from msmu mudata['peptide']

        Returns:
            ptm_data (pd.DataFrame): PTM data arranged by peptide - peptide site
        """
        ptm_info: pd.DataFrame = data.copy()
        ptm_info["peptide_site"] = (
            ptm_info["peptide"].astype(str).apply(lambda x: self._get_mod_sites(x, self._modi_identifier))
        )

        # label peptide site
        ptm_info["peptide_site"] = ptm_info["peptide_site"].apply(lambda x: self._label_peptide_site(x))

        # explode data to single protein for label protein site
        ptm_info = self._explode_mod_site(ptm_info)
        ptm_info = self._explode_protein_groups(ptm_info)
        ptm_info = self._explode_protein_group(ptm_info)

        # label protein site to each single protein
        ptm_info["protein_site"] = ptm_info.apply(
            lambda x: self._label_protein_site(
                protein=x._prots,
                peptide=x.stripped_peptide,
                pep_site=x.peptide_site,
                fasta_dict=self._fasta_dict,
            ),
            axis=1,
        )
        ptm_info = ptm_info.loc[ptm_info["protein_site"].str.len() > 0].copy()
        ptm_info["modified_protein"] = ptm_info["protein_site"].apply(lambda x: x.split("|")[0])

        # wrap up single protein to single protein group
        ptm_info = self._implode_protein_group(ptm_info)

        # group by modified peptide and its peptide site
        ptm_info = self._implode_peptide_peptide_site(ptm_info)

        return ptm_info

    def _get_mod_sites(self, pep: str, modi_identifier: str) -> list:
        mod_sites: list = pep.split(modi_identifier)
        mod_sites: list = mod_sites[:-1]

        return mod_sites

    def _label_peptide_site(self, mod_sites: list) -> list:
        sites = list()
        site_pos: int = 0
        for mod in mod_sites:
            mod = "".join(filter(str.isalpha, mod))
            site_pos = site_pos + len(mod)
            site = f"{mod[-1]}{site_pos}"
            sites.append(site)

        return sites

    def _label_protein_site(self, protein: str, peptide: str, pep_site: str, fasta_dict: dict) -> str:
        aa: str = pep_site[0]
        pos: int = int(pep_site[1:])
        prot_site: str = ""

        res: list = list()
        prot_split = self._get_uniprot(protein)

        if prot_split in fasta_dict.keys():
            refseq: str = fasta_dict[prot_split]
            for match in re.finditer(peptide, refseq):
                matched = f"{prot_split}|{aa}{pos + match.span()[0]}"
                res.append(matched)
            prot_site = "/".join(res)

        return prot_site

    def _explode_mod_site(self, pep_labed_data: pd.DataFrame) -> pd.DataFrame:
        pep_labed_data = pep_labed_data.explode("peptide_site", ignore_index=True)

        return pep_labed_data

    def _explode_protein_groups(self, pep_labed_data: pd.DataFrame) -> pd.DataFrame:
        pep_labed_data["_prot_gr"] = pep_labed_data["protein_group"]
        pep_labed_data["_prot_gr"] = pep_labed_data["_prot_gr"].str.split(";")
        exploded_data = pep_labed_data.explode("_prot_gr", ignore_index=True)

        return exploded_data

    def _explode_protein_group(self, data) -> pd.DataFrame:
        data["_prots"] = data["_prot_gr"]
        data["_prots"] = data["_prots"].str.split(",")
        exploded_data = data.explode("_prots", ignore_index=True)

        return exploded_data

    def _implode_protein_group(self, data) -> pd.DataFrame:
        data = (
            data.groupby(["peptide", "peptide_site", "_prot_gr"], as_index=False, observed=True)
            .agg(
                {
                    "protein_site": ",".join,
                    "protein_group": "first",
                    "modified_protein": ",".join,
                    "stripped_peptide": "first",
                    "count_psm": "sum",
                    # "repr_protein": "first",
                }
            )
            .copy()
        )

        return data

    def _implode_peptide_peptide_site(self, data) -> pd.DataFrame:
        data = data.groupby(["peptide", "peptide_site"], as_index=False, observed=True).agg(
            {
                "protein_site": ";".join,
                "protein_group": "first",
                "modified_protein": ";".join,
                "stripped_peptide": "first",
                "count_psm": "sum",
                # "repr_protein": "first",
            }
        )

        return data

    def _get_uniprot(self, protein: str) -> str:
        return protein
