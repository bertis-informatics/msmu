import re
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..logging_utils import get_logger
from .._core._blockdiag import aggregate_features_by_group, dense_block, is_sparse, to_dense_df
from .._utils._anndata import _require_columns
from .._utils._pandas import split_delimited_strings
from .._utils.fasta import CANONICAL_CONTAMINANT_PREFIX
from .._utils.peptide import (
    MODIFICATION_TAG_CLOSER_BY_OPENER,
    is_residue_qualified_modification,
    parse_modified_peptide,
    residue_carries_modification,
)
from ._filter import _mask_boolean_filter

# for type checking only
import anndata as ad
from typing import Callable, Literal, get_args


logger = get_logger(__name__)

# How many unreadable peptidoforms a notation error quotes, and how many present tags a
# modification-not-found error lists; enough to diagnose, few enough to read.
MAX_REPORTED_MISREAD_PEPTIDES: int = 5
MAX_REPORTED_PRESENT_TAGS: int = 10
# How many accessions or peptide-accession pairs a FASTA mismatch names before it says "and N more".
MAX_REPORTED_UNLOCALISABLE_EXAMPLES: int = 5

# "single" is reserved for a single-site table that prefers singly modified peptidoforms and fills the
# sites seen only on multiply modified ones from the least-modified form (TMT-Integrator, PTM-SEA).
Multisite = Literal["pool", "combination"]
_MULTISITE_OPTIONS: tuple[str, ...] = get_args(Multisite)
# Joins the sites of one peptidoform inside a site-combination label: "P1|S5_S8". An underscore rather
# than "+": it survives regular expressions, R column names and file names, and it is what MSstatsPTM
# writes. It only ever follows the label's last "|", so an accession containing "_" stays unambiguous.
SITE_COMBINATION_SEPARATOR: str = "_"


def _format_examples(values: set[str]) -> str:
    """Render a few sorted examples, saying how many were left out."""
    shown = sorted(values)[:MAX_REPORTED_UNLOCALISABLE_EXAMPLES]
    remainder = len(values) - len(shown)
    return ", ".join(shown) + (f", and {remainder} more" if remainder else "")


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
    (e.g. apply [`log2_transform`][msmu.pp.log2_transform] first) and the output is log2.

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


# log2 intensities from real MS data sit well under 64 (2**64 is ~1.8e19, far above any observed
# linear intensity), so a maximum above it is a reliable sign the matrix is still on a linear scale.
MAX_PLAUSIBLE_LOG2_INTENSITY: float = 64.0


def warn_if_not_log_scale(matrix, context: str) -> None:
    """Warn when a matrix headed for an additive rollup still looks like linear intensities.

    ``median_polish`` and ``directlfq`` fit an additive feature + sample model, which is only
    meaningful in log space. Applying them to linear intensities silently produces nonsense rather
    than failing, so this warns instead of leaving the user with no signal. It is a heuristic, hence
    a warning and not an error.
    """
    values = dense_block(matrix) if is_sparse(matrix) else np.asarray(matrix, dtype=float)
    if values.size == 0:
        return

    observed_maximum = np.nanmax(values) if np.any(~np.isnan(values)) else np.nan
    if np.isfinite(observed_maximum) and observed_maximum > MAX_PLAUSIBLE_LOG2_INTENSITY:
        logger.warning(
            "%s: values reach %.3g, which looks like linear intensity rather than log2. "
            "Additive rollups (median_polish, directlfq) assume log space -- apply log2_transform first.",
            context,
            observed_maximum,
        )


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


# Rollups that consume a group's whole feature-by-sample submatrix rather than reducing each column
# independently. Both fit an additive model and so require log-space input; keeping the mapping here
# gives the aggregator and the callers that must warn about scale a single source of truth.
MATRIX_ROLLUP_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "median_polish": _median_polish,
    "directlfq": _directlfq_rollup,
}
MATRIX_ROLLUP_METHODS: tuple[str, ...] = tuple(MATRIX_ROLLUP_FUNCTIONS)


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
            "count_site": ("count_site", "first"),
        }

        return aggregator

    def aggregate_identification(self) -> pd.DataFrame:
        agg_id_df: pd.DataFrame = self._id_df.copy()
        col_to_groupby = self._col_to_groupby

        agg_id_df = agg_id_df.groupby(col_to_groupby, observed=True).agg(**self._id_agg_dict)

        agg_id_df = agg_id_df.rename_axis(index=None)

        return agg_id_df

    def aggregate_quantification(self) -> pd.DataFrame:
        if isinstance(self._quant_df, SparseQuant):
            return self._aggregate_quantification_sparse()

        # The PTM path carries its grouping column inside the quantification frame (it is what aligns
        # quant rows to the exploded site rows), so exclude it explicitly. The column-wise ``.agg``
        # branch drops the groupby key on its own, but a matrix rollup would try to average a string.
        sample_columns = [column for column in self._quant_df.columns if column != self._col_to_groupby]
        agg_quant_df: pd.DataFrame = self._quant_df.copy()
        agg_quant_df[self._col_to_groupby] = self._id_df[self._col_to_groupby]
        grouped_quant = agg_quant_df.groupby(self._col_to_groupby, observed=True)

        # Matrix rollups operate on each group's full feature-by-sample submatrix, so they cannot
        # be expressed as a column-wise pandas aggregation and are applied per group instead.
        if self._agg_method in MATRIX_ROLLUP_FUNCTIONS:
            rollup_function = MATRIX_ROLLUP_FUNCTIONS[self._agg_method]
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
        group_order = self._id_df.groupby(self._col_to_groupby, observed=True).size().index.to_numpy()
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
        agg_decoy_df = agg_decoy_df.groupby(self._col_to_groupby, observed=True).agg(**self._decoy_agg_dict)

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


def normalise_target_modifications(modification: str | Sequence[str]) -> tuple[str, ...]:
    """Validate [`to_ptm`][msmu.pp.to_ptm]'s ``modification`` argument and return it as a tuple of tags.

    Each entry must be a tag exactly as the search engine writes it (``[+79.9663]``, ``(UniMod:21)``)
    or a tag qualified by its residue (``S[167]``). Matching is by equality with a parsed tag, not by
    substring, so anything that does not look like a tag can never match and is rejected here.
    """
    modifications = (modification,) if isinstance(modification, str) else tuple(modification)
    if not modifications:
        raise ValueError("modification must name at least one modification tag.")

    for target_modification in modifications:
        tag_text = _get_modification_tag_text(target_modification) if isinstance(target_modification, str) else ""
        if tag_text[:1] not in MODIFICATION_TAG_CLOSER_BY_OPENER:
            raise ValueError(
                f"Invalid modification {target_modification!r}. Pass the tag exactly as it appears in the "
                "peptide string, brackets included -- e.g. '[+79.9663]' (Sage), '(UniMod:21)' (DIA-NN) -- "
                "or qualify it with its residue, e.g. 'S[167]' (FragPipe)."
            )

    return modifications


def _get_modification_tag_text(modification: str) -> str:
    """The tag part of a target modification: ``S[167]`` -> ``[167]``; a bare tag is returned as is."""
    return modification[1:] if is_residue_qualified_modification(modification) else modification


class PtmSummarisationPrep(SummarisationPrep):
    """
    Preparation steps for PTM site summarisation.
        1. Keep the peptidoforms carrying a target modification, labelled with the peptide positions
           of the residues that carry it
        2. Explode data to one row per peptide site
        3. Explode data to the peptide's own accessions for labeling protein site
        4. Label protein site to each single protein
        5. Group by modified peptide and its peptide site
        6. Merge data with peptide value indexed by peptide
    """

    def __init__(
        self,
        adata: ad.AnnData,
        modification: str | Sequence[str],
        fasta: pd.DataFrame,
        multisite: Multisite = "combination",
    ) -> None:
        if multisite not in _MULTISITE_OPTIONS:
            raise ValueError(f"Unknown multisite option '{multisite}'. Choose from {_MULTISITE_OPTIONS}.")
        self._multisite: Multisite = multisite
        self._target_modifications: tuple[str, ...] = normalise_target_modifications(modification)
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
        _require_columns(
            identification_df,
            columns=["proteins", "stripped_peptide"],
            context="peptide.var (PTM site localisation)",
        )
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
        """Keep the peptidoforms carrying a target modification, labelled with where it sits.

        ``peptide_site`` lists every residue carrying a target modification as ``<residue><1-based
        position>`` (e.g. ``S7``), counted over residues only, so the letters inside a tag such as
        ``(UniMod:35)`` never shift it. Each parse is checked against the engine's own
        ``stripped_peptide``; a disagreement means msmu misread the notation, and every site in that
        peptide would be misplaced, so it raises instead.
        """
        peptide_strings = data["peptide"].astype(str)
        # A cheap substring pre-filter, so only plausible peptidoforms are parsed -- and a notation
        # msmu cannot read fails only if it actually carries the modification being summarised.
        has_target_tag_text = pd.Series(False, index=data.index)
        for target_modification in self._target_modifications:
            has_target_tag_text |= peptide_strings.str.contains(
                _get_modification_tag_text(target_modification), regex=False
            )
        candidate_df: pd.DataFrame = data.loc[has_target_tag_text].copy()

        peptide_sites: list[list[str]] = []
        misread_peptide_descriptions: list[str] = []
        for peptide, stripped_peptide in zip(
            candidate_df["peptide"].astype(str), candidate_df["stripped_peptide"].astype(str)
        ):
            try:
                modified_residues = parse_modified_peptide(peptide)
            except ValueError as parse_error:
                misread_peptide_descriptions.append(str(parse_error))
                peptide_sites.append([])
                continue

            parsed_sequence = "".join(modified_residue.residue for modified_residue in modified_residues)
            if parsed_sequence != stripped_peptide:
                misread_peptide_descriptions.append(
                    f"{peptide!r} parsed as {parsed_sequence!r}, but stripped_peptide is {stripped_peptide!r}"
                )
                peptide_sites.append([])
                continue

            peptide_sites.append(
                [
                    f"{modified_residue.residue}{modified_residue.position_in_peptide}"
                    for modified_residue in modified_residues
                    if any(
                        residue_carries_modification(modified_residue, target_modification)
                        for target_modification in self._target_modifications
                    )
                ]
            )

        if misread_peptide_descriptions:
            raise ValueError(
                f"Could not read the modification notation of {len(misread_peptide_descriptions)} "
                f"peptidoform(s), so their site positions would be wrong. First "
                f"{min(len(misread_peptide_descriptions), MAX_REPORTED_MISREAD_PEPTIDES)}:\n  "
                + "\n  ".join(misread_peptide_descriptions[:MAX_REPORTED_MISREAD_PEPTIDES])
            )

        candidate_df["peptide_site"] = pd.Series(peptide_sites, index=candidate_df.index, dtype=object)
        has_target_site = np.array([len(sites) > 0 for sites in peptide_sites], dtype=bool)
        extracted_df = candidate_df.loc[has_target_site].copy()
        if extracted_df.empty:
            raise ValueError(self._describe_missing_target_modification(peptide_strings))

        logger.debug("Extracted modified peptides: %d / %d", len(extracted_df), len(data))

        return extracted_df

    def _describe_missing_target_modification(self, peptide_strings: pd.Series) -> str:
        """Explain a modification that matched nothing, listing the tags the data does contain."""
        peptidoform_count_by_tag: Counter[str] = Counter()
        for peptide in peptide_strings.unique():
            try:
                modified_residues = parse_modified_peptide(peptide)
            except ValueError:
                continue
            peptidoform_count_by_tag.update(
                {tag for modified_residue in modified_residues for tag in modified_residue.tags}
            )

        present_tags = ", ".join(
            f"{tag!r} x{count}" for tag, count in peptidoform_count_by_tag.most_common(MAX_REPORTED_PRESENT_TAGS)
        )
        return (
            f"No peptidoform carries the modification {list(self._target_modifications)}. Tags are matched "
            f"exactly, brackets and case included. Tags present (peptidoform counts): "
            f"{present_tags or 'none'}. A tag may also be qualified by its residue, e.g. 'S[167]'."
        )

    def label_ptm_site(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Label PTM site to each single protein and get data arranged by peptide - peptide site

        Site identity is derived from the peptide's own ``proteins`` accessions rather than from an
        inferred ``protein_group``. Localisation needs only an accession, a sequence and the FASTA;
        protein grouping is a judgement made from one dataset's peptide evidence, so folding it into
        the site name would make the same PTM data yield different site ids depending on which global
        dataset it was processed alongside. Keeping accessions flat makes the site id a function of
        (peptide, FASTA) alone. Resolving an accession to a quantifiable protein group is the
        denominator step's job, not this one's.

        Parameters:
            data (pd.DataFrame): Peptide data from msmu mudata['peptide'], already reduced to the
                target peptidoforms and labelled with ``peptide_site`` by ``_extract_modi_peptide_df``

        Returns:
            ptm_data (pd.DataFrame): PTM data arranged by peptide - peptide site
        """
        ptm_info: pd.DataFrame = data.copy()

        if self._multisite == "combination":
            # The peptidoform is assigned to the set of sites it carries, as one unit: a multiply
            # modified peptide's change cannot be attributed to one of its sites, the way a shared
            # peptide's cannot be attributed to one protein. Joining the sites here makes the explode
            # below a no-op, so each peptidoform reaches exactly one feature.
            ptm_info["count_site"] = ptm_info["peptide_site"].map(len)
            ptm_info["peptide_site"] = ptm_info["peptide_site"].map(SITE_COMBINATION_SEPARATOR.join)
        else:
            ptm_info["count_site"] = 1

        # explode data to single protein for label protein site
        ptm_info = self._explode_mod_site(ptm_info)
        ptm_info = self._explode_proteins(ptm_info)

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
        self._report_unlocalisable_matches(ptm_info)
        ptm_info = ptm_info.loc[ptm_info["protein_site"].str.len() > 0].copy()
        # The accession itself, not protein_site cut at its first "|": accessions such as GENCODE ids
        # contain "|", and this column is what adjust_ptm_by_protein resolves denominators from.
        ptm_info["modified_protein"] = ptm_info["_prots"]

        # group by modified peptide and its peptide site
        ptm_info = self._implode_peptide_peptide_site(ptm_info)

        return ptm_info

    def _report_unlocalisable_matches(self, labelled_ptm_info: pd.DataFrame) -> None:
        """Report the peptide-accession matches the attached FASTA cannot reproduce.

        The search engine found each peptide in that accession's sequence, so an accession the
        attached FASTA does not hold -- or a sequence of it that does not contain the peptide --
        means the attached FASTA is not the one the search used. Such matches are dropped, which
        silently costs sites and can also turn a site that should be ``shared_groups`` into an
        adjusted one, so they are counted here rather than passed over.

        Contaminant accessions are reported separately: search engines add contaminant entries of
        their own, so a user's FASTA routinely lacks them and that alone is not a mismatch.
        """
        is_unlocalised = labelled_ptm_info["protein_site"].str.len() == 0
        absent_accessions: set[str] = set()
        absent_contaminants: set[str] = set()
        sequence_mismatches: set[str] = set()
        unreproducible_match_count = 0
        for accession, stripped_peptide in zip(
            labelled_ptm_info.loc[is_unlocalised, "_prots"].astype(str),
            labelled_ptm_info.loc[is_unlocalised, "stripped_peptide"].astype(str),
        ):
            if self._get_uniprot(accession) in self._fasta_dict:
                sequence_mismatches.add(f"{stripped_peptide} in {accession}")
            elif CANONICAL_CONTAMINANT_PREFIX in accession:
                absent_contaminants.add(accession)
                continue
            else:
                absent_accessions.add(accession)
            unreproducible_match_count += 1

        if absent_contaminants:
            logger.info(
                "%d contaminant accessions are not in the attached FASTA (%s). Search engines add "
                "their own contaminant entries, so this is expected unless the same contaminants "
                "were part of the search database.",
                len(absent_contaminants),
                _format_examples(absent_contaminants),
            )

        if not absent_accessions and not sequence_mismatches:
            return

        reasons = []
        if absent_accessions:
            reasons.append(
                f"{len(absent_accessions)} accessions are absent from it ({_format_examples(absent_accessions)})"
            )
        if sequence_mismatches:
            reasons.append(
                f"{len(sequence_mismatches)} peptides are not in the attached sequence of their "
                f"accession ({_format_examples(sequence_mismatches)})"
            )

        has_any_site = labelled_ptm_info.groupby("peptide", observed=True)["protein_site"].apply(
            lambda protein_sites: bool((protein_sites.str.len() > 0).any())
        )
        logger.warning(
            "The attached FASTA cannot reproduce %d of the search engine's peptide-protein matches: %s. "
            "%d of %d modified peptidoforms produced no site at all. Attach the FASTA the search used -- "
            "sites are otherwise lost, and a site whose accessions no longer span two protein groups can "
            "be adjusted as if it were unambiguous.",
            unreproducible_match_count,
            "; ".join(reasons),
            int((~has_any_site).sum()),
            len(has_any_site),
        )

    def _label_protein_site(self, protein: str, peptide: str, pep_site: str, fasta_dict: dict) -> str:
        # One peptide site ("S5") or a site combination ("S5_S8"); each is shifted by the peptide's offset.
        peptide_sites: list[tuple[str, int]] = [
            (site[0], int(site[1:])) for site in pep_site.split(SITE_COMBINATION_SEPARATOR)
        ]
        prot_site: str = ""

        res: list = list()
        prot_split = self._get_uniprot(protein)

        if prot_split in fasta_dict.keys():
            refseq: str = fasta_dict[prot_split]
            for match in re.finditer(peptide, refseq):
                protein_sites = SITE_COMBINATION_SEPARATOR.join(
                    f"{residue}{position + match.span()[0]}" for residue, position in peptide_sites
                )
                res.append(f"{prot_split}|{protein_sites}")
            prot_site = "/".join(res)

        return prot_site

    def _explode_mod_site(self, pep_labed_data: pd.DataFrame) -> pd.DataFrame:
        pep_labed_data = pep_labed_data.explode("peptide_site", ignore_index=True)

        return pep_labed_data

    def _explode_proteins(self, pep_labed_data: pd.DataFrame) -> pd.DataFrame:
        """Explode the peptide's accession list to one row per accession, in canonical order.

        The accessions are sorted so the imploded ``protein_site`` is the same string whichever order
        the search engine happened to list them in -- otherwise two peptidoforms covering one site
        could disagree on the site name and split into two features.
        """
        accessions = split_delimited_strings(pep_labed_data["proteins"], ";")
        pep_labed_data["_prots"] = accessions.apply(lambda parts: sorted(parts) if isinstance(parts, list) else parts)
        exploded_data = pep_labed_data.explode("_prots", ignore_index=True)

        return exploded_data

    def _implode_peptide_peptide_site(self, data) -> pd.DataFrame:
        data = data.groupby(["peptide", "peptide_site"], as_index=False, observed=True).agg(
            {
                "protein_site": ";".join,
                "modified_protein": ";".join,
                "stripped_peptide": "first",
                # "first", not "sum": the accession explode duplicated this peptidoform's row, so
                # summing here would multiply its PSM count by the number of accessions it maps to.
                "count_psm": "first",
                "count_site": "first",
            }
        )

        return data

    def _get_uniprot(self, protein: str) -> str:
        return protein
