import numpy as np
import pandas as pd
import mudata as md
from scipy.interpolate import interp1d
from scipy.stats import rankdata
from sklearn.linear_model import Ridge

from typing import Callable, Literal, get_args

from .._utils._mudata import get_anndata_mod, get_mudata_mod_as_mutable
from .._utils._anndata import _require_columns
from .._utils._pandas import split_delimited_strings
from .._core._blockdiag import to_dense_df
from ..logging_utils import get_logger

logger = get_logger(__name__)

NormalisationMethod = Literal["median", "median_center", "quantile", "total_sum"]
# Runtime tuple derived from the Literal so the accepted methods have a single source of truth.
_NORMALISATION_METHODS: tuple[str, ...] = get_args(NormalisationMethod)

PTMAdjustmentMethod = Literal["ratio", "ridge"]
_PTM_ADJUSTMENT_METHODS: tuple[str, ...] = get_args(PTMAdjustmentMethod)

# Per-site outcome of the denominator search, recorded in var["adjustment_status"].
ADJUSTMENT_STATUS_ADJUSTED: str = "adjusted"
# Accessions resolve to two or more quantified global groups. The site's signal is a sum over those
# groups and rollup values are not comparable between groups, so no valid denominator exists.
ADJUSTMENT_STATUS_SHARED_GROUPS: str = "shared_groups"
# An accession maps to a global protein group, but that group has no quantification (e.g. all of its
# peptides were shared, so to_protein discarded them).
ADJUSTMENT_STATUS_NOT_QUANTIFIED: str = "not_quantified"
# No accession appears in the global protein_map, though at least one is in the global FASTA -- the
# protein was searchable there but never identified. Expected under enrichment.
ADJUSTMENT_STATUS_NO_GLOBAL_GROUP: str = "no_global_group"
# No accession is in the global FASTA at all: the two searches used different databases.
ADJUSTMENT_STATUS_NOT_IN_GLOBAL_FASTA: str = "not_in_global_fasta"
# A denominator was found, but the estimator declined to produce a value (ridge needs more paired
# observations than the site has).
ADJUSTMENT_STATUS_NO_ESTIMATE: str = "no_estimate"

# Ridge needs more than this many paired observations before a slope is worth estimating.
MIN_RIDGE_PAIRED_SAMPLES: int = 2
# A single-predictor ridge keeps ``Sxx / (Sxx + alpha)`` of the least-squares slope, where
# ``Sxx = (n - 1) * var(protein)``. At proteomics-scale spread (~0.3-1.0 log2) and sample counts,
# an alpha near 100 leaves under a tenth of the slope -- the residual then reduces to centring the
# site and no protein correction happens at all. This default keeps most of the slope; raise it
# deliberately if shrinkage is wanted.
DEFAULT_RIDGE_ALPHA: float = 1.0


def _resolve_site_denominator(
    accessions: list[str],
    accession_to_group: dict[str, str],
    quantified_groups: set[str],
    global_fasta_accessions: set[str] | None,
) -> tuple[str | None, str]:
    """Pick the global protein group whose quantification should serve as a site's denominator.

    Works from the site's accessions rather than any protein-group string, because groups are a
    per-dataset judgement while accessions are shared. A site is adjustable exactly when its
    accessions land on one quantified global group -- including the common case where several
    accessions are indistinguishable in the global data and therefore *are* one group.
    """
    mapped_groups = {accession_to_group[accession] for accession in accessions if accession in accession_to_group}
    quantified = sorted(group for group in mapped_groups if group in quantified_groups)

    if len(quantified) == 1:
        return quantified[0], ADJUSTMENT_STATUS_ADJUSTED
    if len(quantified) > 1:
        return None, ADJUSTMENT_STATUS_SHARED_GROUPS
    if mapped_groups:
        return None, ADJUSTMENT_STATUS_NOT_QUANTIFIED
    if global_fasta_accessions is not None and not any(
        accession in global_fasta_accessions for accession in accessions
    ):
        return None, ADJUSTMENT_STATUS_NOT_IN_GLOBAL_FASTA

    return None, ADJUSTMENT_STATUS_NO_GLOBAL_GROUP


class Normalisation:
    def __init__(self, method: NormalisationMethod, axis: str) -> None:
        if method not in _NORMALISATION_METHODS:
            raise ValueError(f"Unknown normalisation method '{method}'. Choose from {_NORMALISATION_METHODS}.")
        self._method = method
        self._method_call: Callable = getattr(self, f"_{method}")
        self._axis = axis

    def _quantile(self, arr) -> np.ndarray:
        return normalise_quantile(arr=arr)

    def _median(self, arr) -> np.ndarray:
        all_median = np.nanmedian(arr.flatten())
        arr = normalise_median_center(arr=arr)  # median center
        arr = arr + all_median  # add scalar to preserve overall scale

        return arr

    def _median_center(self, arr) -> np.ndarray:
        return normalise_median_center(arr=arr)

    def _total_sum(self, arr) -> np.ndarray:
        return normalise_total_sum(arr)

    # Sparse per-block rescalers, mirroring the dense ``_{method}`` above. The obs/var partitioning and
    # cell gathering live in ``_normalise._normalise_per_group_sparse``; these just rewrite the given
    # block's stored ``.data`` in place, nan-aware to match the dense path. A method is sparse-native iff
    # it defines ``_{method}_sparse`` -- quantile does not (its per-sample rank mapping couples all
    # samples), so it densifies instead.
    def _total_sum_sparse(self, csr, cell_indices) -> None:
        data = csr.data
        row_totals = np.array([np.nansum(np.exp2(data[idx].astype(np.float64))) for idx in cell_indices])
        block_log2_target = np.log2(np.median(row_totals))
        for idx, row_total in zip(cell_indices, row_totals):
            data[idx] = data[idx] + csr.dtype.type(block_log2_target - np.log2(row_total))

    def _median_sparse(self, csr, cell_indices) -> None:
        self._centre_on_median_sparse(csr, cell_indices, add_block_median=True)

    def _median_center_sparse(self, csr, cell_indices) -> None:
        self._centre_on_median_sparse(csr, cell_indices, add_block_median=False)

    @staticmethod
    def _centre_on_median_sparse(csr, cell_indices, add_block_median: bool) -> None:
        data = csr.data
        if add_block_median:
            block_median = np.nanmedian(np.concatenate([data[idx] for idx in cell_indices]))
        else:
            block_median = csr.dtype.type(0)
        for idx in cell_indices:
            values = data[idx]
            data[idx] = values - np.nanmedian(values) + block_median

    @property
    def is_sparse_native(self) -> bool:
        """Whether this method can be computed on the sparse block-diagonal without densifying."""
        return hasattr(self, f"_{self._method}_sparse")

    def rescale_sparse_block(self, csr, cell_indices) -> None:
        """Rescale one block's stored cells in place, dispatching to this method's sparse rescaler."""
        getattr(self, f"_{self._method}_sparse")(csr, cell_indices)

    def normalise(self, arr) -> np.ndarray:
        na_idx = np.isnan(arr)
        if self._axis == "obs":
            transposed_arr = arr.T
            normalised_arr = self._method_call(arr=transposed_arr)
            normalised_arr = normalised_arr.T

        elif self._axis == "var":
            normalised_arr = self._method_call(arr=arr)

        else:
            raise ValueError(f"Axis {self._axis} not recognised. Please choose from 'obs' or 'var'")

        normalised_arr[na_idx] = np.nan

        return normalised_arr


def normalise_quantile(arr: np.ndarray) -> np.ndarray:
    # set defaults
    values = np.array(arr)
    tiedFlag = True

    # allocate some space for the normalized values
    normalizedVals = values
    valSize = values.shape
    rankedVals = np.zeros(valSize) * np.nan

    # find nans
    nanvals = np.isnan(values)
    numNans = np.sum(nanvals, axis=0)
    ndx = np.ones(valSize, dtype=np.int64)
    N = valSize[0]

    # create space for output
    if tiedFlag:
        rr = np.empty([valSize[1]], dtype=object)

    # for each column we want to ordered values and the ranks with ties
    for col in range(valSize[1]):
        sortedVals = np.sort(values[:, col])
        ndx[:, col] = np.argsort(values[:, col])
        if tiedFlag:
            rr[col] = np.sort(rankdata(values[~nanvals[:, col], col]))
        M = N - numNans[col]
        x = np.arange(0, N, (N - 1) / (M - 1))
        y = sortedVals[0:M]
        try:
            f = interp1d(x, y, bounds_error=False)
        except Exception as exc:
            logger.exception("Interpolation failed at column %s with shape %s.", col, y.shape)
            raise RuntimeError(f"Interpolation failed during quantile normalization at column {col}.") from exc
        xnew = np.arange(0, N)
        ynew = f(xnew)
        rankedVals[:, col] = ynew

    # take the mean of the ranked values
    mean_vals = np.nanmean(rankedVals, axis=1)

    # Extract the values from the normalized distribution
    for col in range(valSize[1]):
        M = N - numNans[col]
        if tiedFlag:
            x = np.arange(0, N)
            y = mean_vals
            f = interp1d(x, y, bounds_error=False)
            xnew = (N - 1) * (rr[col] - 1) / (M - 1)
            ynew = f(xnew)
            normalizedVals[ndx[0:M, col], col] = ynew
        else:
            x = np.arange(0, N)
            y = mean_vals
            f = interp1d(x, y, bounds_error=False)
            xnew = np.arange(0, N, (N - 1) / (M - 1))
            ynew = f(xnew)
            normalizedVals[ndx[0:M, col], col] = ynew

    normalizedVals[nanvals] = np.nan

    return normalizedVals


def normalise_median_center(arr: np.ndarray) -> np.ndarray:
    """Median centering of data"""
    raw_arr = arr.copy()
    median_data = np.nanmedian(raw_arr, axis=0)

    median_centered_data = raw_arr - median_data

    return median_centered_data


def normalise_total_sum(arr: np.ndarray) -> np.ndarray:
    """Total-intensity (constant-sum) normalisation.

    Rescales every sample so its summed intensity equals ``T`` -- the median of the per-sample totals
    -- correcting sample-to-sample loading / injection differences while leaving within-sample feature
    ratios unchanged. ``arr`` is oriented (features x samples) here (``Normalisation`` transposes the
    obs axis before calling), so the totals are taken per column.

    The input is assumed log2-transformed, matching the msmu convention that ``normalise`` runs after
    ``log2_transform``. Summing log values is meaningless (it yields the log of the product, not the
    total), so each sample total is computed on the linear scale (``2 ** arr``) and the rescale is
    returned to log2. On the log2 scale this reduces to a per-sample additive shift
    ``log2(T) - log2(S_i)``. Structurally-absent cells (NaN) contribute nothing to the total and stay
    NaN in the result.
    """
    linear_values: np.ndarray = np.exp2(arr.astype(np.float64))
    sample_totals: np.ndarray = np.nansum(linear_values, axis=0)
    target_total: float = np.median(sample_totals)
    per_sample_shift: np.ndarray = np.log2(target_total) - np.log2(sample_totals)

    return arr + per_sample_shift


class PTMProteinAdjuster:
    """Adjust PTM site intensities by their parent protein's abundance in a matched global dataset.

    This computes *differential PTM usage* (DPU) in the sense of Demeulemeester et al. 2024: the
    site's log intensity minus its parent protein's summarised log intensity in the same sample. It
    is not occupancy/stoichiometry, which additionally needs the unmodified counterpart peptide.

    The denominator is found by **accession**, not by peptide and not by comparing protein-group
    strings. A site carries the accessions it was localised on; each is translated through the global
    dataset's own ``protein_map`` into a global protein group, and that group's quantification is the
    denominator. Protein groups are a judgement derived from one dataset's peptide evidence, so two
    datasets need not agree on them -- but they do agree on accessions. Going through accessions also
    means a PTM peptide that was never observed in the global run is still adjustable as long as its
    protein was quantified there, which is the common case under enrichment.

    Sites whose accessions resolve to two or more *quantified* global groups are left unadjusted.
    Their measured signal is a sum over those groups, and protein rollup values carry a per-protein
    offset that makes them incomparable across groups -- so neither picking one group nor summing
    them yields a valid denominator. This is a property of the measurement, not a lookup failure.
    """

    def __init__(
        self,
        ptm_mdata: md.MuData,
        global_mdata: md.MuData,
        ptm_mod: str,
        global_mod: str,
        accession_column: str = "modified_protein",
    ):
        self.ptm_mdata = ptm_mdata
        self.ptm_mod = ptm_mod
        self.global_mdata = global_mdata
        self.global_mod = global_mod
        self.accession_column = accession_column
        self.sample_cols: list[str] = list(ptm_mdata.obs.index)

        self.resolution = self._resolve_denominators()
        self.ptm_data, self.global_data = self._extract_data()

    # ------------------------------------------------------------------ resolution

    def _resolve_denominators(self) -> pd.DataFrame:
        """Map every PTM site to the global protein group that should serve as its denominator."""
        ptm_adata = get_anndata_mod(self.ptm_mdata, self.ptm_mod)
        global_adata = get_anndata_mod(self.global_mdata, self.global_mod)

        if self.accession_column not in ptm_adata.var.columns:
            raise ValueError(
                f"Required column missing from {self.ptm_mod}.var: '{self.accession_column}'. "
                "It is written by to_ptm and holds the accessions each site was localised on."
            )

        accession_to_group = self._read_accession_to_group()
        quantified_groups = set(global_adata.var_names)
        global_fasta_accessions = self._read_global_fasta_accessions()

        site_accessions = split_delimited_strings(ptm_adata.var[self.accession_column].astype(str), ";")

        denominator_groups: list[str | None] = []
        statuses: list[str] = []
        for accessions in site_accessions:
            accession_list = [] if not isinstance(accessions, list) else [a for a in accessions if a]
            group, status = _resolve_site_denominator(
                accessions=accession_list,
                accession_to_group=accession_to_group,
                quantified_groups=quantified_groups,
                global_fasta_accessions=global_fasta_accessions,
            )
            denominator_groups.append(group)
            statuses.append(status)

        resolution = pd.DataFrame(
            {"denominator_group": denominator_groups, "adjustment_status": statuses},
            index=ptm_adata.var_names,
        )
        self._log_resolution(resolution)

        return resolution

    def _read_accession_to_group(self) -> dict[str, str]:
        """Read the global dataset's accession -> protein group translation table."""
        if "protein_map" not in self.global_mdata.uns:
            raise ValueError(
                "Global MuData is missing uns['protein_map']; run infer_protein on the global "
                "dataset before adjusting PTM data."
            )

        protein_map = self.global_mdata.uns["protein_map"]
        _require_columns(
            protein_map,
            columns=["initial_protein", "protein_group"],
            context="global uns['protein_map']",
        )

        return dict(zip(protein_map["initial_protein"].astype(str), protein_map["protein_group"].astype(str)))

    def _read_global_fasta_accessions(self) -> set[str] | None:
        """Accessions present in the global dataset's FASTA, when one is attached.

        Used only to tell two very different failures apart: an accession the global search could
        have found but did not (normal, expected under enrichment), versus one that was not in the
        global search database at all (the two searches used different FASTAs -- a configuration
        error no policy should paper over). Returns None when no FASTA is attached, in which case
        the two are reported together.
        """
        protein_info = self.global_mdata.uns.get("protein_info")
        if protein_info is None or not hasattr(protein_info, "index"):
            return None

        return set(protein_info.index.astype(str))

    def _log_resolution(self, resolution: pd.DataFrame) -> None:
        counts = resolution["adjustment_status"].value_counts()
        total_sites = len(resolution)
        adjustable = int(counts.get(ADJUSTMENT_STATUS_ADJUSTED, 0))
        logger.info(
            "PTM denominator resolution: %d / %d sites adjustable (%.1f%%)",
            adjustable,
            total_sites,
            100.0 * adjustable / total_sites if total_sites else 0.0,
        )
        for status, count in counts.items():
            if status != ADJUSTMENT_STATUS_ADJUSTED:
                logger.info("  unadjusted [%s]: %d", status, count)

        if counts.get(ADJUSTMENT_STATUS_NOT_IN_GLOBAL_FASTA, 0):
            logger.warning(
                "%d sites have no accession in the global dataset's FASTA. The PTM and global "
                "searches appear to use different protein databases.",
                int(counts[ADJUSTMENT_STATUS_NOT_IN_GLOBAL_FASTA]),
            )
        if adjustable == 0:
            logger.warning(
                "No PTM site could be matched to a quantified global protein group; nothing will be "
                "adjusted. Check that both datasets were searched against the same FASTA and that "
                "the global dataset has been summarised to protein level."
            )

    # ------------------------------------------------------------------ data extraction

    def _extract_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Assemble the adjustable sites and the aligned global denominator values."""
        ptm_adata = get_anndata_mod(self.ptm_mdata, self.ptm_mod)
        global_adata = get_anndata_mod(self.global_mdata, self.global_mod)

        missing_samples = [sample for sample in self.sample_cols if sample not in set(global_adata.obs_names)]
        if missing_samples:
            raise ValueError(
                f"Global dataset is missing samples present in the PTM data: {missing_samples}. "
                "PTM adjustment requires both datasets to cover the same samples."
            )

        adjustable_mask = self.resolution["adjustment_status"] == ADJUSTMENT_STATUS_ADJUSTED

        ptm_data: pd.DataFrame = to_dense_df(ptm_adata).T.copy()
        ptm_data = ptm_data.loc[adjustable_mask.values]
        ptm_data["ptm_site"] = ptm_data.index
        ptm_data["denominator_group"] = self.resolution.loc[adjustable_mask, "denominator_group"].values

        global_data: pd.DataFrame = to_dense_df(global_adata).T.copy()
        global_data = global_data[self.sample_cols]  # sort sample order

        return ptm_data, global_data

    # ------------------------------------------------------------------ estimators

    def _ratio(self) -> pd.DataFrame:
        """Subtract the parent protein's level from the site's, assuming a slope of one.

        Mass action says a doubled protein at fixed occupancy doubles the site, so slope one is the
        principled default; it also needs no fitting, which is what keeps it usable at the sample
        counts proteomics actually has. This is the estimator used by msqrob2PTM.
        """
        ptm_values = self.ptm_data[self.sample_cols]
        global_values = self.global_data.loc[self.ptm_data["denominator_group"], self.sample_cols]

        result_df = self.ptm_data.copy()
        result_df[self.sample_cols] = ptm_values.to_numpy() - global_values.to_numpy()

        return result_df

    def _ridge(self, alpha: float = DEFAULT_RIDGE_ALPHA) -> pd.DataFrame:
        """Regress each site on its parent protein and keep the residual.

        Relaxes the slope-one assumption at the cost of estimating a slope from as many points as
        there are samples. NOTE: the ridge penalty shrinks that slope toward zero by a factor
        ``Sxx / (Sxx + alpha)`` where ``Sxx = (n - 1) * var(protein)``; with proteomics-scale
        variance and sample counts, a large alpha leaves the residual indistinguishable from simply
        centring the site, i.e. no protein correction at all. Choose alpha with that in mind.
        """
        records: list = list()
        skipped_for_few_pairs = 0

        for group_id, site_rows in self.ptm_data.groupby("denominator_group", sort=False, observed=True):
            protein_values: np.ndarray = self.global_data.loc[group_id, self.sample_cols].to_numpy(float)
            for _, row in site_rows.iterrows():
                site_values: np.ndarray = row[self.sample_cols].to_numpy(float)

                paired_mask: np.ndarray = ~np.isnan(protein_values) & ~np.isnan(site_values)
                if paired_mask.sum() <= MIN_RIDGE_PAIRED_SAMPLES:
                    skipped_for_few_pairs += 1
                    continue

                model = Ridge(alpha=alpha, fit_intercept=True).fit(
                    protein_values[paired_mask].reshape(-1, 1),
                    site_values[paired_mask],
                )

                fitted: np.ndarray = np.full_like(site_values, np.nan, dtype=float)
                fitted[paired_mask] = model.predict(protein_values[paired_mask].reshape(-1, 1))

                records.append(
                    {
                        "ptm_site": row["ptm_site"],
                        "denominator_group": group_id,
                        "residual": site_values - fitted,
                    }
                )

        if skipped_for_few_pairs:
            logger.info(
                "ridge: %d sites left unadjusted for having %d or fewer paired observations.",
                skipped_for_few_pairs,
                MIN_RIDGE_PAIRED_SAMPLES,
            )

        if not records:
            return self.ptm_data.iloc[0:0].copy()

        result_df: pd.DataFrame = pd.DataFrame(records)
        residual_values = pd.DataFrame(result_df["residual"].tolist(), columns=self.sample_cols)
        result_df = pd.concat([result_df.drop(columns="residual"), residual_values], axis=1)

        return result_df

    # ------------------------------------------------------------------ output

    def _rescale(self, adjusted_ptm: pd.DataFrame) -> pd.DataFrame:
        """Shift the residuals back onto a plausible intensity scale.

        A single constant added to every site and sample, so it cancels in any between-condition
        contrast; it exists so the adjusted values are readable next to the unadjusted ones.
        """
        total_median: float = np.nanmedian(self.ptm_data[self.sample_cols].to_numpy().flatten())
        adjusted_ptm[self.sample_cols] = adjusted_ptm[self.sample_cols] + total_median

        return adjusted_ptm

    def _write_back(self, adjusted_ptm: pd.DataFrame, layer: str) -> md.MuData:
        """Store adjusted values in a layer and annotate every site with how it was resolved.

        The unadjusted matrix stays in ``.X`` and no site is dropped. Residuals and raw abundances
        are different quantities, so leaving them in one matrix would let downstream testing compare
        them along the same axis with no way to tell them apart; keeping DPU beside DPA rather than
        replacing it is also what the literature reports.
        """
        adj_ptm_mdata: md.MuData = self.ptm_mdata.copy()
        adj_ptm_adata = get_anndata_mod(adj_ptm_mdata, self.ptm_mod).copy()

        adjusted_matrix = pd.DataFrame(
            np.nan,
            index=adj_ptm_adata.obs_names,
            columns=adj_ptm_adata.var_names,
            dtype=float,
        )
        if len(adjusted_ptm):
            adjusted_values = adjusted_ptm.set_index("ptm_site")[self.sample_cols]
            adjusted_matrix.loc[adjusted_values.columns, adjusted_values.index] = adjusted_values.T.to_numpy()

        produced_values = ~adjusted_matrix.isna().all(axis=0).to_numpy()

        # A site can resolve to a denominator and still yield nothing -- ridge drops any site with too
        # few paired observations to fit a slope. Reporting it as "adjusted" would make the status
        # column disagree with the layer, so the estimator's own refusal is recorded here instead.
        final_status = self.resolution["adjustment_status"].reindex(adj_ptm_adata.var_names)
        resolved_but_empty = (final_status == ADJUSTMENT_STATUS_ADJUSTED) & ~produced_values
        final_status = final_status.mask(resolved_but_empty, ADJUSTMENT_STATUS_NO_ESTIMATE)
        if resolved_but_empty.any():
            logger.info(
                "  unadjusted [%s]: %d (denominator found, but the estimator produced no value)",
                ADJUSTMENT_STATUS_NO_ESTIMATE,
                int(resolved_but_empty.sum()),
            )

        adj_ptm_adata.layers[layer] = adjusted_matrix.to_numpy()
        adj_ptm_adata.var["denominator_group"] = self.resolution["denominator_group"].reindex(adj_ptm_adata.var_names)
        adj_ptm_adata.var["adjustment_status"] = final_status
        adj_ptm_adata.var["is_protein_adjusted"] = produced_values

        get_mudata_mod_as_mutable(adj_ptm_mdata)[self.ptm_mod] = adj_ptm_adata.copy()
        adj_ptm_mdata.update()

        logger.info(
            "Protein-adjusted values written to %s.layers['%s']; .X still holds unadjusted intensities.",
            self.ptm_mod,
            layer,
        )

        return adj_ptm_mdata

    def adjust(self, method: str, rescale: bool, layer: str, alpha: float | None = None) -> md.MuData:
        if method not in _PTM_ADJUSTMENT_METHODS:
            raise ValueError(f"Unknown PTM adjustment method '{method}'. Choose from {_PTM_ADJUSTMENT_METHODS}.")

        if method == "ridge":
            adjusted_ptm = self._ridge(alpha=DEFAULT_RIDGE_ALPHA if alpha is None else alpha)
        else:
            adjusted_ptm = self._ratio()

        if rescale and len(adjusted_ptm):
            adjusted_ptm = self._rescale(adjusted_ptm)

        return self._write_back(adjusted_ptm, layer=layer)
