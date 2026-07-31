from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import mudata as md
import numpy as np
import pandas as pd

from .._utils._mudata import get_anndata_mod
from .._core._blockdiag import to_dense_df
from ..logging_utils import get_logger
from .._statistics._permutation import PermutationTest
from .._statistics._fc_threshold import compute_fc_guidance_line
from .._statistics._de_base import (
    StatTestResult,
    DeaResult,
    DeaValidator,
)
from .._statistics._statistics import (
    _measure_central_tendency,
    _calc_log2fc,
    _get_pct_expression,
)
from .._statistics._limma import (
    LimmaContrast,
    build_contrast,
    cell_blocks_for_contrast,
    fit_limma,
)
from .._statistics._multiple_test_correction import P_ADJUST_METHODS


logger = get_logger(__name__)

_PERMUTATION_METHODS = ("welch", "student", "wilcoxon")
_STAT_METHODS = _PERMUTATION_METHODS + ("limma",)

# Shuffle count for the engine-independent fold-change guidance line. Only limma reaches the
# standalone computation (the permutation engines reuse their own shuffle byproduct), and limma
# never permutes for its test, so the guidance line uses this fixed count independent of the
# caller's ``n_resamples`` — keeping limma results reproducible regardless of that permutation knob.
_FC_GUIDANCE_LINE_RESAMPLES = 1000

# BID-71 transition notice: the default DE engine changed to limma. Shown once per process while
# limma runs, so callers who relied on the old permutation default notice the change. A literal
# default (not a sentinel) can't tell an explicit "limma" from the default, so this fires for any
# limma run — once per session keeps it unobtrusive.
# TODO(BID-71): remove this notice in the release after the one that ships the new default.
_default_engine_notice_shown = False


def _notify_default_engine_transition(stat_method: str) -> None:
    global _default_engine_notice_shown
    if stat_method != "limma" or _default_engine_notice_shown:
        return
    _default_engine_notice_shown = True
    logger.warning(
        "mm.tl.run_de: the default DE engine is now 'limma' (previously a permutation test). "
        "Pass stat_method explicitly to choose an engine. This notice will be removed in a future release."
    )


@dataclass(frozen=True)
class DeaInputs:
    """Prepared inputs for one DE comparison, read from the MuData exactly once.

    The single place the modality is pulled out of the container, so both engines consume the same
    arrays instead of re-reading it. Holds the features x samples matrix and ``obs`` (limma's
    design needs per-sample metadata) alongside the two group arrays (the permutation test and the
    engine-independent fold-change / guidance steps).

    Attributes:
        expr_matrix: features x samples log2-intensity DataFrame (missing = NaN).
        obs: sample metadata, index aligned to ``expr_matrix`` columns.
        features: feature (var) names, aligned to result arrays.
        ctrl_arr: control group data (n_ctrl_samples x n_features).
        expr_arr: experimental group data (n_expr_samples x n_features).
        ctrl_label: control group name.
        expr_label: experimental group name ("all_other_groups" when ``expr`` was None).
    """

    expr_matrix: pd.DataFrame
    obs: pd.DataFrame
    features: np.ndarray
    ctrl_arr: np.ndarray
    expr_arr: np.ndarray
    ctrl_label: str
    expr_label: str

    @classmethod
    def from_mudata(
        cls,
        mdata: md.MuData,
        modality: str,
        category: str,
        ctrl: str,
        expr: str | None,
        layer: str | None,
    ) -> "DeaInputs":
        mod_adata = get_anndata_mod(mdata, modality)
        # to_dense_df restores absent cells as NaN for a sparse .X/layer (a plain DataFrame over
        # the raw sparse matrix crashes, and a densify would poison absent cells with 0).
        data = to_dense_df(mod_adata, layer=layer)

        expr_matrix = data.T  # features x samples (single transpose, reused below)

        obs = mod_adata.obs
        ctrl_samples = obs.loc[obs[category] == ctrl].index.to_list()
        if expr is not None:
            expr_samples = obs.loc[obs[category] == expr].index.to_list()
            expr_label = expr
        else:
            expr_samples = obs.loc[obs[category] != ctrl].index.to_list()
            expr_label = "all_other_groups"

        return cls(
            expr_matrix=expr_matrix,
            obs=obs,
            features=mod_adata.var.index.to_numpy(),
            ctrl_arr=expr_matrix[ctrl_samples].values.T,
            expr_arr=expr_matrix[expr_samples].values.T,
            ctrl_label=ctrl,
            expr_label=expr_label,
        )


@dataclass
class DeaValidation:
    """An engine's pre-flight result: the usable-feature set plus feasibility.

    ``feature_mask`` marks the features the engine can test (and that the guidance line uses).
    ``has_enough_samples`` is the permutation path's sample-size feasibility. ``contrast`` carries
    limma's means-model contrast forward to the fit so it is built only once.
    """

    feature_mask: np.ndarray
    has_enough_samples: bool
    contrast: LimmaContrast | None = None


class DeEngine(ABC):
    """Strategy for one differential-expression test engine.

    ``validate`` produces the engine's usable-feature mask and feasibility from the prepared
    inputs; ``test`` runs the engine's statistic on the validated data. ``effect_measure`` is the
    central tendency the engine's fold change is expressed in — the engine-independent fold-change
    and guidance steps follow it (the permutation path uses the caller's ``measure``; limma's
    contrast is mean-based).

    ``provides_comparable_fold_change`` is True when the engine's reported log2 fold change is a
    plain two-group difference, on the same scale as the raw label-permutation FC null — the
    condition for an automatic guidance line to be meaningful. It is False for limma interaction
    (difference-in-differences) and covariate-adjusted contrasts, whose reported log2fc is not the
    raw group difference the null is built from.
    """

    effect_measure: str
    provides_comparable_fold_change: bool

    @abstractmethod
    def validate(self, inputs: DeaInputs) -> DeaValidation: ...

    @abstractmethod
    def test(self, inputs: DeaInputs, validation: DeaValidation) -> DeaResult: ...


class PermutationEngine(DeEngine):
    """Label-permutation DE test over the two groups (welch / student / wilcoxon statistic).

    Validation masks features below ``min_pct`` coverage on the two groups (no model is fitted, so
    residual degrees of freedom are not required). The test always runs the permutation test — the
    parametric alternative is limma, not a special mode of this engine — and its p-values are
    FDR-corrected by ``fdr`` (default ``"empirical"``, the permutation FDR built from the null; or
    any R p.adjust method applied to the permutation p-values). The fold-change guidance line falls
    out of the shuffles as a byproduct, so it is set here and not recomputed downstream.

    ``effect_measure`` follows the statistic's location so significance and effect size stay on the
    same central tendency: welch / student test the mean (mean-based fold change), wilcoxon the
    rank/median (median-based). For wilcoxon the reported fold change is the difference of group
    medians — a pragmatic proxy for the Hodges-Lehmann shift the rank-sum statistic localizes (they
    coincide under a pure location shift).
    """

    # The permutation log2fc is the raw two-group difference, so its guidance line is always comparable.
    provides_comparable_fold_change = True

    def __init__(
        self,
        stat_method: str,
        n_resamples: int,
        log_transformed: bool,
        min_pct: float,
        force_resample: bool,
        p_adjust: str,
    ) -> None:
        self.stat_method = stat_method
        self.effect_measure = "mean" if stat_method in ("welch", "student") else "median"
        self.n_resamples = n_resamples
        self.log_transformed = log_transformed
        self.min_pct = min_pct
        self.force_resample = force_resample
        # Concrete correction for the q-value (already resolved from "auto" to "empirical"):
        # "empirical" (FDR from the null) or an R p.adjust method applied to the permutation p-values.
        # Named p_adjust to match the public run_de knob; passed to PermutationTest as its ``fdr``.
        self.p_adjust = p_adjust

    def validate(self, inputs: DeaInputs) -> DeaValidation:
        validator = DeaValidator(
            [inputs.ctrl_arr.T, inputs.expr_arr.T], self.min_pct, require_residual_df=False
        )
        logger.debug(
            "DEA feature mask retained %d of %d features.",
            int(np.sum(validator.feature_mask)),
            int(validator.feature_mask.size),
        )
        return DeaValidation(feature_mask=validator.feature_mask, has_enough_samples=validator.has_enough_samples)

    def test(self, inputs: DeaInputs, validation: DeaValidation) -> DeaResult:
        if not validation.has_enough_samples:
            logger.warning("Not enough samples to perform DEA. Returning result with only fold changes.")
            return DeaResult(_make_dummy_de_result(inputs.features.size))

        valid_ctrl_arr = inputs.ctrl_arr.copy()
        valid_ctrl_arr[:, ~validation.feature_mask] = np.nan
        valid_expr_arr = inputs.expr_arr.copy()
        valid_expr_arr[:, ~validation.feature_mask] = np.nan

        logger.debug("Running permutation DEA with %s resamples (%s FDR).", self.n_resamples, self.p_adjust)
        perm_test = PermutationTest(
            ctrl_arr=valid_ctrl_arr,
            expr_arr=valid_expr_arr,
            n_resamples=self.n_resamples,
            _force_resample=self.force_resample,
            fdr=self.p_adjust,
        )
        test_res = perm_test.run(
            n_permutations=self.n_resamples,
            stat_method=self.stat_method,
            measure=self.effect_measure,
            log_transformed=self.log_transformed,
        )
        return DeaResult(test_res)


class LimmaEngine(DeEngine):
    """limma moderated-t DE test for a Level 1 (main effect) or Level 2 (interaction) contrast.

    Validation builds the means-model contrast and the estimable-feature mask (every design cell
    needs coverage AND the per-feature fit needs residual degrees of freedom). The test fits the
    validated contrast. The model log2 fold change is the contrast coefficient — an intrinsic,
    mean-based output of the fit — so ``effect_measure`` is fixed to "mean" (limma is the parametric
    engine; the permutation engines carry the median option).
    """

    effect_measure = "mean"

    def __init__(
        self,
        category: str,
        ctrl: str,
        expr: str,
        interaction: str | None,
        interaction_levels: list | None,
        covariates: list[str] | None,
        min_pct: float,
        p_adjust: str,
    ) -> None:
        self.category = category
        self.ctrl = ctrl
        self.expr = expr
        self.interaction = interaction
        self.interaction_levels = interaction_levels
        self.covariates = covariates
        self.min_pct = min_pct
        # Concrete R p.adjust method for the q-value (already resolved from "auto" to "bh").
        self.p_adjust = p_adjust

    @property
    def provides_comparable_fold_change(self) -> bool:
        # Only a plain two-group main effect reports the raw group difference the label-permutation
        # FC null is built from. An interaction (difference-in-differences) reports a DiD, and a
        # covariate-adjusted contrast reports an adjusted coefficient — both on a different scale
        # than the raw null — so no automatic guidance line is drawn for them.
        return self.interaction is None and self.interaction_levels is None and not self.covariates

    def validate(self, inputs: DeaInputs) -> DeaValidation:
        contrast = build_contrast(
            obs=inputs.obs,
            category=self.category,
            ctrl=self.ctrl,
            expr=self.expr,
            interaction=self.interaction,
            interaction_levels=self.interaction_levels,
            covariates=self.covariates,
        )
        fit_matrix = inputs.expr_matrix.loc[:, contrast.kept_samples]
        validator = DeaValidator(
            cell_blocks_for_contrast(fit_matrix, contrast), self.min_pct, require_residual_df=True
        )
        n_estimable = int(np.sum(validator.feature_mask))
        logger.debug(
            "limma contrast '%s': %d/%d features estimable after min_pct=%.2f filter.",
            contrast.label,
            n_estimable,
            validator.feature_mask.size,
            self.min_pct,
        )
        if n_estimable == 0:
            raise ValueError(
                f"No features pass the min_pct={self.min_pct} coverage filter for contrast '{contrast.label}'."
            )
        return DeaValidation(
            feature_mask=validator.feature_mask,
            has_enough_samples=validator.has_enough_samples,
            contrast=contrast,
        )

    def test(self, inputs: DeaInputs, validation: DeaValidation) -> DeaResult:
        contrast = validation.contrast
        fit_matrix = inputs.expr_matrix.loc[:, contrast.kept_samples]
        result = fit_limma(fit_matrix, contrast, validation.feature_mask, p_adjust=self.p_adjust)
        logger.debug(
            "limma DEA contrast '%s' produced %d features.", result.contrast_label, result.features.size
        )

        de_res = DeaResult(
            StatTestResult(
                stat_method="limma",
                statistic=result.moderated_t,
                p_value=result.p_value,
                q_value=result.q_value,
            )
        )
        de_res.log2fc = result.log2fc
        de_res.contrast_label = result.contrast_label
        return de_res


def run_de(
    mdata: md.MuData,
    modality: str,
    category: str,
    ctrl: str,
    expr: str | None = None,
    min_pct: float = 0.0,
    layer: str | None = None,
    stat_method: Literal["welch", "student", "wilcoxon", "limma"] = "limma",
    n_resamples: int = 1000,
    log_transformed: bool = True,
    interaction: str | None = None,
    interaction_levels: list | None = None,
    covariates: list[str] | None = None,
    p_adjust: str = "auto",
    _force_resample: bool = False,
) -> DeaResult:
    """
    Run Differential Expression Analysis (DEA) between two groups in a MuData object.

    The analysis reads as four stages: (1) data validation — prepare the inputs once and let the
    engine mask the usable features; (2) test — the engine-specific statistic; (3) fold change —
    engine-independent group centres and the log2 fold change; (4) fold-change guidance line.

    Parameters:
        mdata: MuData object containing the data.
        modality: Modality name within the MuData to analyze.
        category: Observation category to define groups.
        ctrl: Name of the control group.
        expr: Name of the experimental group. If None, all other groups are used
            (not supported for stat_method="limma", which needs an explicit group).
        layer: Layer to use for quantification aggregation. If None, the default layer (.X) will be used. Defaults to None.
        min_pct: Minimum non-missing coverage required in **every** group (design cell), not in at
            least one, applied as a count: a feature needs ``max(1, ceil(min_pct * n))`` non-missing
            values in each group. The default ``0.0`` imposes only the estimability floor — every
            group needs at least one non-missing value (limma additionally requires residual
            df >= 1) — so every feature whose contrast can be estimated is tested. Raise it as an
            opt-in stringency knob to also demand a minimum coverage per group (e.g. ``0.5`` is 2 of
            3, 2 of 4, 3 of 5); choose it from the coverage you observe (``pct_ctrl`` / ``pct_expr``),
            not from which features come out significant. Features below the threshold are not tested:
            they stay in the result as rows with ``repr_*`` / ``pct_*`` filled but ``p_value`` /
            ``q_value`` set to NaN. Requiring both groups is what makes the contrast estimable
            without imputation; the consequence is that on/off features (present in one group,
            absent in the other) are reported this way rather than receiving a p-value — read them
            off ``pct_ctrl`` / ``pct_expr``.
        stat_method: Statistical test to use. Defaults to "limma" (empirical-Bayes moderated-t,
            recommended for the small sample sizes where a permutation null is degenerate). The
            permutation engines "welch"/"student"/"wilcoxon" always run a label-permutation test.
            The fold-change central tendency follows the test: welch/student/limma are mean-based,
            wilcoxon median-based (so significance and effect size stay on the same scale). For
            wilcoxon the fold change is the median difference, a pragmatic proxy for the
            Hodges-Lehmann shift the rank-sum statistic localizes.
        n_resamples: Number of label permutations for the permutation engines (welch/student/
            wilcoxon); must be a positive integer (e.g. 1000). Ignored by limma (which does not
            permute). It is not an on/off switch — for a parametric analysis use stat_method="limma".
        log_transformed: If True, data is assumed to be log-transformed. Defaults to True.
        interaction: limma only — obs column of a second factor. If set, tests the
            interaction (difference-in-differences) of ``expr - ctrl`` across two of its levels.
        interaction_levels: limma only — the two ``interaction`` levels to contrast.
        covariates: limma only — obs columns to adjust for.
        p_adjust: Multiple-testing correction for the q-value. Default ``"auto"`` uses each engine's
            native default: limma adjusts its moderated p-values with Benjamini-Hochberg ("bh"), the
            permutation engines use ``"empirical"`` (an FDR built from the label-permutation null).
            The R ``p.adjust`` / limma ``adjust.method`` family — ``"bh"``, ``"by"``, ``"holm"``,
            ``"hochberg"``, ``"hommel"``, ``"bonferroni"`` (case-insensitive) — is accepted by both
            engines and applied identically to their p-values. ``"empirical"`` is permutation-only
            (limma has no null distribution to build it from, so limma + ``"empirical"`` raises).
            There is no "no correction" option — the uncorrected values are the ``p_value`` column.
        _force_resample: If True, forces resampling even if the number of resamples exceeds the number of combinations.

    Returns:
        DeaResult containing DE analysis results.
    """
    # Notify before validating so a migrating expr=None caller (old default "vs all other groups")
    # learns the default engine changed before hitting limma's "explicit expr required" error.
    _notify_default_engine_transition(stat_method)
    _validate_run_de_args(stat_method, expr, interaction, interaction_levels, covariates, n_resamples, p_adjust)

    # 1. Data validation: read the modality once, then let the engine validate the features. The
    #    engines' feasibility checks are not interchangeable (the permutation path validates sample
    #    size + group coverage; limma validates its design levels and per-feature estimability), so
    #    each engine owns its own validate() while sharing the single min_pct rule (DeaValidator).
    inputs = DeaInputs.from_mudata(
        mdata=mdata, modality=modality, category=category, ctrl=ctrl, expr=expr, layer=layer
    )
    engine = _select_engine(
        stat_method,
        n_resamples=n_resamples,
        log_transformed=log_transformed,
        force_resample=_force_resample,
        min_pct=min_pct,
        category=category,
        ctrl=ctrl,
        expr=expr,
        interaction=interaction,
        interaction_levels=interaction_levels,
        covariates=covariates,
        p_adjust=p_adjust,
    )
    validation = engine.validate(inputs)
    logger.debug(
        "Prepared DEA for modality '%s': ctrl=%s expr=%s stat=%s effect_measure=%s.",
        modality,
        inputs.ctrl_arr.shape,
        inputs.expr_arr.shape,
        stat_method,
        engine.effect_measure,
    )

    # 2. Statistical test (engine-specific).
    result = engine.test(inputs, validation)

    # 3. Fold change (engine-independent): labels, features, group centres/detection, and the log2
    #    fold change when the engine did not already report it (limma reports the model contrast).
    _attach_fold_change(result, inputs, engine.effect_measure, log_transformed)

    # 4. Fold-change guidance line (engine-independent). Drawn only when the engine's log2fc is a
    #    plain two-group difference comparable to the raw label-permutation null (not for limma
    #    interaction / covariate-adjusted contrasts), and skipped when the engine already produced
    #    it (the permutation shuffle byproduct), so it never shuffles twice.
    _attach_fc_guidance_line(
        result,
        inputs,
        validation,
        engine.effect_measure,
        log_transformed=log_transformed,
        draw_guidance_line=engine.provides_comparable_fold_change,
    )

    return result


def _validate_run_de_args(
    stat_method: str,
    expr: str | None,
    interaction: str | None,
    interaction_levels: list | None,
    covariates: list[str] | None,
    n_resamples: int,
    p_adjust: str,
) -> None:
    if stat_method not in _STAT_METHODS:
        raise ValueError(
            f"Invalid statistic: {stat_method}. Choose from 'welch', 'student', 'wilcoxon', 'limma'."
        )
    _validate_p_adjust(stat_method, p_adjust)
    if stat_method == "limma":
        if expr is None:
            raise ValueError(
                "stat_method='limma' requires an explicit 'expr' group (expr=None is not supported). "
                "For a comparison against all other groups (expr=None), use a permutation method such "
                "as stat_method='welch'."
            )
        return
    # permutation family (welch / student / wilcoxon)
    if interaction is not None or interaction_levels is not None or covariates is not None:
        raise ValueError(
            "'interaction', 'interaction_levels' and 'covariates' are only supported with stat_method='limma'."
        )
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError(
            f"stat_method='{stat_method}' always uses a permutation test; n_resamples is the number of "
            f"label shuffles, not an on/off switch (n_resamples={n_resamples!r} does not disable it). "
            f"Pass a positive integer (e.g. n_resamples=1000), or stat_method='limma' for a parametric test."
        )


def _validate_p_adjust(stat_method: str, p_adjust: str) -> None:
    """Validate ``p_adjust`` against the engine's supported multiple-testing corrections.

    Both engines share the R p.adjust family (``P_ADJUST_METHODS``) and accept ``"auto"`` (engine
    default). The permutation engines additionally accept ``"empirical"`` — the FDR built from the
    label-permutation null — which limma cannot produce (it has no null), so limma + ``"empirical"``
    is reported as a clear, engine-specific error rather than a generic "unknown method".
    """
    if not isinstance(p_adjust, str):
        raise ValueError(f"p_adjust must be a string, got {type(p_adjust).__name__}.")
    method_key = p_adjust.lower()
    if method_key in P_ADJUST_METHODS or method_key == "auto":
        return
    if stat_method == "limma":
        if method_key == "empirical":
            raise ValueError(
                "p_adjust='empirical' is the permutation engines' FDR, built from the "
                "label-permutation null; stat_method='limma' has no null to build it from. Choose a "
                f"p-value adjustment method {list(P_ADJUST_METHODS)}, or use a permutation method "
                "(e.g. stat_method='welch') for empirical FDR."
            )
        raise ValueError(
            f"Unknown p_adjust={p_adjust!r} for stat_method='limma'. "
            f"Choose from {list(P_ADJUST_METHODS)} or 'auto' (= 'bh')."
        )
    # permutation family (welch / student / wilcoxon)
    if method_key == "empirical":
        return
    raise ValueError(
        f"Unknown p_adjust={p_adjust!r} for stat_method='{stat_method}'. "
        f"Choose from {list(P_ADJUST_METHODS)}, 'empirical', or 'auto' (= 'empirical')."
    )


def _resolve_p_adjust(stat_method: str, p_adjust: str) -> str:
    """Resolve ``"auto"`` to the engine's native default and lowercase the method name.

    Assumes ``p_adjust`` already passed :func:`_validate_p_adjust`. ``"auto"`` becomes ``"bh"`` for
    limma and ``"empirical"`` for the permutation engines; any explicit method is returned lowercased.
    """
    method_key = p_adjust.lower()
    if method_key == "auto":
        return "bh" if stat_method == "limma" else "empirical"
    return method_key


def _select_engine(
    stat_method: str,
    *,
    n_resamples: int,
    log_transformed: bool,
    force_resample: bool,
    min_pct: float,
    category: str,
    ctrl: str,
    expr: str | None,
    interaction: str | None,
    interaction_levels: list | None,
    covariates: list[str] | None,
    p_adjust: str,
) -> DeEngine:
    """Pick the DE engine for ``stat_method``, passing each only the knobs it uses."""
    correction = _resolve_p_adjust(stat_method, p_adjust)
    if stat_method == "limma":
        return LimmaEngine(
            category=category,
            ctrl=ctrl,
            expr=expr,
            interaction=interaction,
            interaction_levels=interaction_levels,
            covariates=covariates,
            min_pct=min_pct,
            p_adjust=correction,
        )
    return PermutationEngine(
        stat_method=stat_method,
        n_resamples=n_resamples,
        log_transformed=log_transformed,
        min_pct=min_pct,
        force_resample=force_resample,
        p_adjust=correction,
    )


def _attach_fold_change(
    result: DeaResult,
    inputs: DeaInputs,
    effect_measure: str,
    log_transformed: bool,
) -> None:
    """Fill the engine-independent fold-change fields of ``result`` in place.

    Group labels, features, group centres (``repr``) and detection percentages are always set from
    the prepared inputs, in the engine's ``effect_measure``. The log2 fold change is derived from
    the group centres here only when the engine did not report one: limma reports the model
    contrast (an intrinsic, mean-based test output), while the permutation path gets its
    effect_measure-based difference at this step.
    """
    result.ctrl = inputs.ctrl_label
    result.expr = inputs.expr_label
    result.features = inputs.features
    result.repr_ctrl = _measure_central_tendency(inputs.ctrl_arr, effect_measure)
    result.repr_expr = _measure_central_tendency(inputs.expr_arr, effect_measure)
    result.pct_ctrl = _get_pct_expression(inputs.ctrl_arr)
    result.pct_expr = _get_pct_expression(inputs.expr_arr)

    engine_reported_log2fc = result.log2fc is not None
    if not engine_reported_log2fc:
        result.log2fc = _calc_log2fc(result.repr_ctrl, result.repr_expr, log_transformed=log_transformed)


def _attach_fc_guidance_line(
    result: DeaResult,
    inputs: DeaInputs,
    validation: DeaValidation,
    effect_measure: str,
    log_transformed: bool,
    draw_guidance_line: bool,
) -> None:
    """Set the fold-change guidance line (``fc_pct_1``/``fc_pct_5``) of ``result`` in place.

    Skipped when ``draw_guidance_line`` is False — the reported log2fc is not a raw two-group
    difference the label-permutation null can match (limma interaction / covariate-adjusted
    contrasts) — and when the engine already produced the line (the permutation shuffle byproduct),
    so the null is never built twice. limma main effects compute it here from the same
    label-permutation log2FC null, reusing the engine's usable-feature mask and its ``effect_measure``.
    """
    if not draw_guidance_line or result.fc_pct_5 is not None:
        return
    # Only limma reaches here (the permutation engine reuses its own shuffle byproduct), and it does
    # not permute for its test, so the guidance line uses a fixed shuffle count — not the caller's
    # n_resamples — so limma output does not depend on that permutation-only knob.
    result.fc_pct_1, result.fc_pct_5 = compute_fc_guidance_line(
        inputs.ctrl_arr,
        inputs.expr_arr,
        validation.feature_mask,
        measure=effect_measure,
        log_transformed=log_transformed,
        n_resamples=_FC_GUIDANCE_LINE_RESAMPLES,
        force_resample=False,
    )


def _make_dummy_de_result(n_features: int) -> StatTestResult:
    """Significance-free result for a design too small to test.

    The statistic / p / q are all NaN at full feature length (not empty) so the fold changes that
    ``run_de`` still fills in remain aligned with the feature axis and ``to_df`` / ``plot_volcano``
    do not raise.
    """
    nan_per_feature = np.full(n_features, np.nan)
    return StatTestResult(
        stat_method="",
        statistic=nan_per_feature.copy(),
        p_value=nan_per_feature.copy(),
        q_value=nan_per_feature.copy(),
    )
