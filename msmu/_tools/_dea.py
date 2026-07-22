from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import mudata as md
import numpy as np
import pandas as pd

from .._utils._mudata import get_anndata_mod
from ..logging_utils import get_logger
from .._statistics._permutation import PermutationTest
from .._statistics._fc_threshold import compute_fc_guidance_line
from .._statistics._de_base import (
    StatTestResult,
    DeaResult,
    DeaValidator,
)
from .._statistics._statistics import (
    simple_test,
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


logger = get_logger(__name__)


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
        if layer is not None:
            data = pd.DataFrame(
                mod_adata.layers[layer],
                index=mod_adata.obs_names,
                columns=mod_adata.var_names,
            )
        else:
            data = mod_adata.to_df()

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
    """Label-permutation (or parametric two-sample) DE test over the two groups.

    Validation masks features below ``min_pct`` coverage on the two groups (no model is fitted, so
    residual degrees of freedom are not required). The test runs the permutation test, or the
    parametric ``simple_test`` when ``n_resamples`` is None. The fold-change guidance line falls
    out of the permutation shuffles as a byproduct, so it is set here and not recomputed downstream.
    """

    # The permutation log2fc is the raw two-group difference, so its guidance line is always comparable.
    provides_comparable_fold_change = True

    def __init__(
        self,
        stat_method: str,
        measure: str,
        n_resamples: int | None,
        fdr: bool | str,
        log_transformed: bool,
        min_pct: float,
        force_resample: bool,
    ) -> None:
        self.stat_method = stat_method
        self.effect_measure = measure
        self.n_resamples = n_resamples
        self.fdr = fdr
        self.log_transformed = log_transformed
        self.min_pct = min_pct
        self.force_resample = force_resample

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

        if self.n_resamples is not None:
            logger.debug("Running permutation-based DEA with %s resamples and fdr=%s.", self.n_resamples, self.fdr)
            perm_test = PermutationTest(
                ctrl_arr=valid_ctrl_arr,
                expr_arr=valid_expr_arr,
                n_resamples=self.n_resamples,
                _force_resample=self.force_resample,
                fdr=self.fdr,
            )
            test_res = perm_test.run(
                n_permutations=self.n_resamples,
                stat_method=self.stat_method,
                measure=self.effect_measure,
                log_transformed=self.log_transformed,
            )
        else:
            logger.debug("Running simple DEA without resampling and fdr=%s.", self.fdr)
            test_res = simple_test(
                ctrl=valid_ctrl_arr, expr=valid_expr_arr, stat_method=self.stat_method, fdr=self.fdr
            )

        return DeaResult(test_res)


class LimmaEngine(DeEngine):
    """limma moderated-t DE test for a Level 1 (main effect) or Level 2 (interaction) contrast.

    Validation builds the means-model contrast and the estimable-feature mask (every design cell
    needs coverage AND the per-feature fit needs residual degrees of freedom). The test fits the
    validated contrast. The model log2 fold change is the contrast coefficient — an intrinsic,
    mean-based output of the fit — so ``effect_measure`` is fixed to "mean" and the caller's
    ``measure`` is ignored.
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
    ) -> None:
        self.category = category
        self.ctrl = ctrl
        self.expr = expr
        self.interaction = interaction
        self.interaction_levels = interaction_levels
        self.covariates = covariates
        self.min_pct = min_pct

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
        result = fit_limma(fit_matrix, contrast, validation.feature_mask)
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
    min_pct: float = 0.5,
    layer: str | None = None,
    stat_method: Literal["welch", "student", "wilcoxon", "limma"] = "welch",
    measure: Literal["median", "mean"] = "median",
    n_resamples: int | None = 1000,
    fdr: bool | Literal["empirical", "bh"] = "empirical",
    log_transformed: bool = True,
    interaction: str | None = None,
    interaction_levels: list | None = None,
    covariates: list[str] | None = None,
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
        stat_method: Statistical test to use ("welch", "student", "wilcoxon", "limma").
        measure: Measure of central tendency for the fold change ("median" or "mean"). Applies to
            the permutation / simple engines; limma's fold change is its mean-based model contrast.
        n_resamples: Number of resamples for permutation test. If None, no permutation test is performed.
        fdr: Method for multiple test correction ("empirical", "bh", or False).
        log_transformed: If True, data is assumed to be log-transformed. Defaults to True.
        interaction: limma only — obs column of a second factor. If set, tests the
            interaction (difference-in-differences) of ``expr - ctrl`` across two of its levels.
        interaction_levels: limma only — the two ``interaction`` levels to contrast.
        covariates: limma only — obs columns to adjust for.
        _force_resample: If True, forces resampling even if the number of resamples exceeds the number of combinations.

    Returns:
        DeaResult containing DE analysis results.
    """
    _validate_run_de_args(stat_method, expr, interaction, interaction_levels, covariates, fdr)

    # 1. Data validation: read the modality once, then let the engine validate the features. The
    #    engines' feasibility checks are not interchangeable (the permutation path validates sample
    #    size + group coverage; limma validates its design levels and per-feature estimability), so
    #    each engine owns its own validate() while sharing the single min_pct rule (DeaValidator).
    inputs = DeaInputs.from_mudata(
        mdata=mdata, modality=modality, category=category, ctrl=ctrl, expr=expr, layer=layer
    )
    engine = _select_engine(
        stat_method,
        measure=measure,
        n_resamples=n_resamples,
        fdr=fdr,
        log_transformed=log_transformed,
        force_resample=_force_resample,
        min_pct=min_pct,
        category=category,
        ctrl=ctrl,
        expr=expr,
        interaction=interaction,
        interaction_levels=interaction_levels,
        covariates=covariates,
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
        n_resamples=n_resamples,
        force_resample=_force_resample,
        draw_guidance_line=engine.provides_comparable_fold_change,
    )

    return result


def _validate_run_de_args(
    stat_method: str,
    expr: str | None,
    interaction: str | None,
    interaction_levels: list | None,
    covariates: list[str] | None,
    fdr: bool | str,
) -> None:
    if stat_method not in ["welch", "student", "wilcoxon", "limma"]:
        raise ValueError(
            f"Invalid statistic: {stat_method}. Choose from 'welch', 'student', 'wilcoxon', 'limma'."
        )
    if stat_method == "limma":
        if expr is None:
            raise ValueError("stat_method='limma' requires an explicit 'expr' group (expr=None is not supported).")
        return
    if interaction is not None or interaction_levels is not None or covariates is not None:
        raise ValueError(
            "'interaction', 'interaction_levels' and 'covariates' are only supported with stat_method='limma'."
        )
    if fdr not in ["empirical", "bh", False]:
        raise ValueError("invalied fdr (mutiple test correction). Choose from 'empirical', 'bh', or False (bool)")


def _select_engine(
    stat_method: str,
    *,
    measure: str,
    n_resamples: int | None,
    fdr: bool | str,
    log_transformed: bool,
    force_resample: bool,
    min_pct: float,
    category: str,
    ctrl: str,
    expr: str | None,
    interaction: str | None,
    interaction_levels: list | None,
    covariates: list[str] | None,
) -> DeEngine:
    """Pick the DE engine for ``stat_method``, passing each only the knobs it uses."""
    if stat_method == "limma":
        return LimmaEngine(
            category=category,
            ctrl=ctrl,
            expr=expr,
            interaction=interaction,
            interaction_levels=interaction_levels,
            covariates=covariates,
            min_pct=min_pct,
        )
    return PermutationEngine(
        stat_method=stat_method,
        measure=measure,
        n_resamples=n_resamples,
        fdr=fdr,
        log_transformed=log_transformed,
        min_pct=min_pct,
        force_resample=force_resample,
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
    contrast (an intrinsic, mean-based test output), while the permutation / simple paths get their
    measure-based difference at this step.
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
    n_resamples: int | None,
    force_resample: bool,
    draw_guidance_line: bool,
) -> None:
    """Set the fold-change guidance line (``fc_pct_1``/``fc_pct_5``) of ``result`` in place.

    Skipped when ``draw_guidance_line`` is False — the reported log2fc is not a raw two-group
    difference the label-permutation null can match (limma interaction / covariate-adjusted
    contrasts) — and when the engine already produced the line (the permutation shuffle byproduct),
    so the null is never built twice. limma main effects and the parametric simple test compute it
    here from the same label-permutation log2FC null, reusing the engine's usable-feature mask and
    its ``effect_measure``.
    """
    if not draw_guidance_line or result.fc_pct_5 is not None:
        return
    result.fc_pct_1, result.fc_pct_5 = compute_fc_guidance_line(
        inputs.ctrl_arr,
        inputs.expr_arr,
        validation.feature_mask,
        measure=effect_measure,
        log_transformed=log_transformed,
        n_resamples=n_resamples,
        force_resample=force_resample,
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
