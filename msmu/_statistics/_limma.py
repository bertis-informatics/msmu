"""limma differential expression for msmu.

A thin, domain-facing layer over inmoose's ``lmFit`` / ``contrasts_fit`` /
``squeezeVar`` — a numerically exact Python port of Smyth's limma (validated
bit-for-bit against R limma 3.62). The caller expresses a comparison in domain
terms via one or two ``obs`` factors; this module builds the means-model design
matrix and the numeric contrast vector internally, so raw contrast coefficients
never surface in the public API.

Two comparison levels are supported:

* **Level 1** (``interaction=None``) — main effect ``expr - ctrl`` of a single factor.
* **Level 2** (``interaction`` given) — interaction / difference-in-differences,
  ``(expr - ctrl) @ interaction_level_a  -  (expr - ctrl) @ interaction_level_b``.

Sign convention: a positive log2 fold change means higher in ``expr``.

The eBayes B-statistic (``lods``) is intentionally not computed: inmoose 0.9.1's
lods path has a DataFrame-indexing bug when the prior degrees of freedom diverge.
Moderated t/p are computed directly from ``squeezeVar`` (the same posterior
variance eBayes uses), which reproduces R limma's moderated t exactly.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t as t_distribution

from ._multiple_test_correction import PvalueCorrection

POSITIVE_DIRECTION_NOTE = "positive log2 fold change means higher in 'expr'"


@dataclass
class LimmaContrast:
    """A means-model design over sample cells plus the numeric contrast vector.

    Attributes:
        design: samples x design-columns DataFrame (one-hot cell means model,
            optionally followed by covariate columns).
        weights: contrast weight for each design column, in ``design`` column order.
        label: human-readable description of the comparison.
        kept_samples: the sample index entering the fit (subset of obs).
        cell_columns: the design columns that are cell indicators (excludes covariates).
    """

    design: pd.DataFrame
    weights: np.ndarray
    label: str
    kept_samples: pd.Index
    cell_columns: list[str]


@dataclass
class LimmaResult:
    """Per-feature limma moderated-t results, aligned to ``features``."""

    features: np.ndarray
    log2fc: np.ndarray
    moderated_t: np.ndarray
    p_value: np.ndarray
    q_value: np.ndarray
    contrast_label: str


def _require_levels_present(values: pd.Series, column_name: str, requested_levels: list) -> None:
    available = list(pd.unique(values.dropna()))
    missing = [level for level in requested_levels if level not in available]
    if missing:
        raise ValueError(
            f"Level(s) {missing} not found in obs column '{column_name}'. "
            f"Available levels: {available}."
        )


def _resolve_interaction_levels(interaction_values: pd.Series, interaction_column: str, interaction_levels) -> list[str]:
    if interaction_levels is None:
        resolved = sorted((str(level) for level in pd.unique(interaction_values.dropna())))
    else:
        resolved = [str(level) for level in interaction_levels]
    if len(resolved) != 2:
        raise ValueError(
            f"Interaction factor '{interaction_column}' requires exactly 2 levels, got {len(resolved)}: "
            f"{resolved}. Pass interaction_levels=[level_a, level_b] to select two."
        )
    _require_levels_present(interaction_values.astype(str), interaction_column, resolved)
    return resolved


def _build_covariate_design(obs: pd.DataFrame, kept_samples: pd.Index, covariates: list[str] | None):
    """Return (covariate_design, n_covariate_columns). Numeric passed through, categorical one-hot drop-first."""
    if not covariates:
        return None, 0
    frames = []
    for covariate in covariates:
        if covariate not in obs.columns:
            raise KeyError(f"Covariate column '{covariate}' not found in obs. Available: {list(obs.columns)}")
        column = obs.loc[kept_samples, covariate]
        if pd.api.types.is_numeric_dtype(column):
            frames.append(column.astype(float).to_frame(name=f"cov_{covariate}"))
        else:
            dummies = pd.get_dummies(column.astype(str), prefix=f"cov_{covariate}", drop_first=True)
            frames.append(dummies.astype(float))
    covariate_design = pd.concat(frames, axis=1)
    return covariate_design, covariate_design.shape[1]


def build_contrast(
    obs: pd.DataFrame,
    category: str,
    ctrl: str,
    expr: str,
    interaction: str | None = None,
    interaction_levels=None,
    covariates: list[str] | None = None,
) -> LimmaContrast:
    """Build the means-model design and the contrast vector for a Level 1 or Level 2 comparison."""
    if category not in obs.columns:
        raise KeyError(f"Column '{category}' not found in obs. Available: {list(obs.columns)}")
    _require_levels_present(obs[category], category, [ctrl, expr])

    if interaction is None:
        keep_mask = obs[category].isin([ctrl, expr])
        kept_samples = obs.index[keep_mask]
        category_values = obs.loc[kept_samples, category].astype(str)
        cell_columns = [str(ctrl), str(expr)]
        cell_design = pd.DataFrame(
            {level: (category_values == level).astype(float).to_numpy() for level in cell_columns},
            index=kept_samples,
        )
        weight_map = {str(ctrl): -1.0, str(expr): 1.0}
        label = f"{expr} vs {ctrl}"
    else:
        if interaction not in obs.columns:
            raise KeyError(f"Column '{interaction}' not found in obs. Available: {list(obs.columns)}")
        level_a, level_b = _resolve_interaction_levels(obs[interaction], interaction, interaction_levels)
        keep_mask = obs[category].isin([ctrl, expr]) & obs[interaction].astype(str).isin([level_a, level_b])
        kept_samples = obs.index[keep_mask]
        category_values = obs.loc[kept_samples, category].astype(str)
        interaction_values = obs.loc[kept_samples, interaction].astype(str)

        # Difference-in-differences: (expr@a - ctrl@a) - (expr@b - ctrl@b).
        # weight = category_sign * interaction_sign, with ctrl=-1/expr=+1 and level_a=+1/level_b=-1.
        cell_columns = []
        weight_map = {}
        cell_masks = {}
        for category_level, category_sign in ((str(ctrl), -1.0), (str(expr), 1.0)):
            for interaction_level, interaction_sign in ((level_a, 1.0), (level_b, -1.0)):
                name = f"{category_level}|{interaction_level}"
                cell_columns.append(name)
                weight_map[name] = category_sign * interaction_sign
                cell_masks[name] = (
                    (category_values == category_level) & (interaction_values == interaction_level)
                ).to_numpy()
        cell_design = pd.DataFrame(
            {name: cell_masks[name].astype(float) for name in cell_columns},
            index=kept_samples,
        )
        label = f"({expr} vs {ctrl}) interaction across {interaction}: {level_a} vs {level_b}"

    empty_cells = [name for name in cell_columns if cell_design[name].sum() == 0]
    if empty_cells:
        raise ValueError(
            f"No samples for cell(s) {empty_cells} in the requested comparison; "
            f"cannot build the design. Check the level names in '{category}'"
            + (f" and '{interaction}'." if interaction is not None else ".")
        )

    covariate_design, n_covariate_columns = _build_covariate_design(obs, kept_samples, covariates)
    if covariate_design is not None:
        design = pd.concat([cell_design, covariate_design], axis=1)
    else:
        design = cell_design

    weights = np.array(
        [weight_map[name] for name in cell_columns] + [0.0] * n_covariate_columns,
        dtype=float,
    )
    return LimmaContrast(
        design=design,
        weights=weights,
        label=label,
        kept_samples=kept_samples,
        cell_columns=cell_columns,
    )


def cell_blocks_for_contrast(fit_matrix: pd.DataFrame, contrast: LimmaContrast) -> list[np.ndarray]:
    """One ``(n_features x n_cell_samples)`` block per design cell, for feature validation.

    The blocks feed the shared ``DeaValidator`` / ``sufficient_feature_mask``: limma keeps a
    feature only when every design cell has at least ``min_pct`` (and at least one) non-missing
    observation and the total residual degrees of freedom are positive (``require_residual_df``),
    so the per-gene observed design stays full rank and inmoose's NA-aware fit never hits a
    singular sub-design.

    Args:
        fit_matrix: features x samples DataFrame aligned to ``contrast.kept_samples``.
        contrast: the means-model contrast whose ``cell_columns`` define the design cells.

    Returns:
        One numpy block per cell, in ``contrast.cell_columns`` order.
    """
    cell_design = contrast.design[contrast.cell_columns]
    return [
        fit_matrix.loc[:, cell_design.index[cell_design[cell] > 0]].to_numpy()
        for cell in contrast.cell_columns
    ]


def _fit_contrast(expr_matrix: pd.DataFrame, contrast: LimmaContrast) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the contrast with inmoose and return (log2fc, moderated_t, p_value) per feature.

    Moderated statistics are computed from ``squeezeVar`` directly to avoid inmoose
    0.9.1's eBayes lods bug; this reproduces R limma's moderated t exactly.
    """
    from inmoose.limma import lmFit, contrasts_fit, squeezeVar

    design = contrast.design
    ordered_matrix = expr_matrix.loc[:, design.index]  # features x samples, aligned to design rows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = lmFit(ordered_matrix, design)
        # inmoose renames design columns to positional labels (column0..N); align the
        # contrast to those labels positionally.
        contrast_frame = pd.DataFrame(
            {"contrast": np.asarray(contrast.weights, dtype=float)},
            index=list(fit.coefficients.columns),
        )
        contrast_fit = contrasts_fit(fit, contrast_frame)

        sigma = np.asarray(contrast_fit.sigma, dtype=float)
        df_residual = np.asarray(contrast_fit.df_residual, dtype=float)
        squeezed = squeezeVar(sigma**2, df_residual)
        posterior_variance = np.asarray(squeezed["var_post"], dtype=float)
        df_prior = squeezed["df_prior"]

        df_total = df_residual + df_prior
        df_total = np.minimum(df_total, np.nansum(df_residual))

        log2fc = np.asarray(contrast_fit.coefficients.iloc[:, 0], dtype=float)
        stdev_unscaled = np.asarray(contrast_fit.stdev_unscaled.iloc[:, 0], dtype=float)
        standard_error = stdev_unscaled * np.sqrt(posterior_variance)
        moderated_t = log2fc / standard_error
        p_value = 2.0 * t_distribution.sf(np.abs(moderated_t), df_total)

    return log2fc, moderated_t, p_value


def fit_limma(fit_matrix: pd.DataFrame, contrast: LimmaContrast, feature_mask: np.ndarray) -> LimmaResult:
    """Fit a validated limma contrast and assemble per-feature moderated-t results.

    The test step of ``run_de(stat_method="limma")``: the contrast and the estimable-feature mask
    are built earlier (validation), so this only fits the estimable features and fills the rest
    with NaN. Positive log2 fold change means higher in ``expr``. The model log2 fold change is the
    contrast coefficient (an intrinsic output of the fit), while representative values, detection %
    and the fold-change guidance line are filled in by ``run_de``'s common post-processing.

    Parameters:
        fit_matrix: features x samples log2-intensity DataFrame aligned to ``contrast.kept_samples``.
        contrast: the means-model contrast to fit.
        feature_mask: boolean mask over ``fit_matrix`` rows marking the estimable features
            (from ``DeaValidator``); assumed to have at least one True (guaranteed by validation).

    Returns:
        LimmaResult with per-feature log2fc, moderated t, p-value and BH q-value, aligned to every
        feature (NaN where not estimable).
    """
    features = np.asarray(fit_matrix.index)
    log2fc = np.full(features.size, np.nan)
    moderated_t = np.full(features.size, np.nan)
    p_value = np.full(features.size, np.nan)

    estimable_matrix = fit_matrix.loc[feature_mask]
    fitted_log2fc, fitted_t, fitted_p = _fit_contrast(estimable_matrix, contrast)
    log2fc[feature_mask] = fitted_log2fc
    moderated_t[feature_mask] = fitted_t
    p_value[feature_mask] = fitted_p

    q_value = PvalueCorrection.bh(p_value)

    return LimmaResult(
        features=features,
        log2fc=log2fc,
        moderated_t=moderated_t,
        p_value=p_value,
        q_value=q_value,
        contrast_label=contrast.label,
    )
