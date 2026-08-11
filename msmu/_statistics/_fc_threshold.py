"""Engine-independent fold-change guidance threshold.

Computes the volcano plot's fold-change guidance line: the symmetric 1st/5th percentile of the
log2 fold-change under label permutation. This is the engine-independent path — the permutation
test produces the same thresholds as a byproduct of its own shuffles (see ``PermutationTest``),
while limma and other non-permutation results call ``compute_fc_thresholds`` so they carry the
same line. The label-permutation building blocks it reuses live in ``_permutation_core``, shared
with ``PermutationTest``, so the line is computed identically across engines.
"""

import numpy as np

from ._permutation_core import resolve_method, make_iterations, permuted_log2fc, fc_threshold_from_null


def compute_fc_thresholds(
    ctrl_arr: np.ndarray,
    expr_arr: np.ndarray,
    measure: str,
    log_transformed: bool,
    n_resamples: int,
    force_resample: bool = False,
) -> tuple[float, float]:
    """Fold-change guidance thresholds (``fc_pct_1``, ``fc_pct_5``) from the label-permutation null.

    Builds ONLY the log2 fold-change null by permuting the group labels (no test statistic, no
    FDR), then takes the symmetric 1st/5th percentile threshold. The permutation engine already
    produces this null as a byproduct of its own shuffles and therefore does NOT call this;
    ``limma`` and other non-permutation results call it so they carry the same guidance line.

    Args:
        ctrl_arr: Control group data (n_samples_ctrl x n_features).
        expr_arr: Experimental group data (n_samples_expr x n_features).
        measure: Central-tendency measure for the fold change ("median" or "mean").
        log_transformed: Whether the data is log-transformed.
        n_resamples: Number of label permutations (exact enumeration is used for small designs).
        force_resample: Force randomised resampling even when n_resamples exceeds the split count.

    Returns:
        (fc_pct_1, fc_pct_5) thresholds.
    """
    n_ctrl, n_expr = len(ctrl_arr), len(expr_arr)
    method = resolve_method(n_ctrl, n_expr, n_resamples, force_resample)
    iterations = make_iterations(n_ctrl, n_expr, method, n_resamples)

    concated_arr = np.concatenate((ctrl_arr, expr_arr), axis=0)
    log2fc_null = np.array(
        [
            permuted_log2fc(concated_arr, combination, method, n_ctrl, measure, log_transformed)
            for combination in iterations
        ]
    )

    return fc_threshold_from_null(log2fc_null, 1), fc_threshold_from_null(log2fc_null, 5)


def compute_fc_guidance_line(
    ctrl_arr: np.ndarray,
    expr_arr: np.ndarray,
    feature_mask: np.ndarray,
    measure: str,
    log_transformed: bool,
    n_resamples: int | None,
    force_resample: bool,
    default_resamples: int = 1000,
) -> tuple[float, float]:
    """Fold-change guidance thresholds (fc_pct_1, fc_pct_5) for a non-permutation DE result.

    Masks features outside ``feature_mask`` (the engine's usable-feature set from validation), then
    builds the label-permutation log2FC null via ``compute_fc_thresholds``. The guidance line is
    always produced: it uses the caller's ``n_resamples`` when that is a positive permutation count,
    otherwise ``default_resamples``. Only limma reaches this (its test does not resample), and it
    passes a fixed count so its guidance line does not depend on the permutation-only ``n_resamples``.

    Args:
        ctrl_arr: Control group data (n_samples_ctrl x n_features).
        expr_arr: Experimental group data (n_samples_expr x n_features).
        feature_mask: boolean mask over features to keep (validation's usable-feature set).
        measure: Central-tendency measure for the fold change ("median" or "mean").
        log_transformed: Whether the data is log-transformed.
        n_resamples: The test's permutation count, or None for a non-resampling test.
        force_resample: Force randomised resampling even when n_resamples exceeds the split count.
        default_resamples: Permutation count to use when ``n_resamples`` is not a positive int.

    Returns:
        (fc_pct_1, fc_pct_5) thresholds.
    """
    valid_ctrl_arr = ctrl_arr.copy()
    valid_ctrl_arr[:, ~feature_mask] = np.nan
    valid_expr_arr = expr_arr.copy()
    valid_expr_arr[:, ~feature_mask] = np.nan

    resamples = n_resamples if isinstance(n_resamples, int) and n_resamples > 0 else default_resamples
    return compute_fc_thresholds(
        valid_ctrl_arr,
        valid_expr_arr,
        measure=measure,
        log_transformed=log_transformed,
        n_resamples=resamples,
        force_resample=force_resample,
    )
