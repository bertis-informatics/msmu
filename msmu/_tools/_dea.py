import mudata as md
import numpy as np
import pandas as pd
from typing import Literal
import logging

from .._statistics._permutation import PermutationTest
from .._statistics._de_base import PermTestResult, StatTestResult, DeaValidator, DeaResult
from .._statistics._statistics import (
    simple_test,
    _measure_central_tendency,
    _calc_log2fc,
    _get_pct_expression,
)


logger = logging.getLogger(__name__)


def _get_test_array(
    mdata: md.MuData,
    modality: str,
    category: str,
    control: str,
    expr: str | None,
    layer: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    mod_adata = mdata[modality]
    if layer is not None:
        data = pd.DataFrame(mod_adata.layers[layer], index=mod_adata.obs_names, columns=mod_adata.var_names)
    else:
        data = mod_adata.to_df()

    ctrl_samples = mod_adata.obs.loc[mod_adata.obs[category] == control,].index.to_list()

    if expr is not None:
        expr_samples = mod_adata.obs.loc[mod_adata.obs[category] == expr,].index.to_list()
    else:
        expr_samples = mod_adata.obs.loc[mod_adata.obs[category] != control,].index.to_list()

    ctrl_arr = data.T[ctrl_samples].values.T
    expr_arr = data.T[expr_samples].values.T

    return ctrl_arr, expr_arr


def run_de(
    mdata: md.MuData,
    modality: str,
    category: str,
    ctrl: str,
    expr: str | None = None,
    min_pct: float = 0.5,
    layer: str | None = None,
    stat_method: Literal["welch", "student", "wilcoxon"] = "welch",  # TODO: add "limma"
    measure: Literal["median", "mean"] = "median",
    n_resamples: int | None = 1000,
    fdr: bool | Literal["empirical", "bh"] = "empirical",
    log_transformed: bool = True,
    _force_resample: bool = False,
) -> DeaResult:
    """
    Run Differential Expression Analysis (DEA) between two groups in a MuData object.

    Parameters:
        mdata: MuData object containing the data.
        modality: Modality name within the MuData to analyze.
        category: Observation category to define groups.
        ctrl: Name of the control group.
        expr: Name of the experimental group. If None, all other groups are used.
        layer: Layer to use for quantification aggregation. If None, the default layer (.X) will be used. Defaults to None.
        stat_method: Statistical test to use ("welch", "student", "wilcoxon").
        measure: Measure of central tendency to use ("median" or "mean") for fold-change.
        n_resamples: Number of resamples for permutation test. If None, no permutation test is performed.
        fdr: Method for multiple test correction ("empirical", "bh", or False).
        log_transformed: If True, data is assumed to be log-transformed. Defaults to True.
        _force_resample: If True, forces resampling even if the number of resamples exceeds the number of combinations.

    Returns:
        DeaResult containing DE analysis results.
    """
    if stat_method not in ["welch", "student", "wilcoxon"]:
        raise ValueError(f"Invalid statistic: {stat_method}. Choose from 'welch', 'student', 'wilcoxon'.")
    if fdr not in ["empirical", "bh", False]:
        raise ValueError(f"invalied fdr (mutiple test correction). Choose from 'empirical', 'bh', or False (bool)")

    ctrl_arr, expr_arr = _get_test_array(
        mdata=mdata,
        modality=modality,
        category=category,
        control=ctrl,
        expr=expr,
        layer=layer,
    )
    dea_validator = DeaValidator(ctrl_arr, expr_arr, min_pct=min_pct)

    if dea_validator.min_sample_size_availability is False:
        logger.warning("Not enough samples to perform DEA. Returning result with only fold changes.")
        de_res: StatTestResult = _make_dummy_de_result()

    else:
        valid_ctrl_arr = ctrl_arr.copy()
        valid_ctrl_arr[:, ~dea_validator.sufficient_feature_indices] = np.nan
        valid_expr_arr = expr_arr.copy()
        valid_expr_arr[:, ~dea_validator.sufficient_feature_indices] = np.nan

        if n_resamples is not None:
            perm_test: PermutationTest = PermutationTest(
                ctrl_arr=valid_ctrl_arr,
                expr_arr=valid_expr_arr,
                n_resamples=n_resamples,
                _force_resample=_force_resample,
                fdr=fdr,
            )

            test_res: PermTestResult = perm_test.run(
                n_permutations=n_resamples,
                stat_method=stat_method,
                measure=measure,
                log_transformed=log_transformed,
            )

        else:
            test_res: StatTestResult = simple_test(
                ctrl=valid_ctrl_arr,
                expr=valid_expr_arr,
                stat_method=stat_method,
                fdr=fdr,
            )

    de_res = DeaResult(test_res)

    repr_ctrl = _measure_central_tendency(ctrl_arr, measure)
    repr_expr = _measure_central_tendency(expr_arr, measure)

    de_res.ctrl = ctrl
    de_res.expr = expr if expr is not None else "all_other_groups"
    de_res.features = mdata[modality].var.index.to_numpy()
    de_res.repr_ctrl = repr_ctrl
    de_res.repr_expr = repr_expr
    de_res.pct_ctrl = _get_pct_expression(ctrl_arr)
    de_res.pct_expr = _get_pct_expression(expr_arr)
    de_res.log2fc = _calc_log2fc(repr_ctrl, repr_expr, log_transformed=log_transformed)

    return de_res


def _make_dummy_de_result() -> StatTestResult:
    dummy_de_res = StatTestResult(
        stat_method="",
        p_value=np.array([]),
        q_value=np.array([]),
    )
    return dummy_de_res
