import mudata as md
import numpy as np
from typing import Literal

from .._statistics._permutation import PermutationTest
from .._statistics._de_base import PermTestResult, StatTestResult
from .._statistics._statistics import simple_test


def _get_test_array(
    mdata: md.MuData,
    modality: str,
    category: str,
    control: str,
    expr: str | None = None,
    layer: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mod_adata = mdata[modality].copy()
    if layer is not None:
        mod_adata.X = mod_adata.layers[layer]
    ctrl_samples = mod_adata.obs.loc[mod_adata.obs[category] == control,].index.to_list()

    if expr is not None:
        expr_samples = mod_adata.obs.loc[mod_adata.obs[category] == expr,].index.to_list()
    else:
        expr_samples = mod_adata.obs.loc[mod_adata.obs[category] != control,].index.to_list()

    ctrl_arr = mod_adata.to_df().T[ctrl_samples].values.T
    expr_arr = mod_adata.to_df().T[expr_samples].values.T

    return ctrl_arr, expr_arr


def run_de(
    mdata: md.MuData,
    modality: str,
    category: str,
    ctrl: str,
    expr: str | None = None,
    layer: str | None = None,
    stat_method: Literal["welch", "student", "wilcoxon", "med_diff", "limma"] = "welch",
    n_resamples: int | None = 1000,
    fdr: bool | Literal["empirical", "bh", "storey"] = "empirical",
    n_jobs: int = 1,
    _force_resample: bool = False,
) -> PermTestResult | StatTestResult:
    """
    Run Differential Expression Analysis (DEA) between two groups in a MuData object.

    Parameters:
        mdata: MuData object containing the data.
        modality: Modality name within the MuData to analyze.
        category: Observation category to define groups.
        ctrl: Name of the control group.
        expr: Name of the experimental group. If None, all other groups are used.
        layer: Layer to use for quantification aggregation. If None, the default layer (.X) will be used. Defaults to None.
        stat_method: Statistical test to use ("welch", "student", "wilcoxon", "med_diff", "limma").
        n_resamples: Number of resamples for permutation test. If None, no permutation test is performed.
        fdr: Method for multiple test correction ("empirical", "bh", "storey", or False).
        n_jobs: Number of parallel jobs to use.
        _force_resample: If True, forces resampling even if the number of resamples exceeds the number of combinations.

    Returns:
        PermTestResult or StatTestResult containing DEA results.
    """
    if stat_method not in ["welch", "student", "wilcoxon", "med_diff", "limma"]:
        raise ValueError(
            f"Invalid statistic: {stat_method}. Choose from 'welch', 'student', 'wilcoxon', 'med_diff', or 'limma'."
        )
    if fdr not in ["empirical", "bh", "storey", False]:
        raise ValueError(
            f"invalied fdr (mutiple test correction). Choose from 'empirical', 'storey' or 'bh'. Or turn off with False (bool)"
        )
    ctrl_arr, expr_arr = _get_test_array(
        mdata=mdata,
        modality=modality,
        category=category,
        control=ctrl,
        expr=expr,
        layer=layer,
    )

    if n_resamples is not None:
        perm_test: PermutationTest = PermutationTest(
            ctrl_arr=ctrl_arr,
            expr_arr=expr_arr,
            n_resamples=n_resamples,
            _force_resample=_force_resample,
            fdr=fdr,
        )

        de_res: PermTestResult = perm_test.run(n_permutations=n_resamples, n_jobs=n_jobs, stat_method=stat_method)

    elif stat_method == "limma":
        pass

    else:
        de_res: StatTestResult = simple_test(
            ctrl=ctrl_arr,
            expr=expr_arr,
            stat_method=stat_method,
            fdr=fdr,
        )

    de_res.ctrl = ctrl
    de_res.expr = expr if expr is not None else "all_other_groups"
    de_res.features = mdata[modality].var.index.to_numpy()

    return de_res
