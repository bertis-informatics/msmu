import mudata as md
import numpy as np

from .PermutationTest import PermutationTest, PermutationTestResult


def _get_test_array(
    mdata: md.MuData,
    modality: str,
    catetory: str,
    control: str,
    expr: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mod_mdata = mdata[modality].copy()
    ctrl_samples = mod_mdata.obs.loc[
        mod_mdata.obs[catetory] == control,
    ].index.to_list()

    if expr is not None:
        expr_samples = mod_mdata.obs.loc[
            mod_mdata.obs[catetory] == expr,
        ].index.to_list()
    else:
        expr_samples = mod_mdata.obs.loc[
            mod_mdata.obs[catetory] != control,
        ].index.to_list()

    ctrl_arr = mod_mdata.to_df().T[ctrl_samples].values.T
    expr_arr = mod_mdata.to_df().T[expr_samples].values.T

    return ctrl_arr, expr_arr


def permutation_test(
    mdata: md.MuData,
    category: str,
    control: str,
    expr: str | None = None,
    modality: str = "protein",
    n_resamples: int = 1000,
    n_jobs: int = 1,
    statistic: str = "welch",
    force_resample: bool = False,
    fdr: bool | str = "empirical"
) -> PermutationTestResult:
    """
    Perform a permutation test on the given MuData object.
    Parameters
    ----------
    mdata : md.MuData
        The MuData object containing the data.
    modality : str
        The modality to perform the test on. Default is 'protein'
    category : str
        The category column in the mdata.obs.
    control : str
        The control group label.
    expr : str | None
        The experimental group label. If None, all other groups are considered experimental. Default is None.
    n_resamples : int
        The number of resamples for the permutation test. Default is 1000.
    n_jobs : int
        The number of parallel jobs to run.
    statistic : str
        The statistical test to use. Options are 'welch', 'student', 'wilcoxon', or 'median_diff'. Default is 'welch'.
    fdr: str | bool
        The FDR control method to use. Options are 'empirical', 'bh', 'storey'. Default is 'empirical'.
    force_resample : bool
        If True, forces resampling even if the number of permutations exceeds the possible combinations. Default is False.
    Returns
    -------
    PermutationTestResult
        The result of the permutation test.
    Raises
    ------
    ValueError
        If the statistic is not one of the supported types.
    """
    ctrl_arr, expr_arr = _get_test_array(
        mdata=mdata,
        modality=modality,
        catetory=category,
        control=control,
        expr=expr,
    )
    if statistic not in ["welch", "student", "wilcoxon", "med_diff"]:
        raise ValueError(
            f"Invalid statistic: {statistic}. Choose from 'welch', 'student', 'wilcoxon', or 'med_diff'."
        )
    if fdr not in ["empirical", "bh", "storey", False]:
        raise ValueError(
            f"invalied fdr (mutiple test correction). Choose from 'empirical', 'storey' or 'bh'. Or turn off with False (bool)"
        )

    perm_test: PermutationTest = PermutationTest(
        ctrl_arr=ctrl_arr,
        expr_arr=expr_arr,
        n_resamples=n_resamples,
        force_resample=force_resample,
        fdr=fdr,
    )
    
    perm_res: PermutationTestResult = perm_test.run(
        n_permutations=n_resamples, n_jobs=n_jobs, statistic=statistic
    )
    perm_res.ctrl = control
    perm_res.expr = expr if expr is not None else "all_other_groups"
    perm_res.features = mdata[modality].var.index.to_numpy()

    return perm_res


# def limma(self):
#     return Limma()
