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
    mdata,
    modality,
    control,
    expr,
    category,
    n_resamples: int = 1000,
    n_jobs: int = 1,
    statistic: str | list = "all",
    force_resample: bool = False,
) -> PermutationTestResult:
    ctrl, expr = _get_test_array(
        mdata=mdata,
        modality=modality,
        catetory=category,
        control=control,
        expr=expr,
    )

    perm_test: PermutationTest = PermutationTest(ctrl=ctrl, expr=expr)
    possible_combinations: list = perm_test.possible_combinations

    if n_resamples == -np.inf:
        permutation_method = "exact"
    elif n_resamples == len(possible_combinations):
        permutation_method = "exact"
    elif (n_resamples > len(possible_combinations)) and not force_resample:
        permutation_method = "exact"
    elif (n_resamples > len(possible_combinations)) and force_resample:
        permutation_method = "randomised"
    else:
        permutation_method = "randomised"

    print(f"Permutation Method: {permutation_method}")

    if statistic == "all":
        statistic = ["t_test", "wilcoxon", "med_diff"]
    else:
        statistic = [statistic]

    print(f"Statistics: {statistic}")

    perm_test.permutation_method = permutation_method
    perm_res: PermutationTestResult = perm_test.run(
        n_permutations=n_resamples, n_jobs=n_jobs, statistic=statistic
    )
    perm_res.features = mdata[modality].var.index.to_numpy()

    return perm_res


# def limma(self):
#     return Limma()
