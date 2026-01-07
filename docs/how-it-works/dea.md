# Differential Expression Analysis (DEA)

## Overview

Differential Expression Analysis (DEA) identifies proteins or peptides with significant abundance changes between experimental conditions. `msmu` provides permutation-based statistical testing to assess differential expression while controlling the false discovery rate (FDR).

## `mm.tl.run_de()`

The `run_de()` function performs a non-parametric permutation test to evaluate differential expression between two groups. It calculates p-values based on the distribution of test statistics obtained from permuted group labels.

This function uses `welch`'s t-statistic by default, which is suitable for unequal variances between groups. Other statistics such as `student` t-test, `wilcoxon` rank-sum test, and `med_diff` (median difference) are also available.

For `FDR` correction, `msmu` supports `empirical` from null distribution, and `bh` (Benjamini-Hochberg) methods. `empirical` FDR is recommended when using permutation tests.

`n_resamples` specifies the number of random permutations to generate the null distribution. If set to `None`, a simple hypothesis test without permutations is performed. `1000` resamples as default provides a good balance between accuracy and computational cost.

If sample sizes are too small to meet n_resamples, all possible permutations are used to compute exact p-values (exact test).

log2 fold-change (`log2FC`) between the two groups is calculated as the difference of log2-transformed median values.

`p-value` from the test is computed with the proportion of permuted statistics that are as extreme or more extreme than the observed statistic in null distribution with two-sided test.

`q-value` with `empirical` FDR is calculated by `E[FDR] = pi0 * E[FP] / E[TP]` referred to [Yang Xie et al., Bioinformatics, 2011.](https://academic.oup.com/bioinformatics/article/21/23/4280/194680) and [Storey et al., 2003](https://www.pnas.org/doi/epdf/10.1073/pnas.1530509100).

See more details in the [`msmu.tl.run_de`](/reference/tl/run_de/) and usage examples in the tutorial [`DE Analysis`](/tutorials/dea/).

```python
de_res = mm.tl.run_de(
    mdata,
    modality="protein",      # or "peptide"
    category="condition",    # column in .obs defining groups
    ctrl="control",          # control group label
    expr="treated",          # experimental group label
    stat_method="welch",     # options: "welch", "student", "wilcoxon", "med_diff", default "welch"
    fdr="empirical",         # options: "empirical", "bh", "storey", or False, default "empirical"
    n_resamples=1000,        # number of permutations, default 1000, if None, simple hypothesis test is performed
)

de_res.to_df() # get results as pandas DataFrame
```

DE analysis results are stored in `PermTestResult`(for permutation test) (or `StatTestResult`; for simple test) object, which contains:

DE results can be accessed as a pandas `DataFrame` using the `to_df()` method.

## Visualization of DEA Results

`msmu` provides visualization function to explore DEA results with volcano plots.

```python
de_res.plot_volcano(
    log2fc_cutoff=None, # (optional) log2 fold-change cutoff line, default None which shows fc_pct_5 line
    pval_cutoff=0.05,   # (optional) p-value cutoff line, default 0.05
    label_top_n=5,      # (optional) number of top significant features to label, default None (no labels)
)
```
