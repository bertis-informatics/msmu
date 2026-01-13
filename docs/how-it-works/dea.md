# Differential Expression (DE) Analysis

## Overview

Differential Expression (DE) Analysis identifies proteins or peptides with significant abundance changes between experimental conditions. `msmu` provides permutation-based statistical testing to assess differential expression while controlling the false discovery rate (FDR).

## `mm.tl.run_de()`

The `run_de()` function performs a non-parametric permutation test to evaluate differential expression between two groups. It calculates p-values based on the distribution of test statistics obtained from permuted group labels.

This function uses Welch's t-statistic by default, which is suitable for unequal variances between groups. Other statistics such as Student's t-statistics, Wilcoxon's W-statistics (rank-sum) test, and median difference are also available.

For multiple testing correction, `msmu` supports empirical FDR, and Benjamini-Hochberg method. Empirical FDR is recommended when using permutation tests.

`n_resamples` specifies the number of random permutations to generate the null distribution. If set to `None`, a simple hypothesis test without permutations is performed. The default of `1000` permutations provides a practical balance between statistical accuracy and computational cost.

If sample sizes are too small to meet `n_resamples`, all possible permutations are used to compute exact p-values (exact test).

Log2 fold-change (`log2FC`) between the two groups is calculated as the difference of log2-transformed median values.

`p-value` from the test is computed with the proportion of permuted statistics that are as extreme or more extreme than the observed statistic in null distribution with two-sided test.

`q-value` with `empirical` FDR is calculated by `E[FDR] = pi0 * E[FP] / E[TP]` referred to [Yang Xie et al., Bioinformatics, 2011.](https://academic.oup.com/bioinformatics/article/21/23/4280/194680) and [Storey et al., 2003](https://www.pnas.org/doi/epdf/10.1073/pnas.1530509100).

See more details in the [`msmu.tl.run_de`](../../reference/tl/run_de/) and usage examples in the tutorial [`DE Analysis`](../../tutorials/dea/).

```python
de_res = mm.tl.run_de(
    mdata,
    modality="protein",      # or "peptide"
    category="condition",    # column in .obs defining groups
    ctrl="control",          # control group label
    expr="treated",          # experimental group label
    stat_method="welch",     # options: "welch", "student", "wilcoxon", default "welch"
    measure="median",        # options: "mean", "median", default "median"
    min_pct=0.5,             # minimum fraction of non-missing values in at least one group, default 0.5
    fdr="empirical",         # options: "empirical", "bh", or False, default "empirical"
    n_resamples=1000,        # number of permutations, default 1000, if None, simple hypothesis test is performed
    log_transformed=True     # whether data is log-transformed, default True
)

de_res.to_df() # get results as pandas DataFrame
```

DE analysis results are stored in `DeaResult` object, which contains: Feature names, test statistics, log2 fold-changes, p-values, q-values, and other relevant information.

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
