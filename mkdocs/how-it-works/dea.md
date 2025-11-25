# Differential Expression Analysis (DEA)

## Overview
Differential Expression Analysis (DEA) identifies proteins or peptides with significant abundance changes between experimental conditions. `msmu` provides permutation-based statistical testing to assess differential expression while controlling the false discovery rate (FDR).

## `mm.tl.dea.permutation_test()`
The `permutation_test()` function performs a non-parametric permutation test to evaluate differential expression between two groups. It calculates p-values based on the distribution of test statistics obtained from permuted group labels.

This function uses `welch`'s t-statistic by default, which is suitable for unequal variances between groups. Other statistics such as `student` t-test, `wilcoxon` rank-sum test, and `med_diff` (median difference) are also available.

For `FDR` correction, `msmu` supports `empirical` (based on permutation p-values) from null-distribution, and `bh` (Benjamini-Hochberg) methods. `empirical` FDR is recommended when using permutation tests.

if sample sizes are small to meet n_permutations, all possible permutations are used to compute exact p-values (exact test).


```python
de = mm.tl.dea.permutation_test(
    mdata,
    modality="protein",      # or "peptide"
    category="condition",    # column in .obs defining groups
    ctrl="control",          # control group label
    expr="treated",          # experimental group label
    statistic="welch",      # options: "welch", "student", "wilcoxon", "med_diff", default "welch"
    fdr="empirical",        # options: "empirical", "bh", "storey", or False, default "empirical"
    n_permutations=1000,     # number of permutations, default 1000
)

de.to_df() # get results as pandas DataFrame
```

DE analysis results are stored in `PermutationTestResult` object, which contains:
- permutation_method: Method used for permutation (exact or randomised).
- statistic: The statistic used for the test (e.g., welch, wilcoxon, median_diff).
- features: Array of feature names (e.g., proteins, ptm sites).
- median_ctrl: Median values for control group.
- median_expr: Median values for experimental group.
- pct_ctrl: Percentage of non-missing values in control group.
- pct_expr: Percentage of non-missing values in experimental group.
- log2fc: Log2 fold change values.
- p_value: P-values for the permutation test.
- q_value: FDR-adjusted q-values.
- fc_pct_1: log2fc cutoff at 1%.
- fc_pct_5: log2fc cutoff at 5%.

DE results can be accessed as a pandas DataFrame using the `to_df()` method.

## Visualization of DEA Results
`msmu` provides visualisation function to explore DEA results with volcano plots.

```python
de.plot_volcano(
    log2fc_cutoff=1.0,    # log2 fold change cutoff for highlighting
    pval_cutoff=0.05,     # p-value cutoff for highlighting
    title="Volcano Plot",  # plot title
)
```
