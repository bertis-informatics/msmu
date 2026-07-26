# Differential Expression (DE) Analysis

## Overview

Differential Expression (DE) Analysis identifies proteins or peptides with significant abundance changes between experimental conditions. `msmu` exposes DE analysis through a single function, `mm.tl.run_de()`, with two complementary engines selected by `stat_method`: **limma moderated-t** (the default) — a parametric empirical-Bayes method that keeps power at the small sample sizes typical of proteomics and supports interactions and covariates — and an opt-in non-parametric **label-permutation test**. Both control the false discovery rate (FDR).

## `mm.tl.run_de()`

The `stat_method` argument selects the engine. It defaults to **`limma`** — the moderated-t test described in [limma moderated-t](#limma_moderated-t) below — because at the small sample sizes common in proteomics a permutation null is degenerate (see the q-value floor note below). Passing `welch`, `student`, or `wilcoxon` instead runs a non-parametric **label-permutation test**, where p-values come from the distribution of test statistics obtained by permuting the group labels.

> **The default engine changed.** `stat_method` previously defaulted to `welch` (permutation); it now defaults to `limma`. `run_de()` emits a one-time notice when limma runs — pass `stat_method` explicitly to state your choice.

Two knobs are no longer needed, because both are now decided by the engine:

- **The fold-change measure follows the test.** The central tendency behind `log2FC` (and the group `repr_ctrl` / `repr_expr` values) is fixed by the statistic, so significance and effect size describe the same quantity: `welch`, `student` and `limma` are mean-based, `wilcoxon` is median-based. There is no `measure` argument. (For `wilcoxon`, the reported difference of medians is a pragmatic proxy for the Hodges–Lehmann shift that the rank-sum statistic actually localizes.)
- **The q-value method follows the engine.** The permutation engines report an empirical (permutation) FDR; limma reports Benjamini-Hochberg q-values. Both always return `p_value` and `q_value`, so there is no `fdr` argument — for uncorrected significance, read the `p_value` column.

`min_pct` sets the minimum non-missing coverage a feature needs **in every group**, not in at least one, applied as a count — `max(1, ceil(min_pct × n))` non-missing values per group. The default `0.0` imposes only the **estimability floor**: every group needs at least one non-missing value (and limma additionally needs residual degrees of freedom), so every feature whose contrast can be estimated is tested. Requiring both groups — rather than at least one — is what keeps that contrast estimable without imputation, since a feature absent from one group has no contrast to estimate. Raise `min_pct` as an **opt-in stringency knob** to also demand a minimum coverage per group: for example `0.5` requires 2 of 3, 2 of 4, or 3 of 5, the count most commonly used in the field. It is a convention rather than a derived optimum — and above the estimability floor it is a pure extra filter whose reach grows with the missing rate (a no-op on complete data, but dropping most features, disproportionately the significant ones, on sparse data) — so choose it from the coverage you observe (`pct_ctrl` / `pct_expr`), not from which features come out significant.

Features below the threshold are **not dropped from the result** — they stay as rows with their descriptive columns filled (`repr_ctrl` / `repr_expr`, `pct_ctrl` / `pct_expr`) but with `p_value` and `q_value` set to `NaN`, since no test was run. In particular, **on/off features — present in one group and absent in the other — are reported this way rather than receiving a p-value**: workflows that test them do so by imputing the missing group, which `msmu` does not. Read them off the detection percentages `pct_ctrl` / `pct_expr` (e.g. `pct_ctrl == 0` with `pct_expr == 100`). Note that sorting or filtering the result by `p_value` / `q_value` moves these `NaN` rows to the end.

See more details in the [`msmu.tl.run_de`](../../reference/tl/run_de/) and usage examples in the tutorial [`DE Analysis`](../../tutorials/dea/).

```python
de_res = mm.tl.run_de(
    mdata,
    modality="protein",      # or "peptide"
    category="condition",    # column in .obs defining groups
    ctrl="control",          # control group label
    expr="treated",          # experimental group label
    # min_pct=0.5,           # (opt-in) also require >=50% coverage per group; default 0.0 tests every estimable feature
    log_transformed=True,    # whether data is log-transformed, default True
)                            # stat_method omitted -> the default limma engine runs

de_res.to_df() # get results as pandas DataFrame
```

DE analysis results are stored in `DeaResult` object, which contains: Feature names, test statistics, log2 fold-changes, p-values, q-values, and other relevant information.

DE results can be accessed as a pandas `DataFrame` using the `to_df()` method.

## limma moderated-t

This is the **default** engine — `stat_method="limma"`, or simply omit the argument. It borrows information across features through an empirical-Bayes shrinkage of the per-feature variance, which stabilizes each variance estimate and recovers power at small sample sizes. `msmu`'s limma is a thin layer over [inmoose](https://inmoose.readthedocs.io/) (a Python port of Smyth's limma, validated numerically against R limma) and reports Benjamini-Hochberg q-values. The moderated-t method is due to [Smyth, Statistical Applications in Genetics and Molecular Biology, 2004](https://doi.org/10.2202/1544-6115.1027), and the limma framework is described in [Ritchie et al., Nucleic Acids Research, 2015](https://doi.org/10.1093/nar/gkv007).

Because limma builds an explicit design matrix, it supports designs the permutation path does not:

- **Main effect** (default) — the two-group contrast `expr - ctrl`.
- **Interaction / difference-in-differences** — set `interaction` to a second `.obs` factor and `interaction_levels` to two of its levels to test whether the `expr - ctrl` effect differs between them (e.g. "does the treatment effect differ by genotype").
- **Covariate adjustment** — pass `covariates` (obs columns) to adjust the contrast for nuisance variables. Numeric covariates are passed through; categorical ones are one-hot encoded.

limma requires an explicit `expr` group (`expr=None` is not supported — for a "vs all other groups" comparison use one of the permutation methods), assumes log2-transformed input, and a positive `log2FC` means higher in `expr`. `n_resamples` does not apply to limma.

```python
# Two-group moderated-t (recommended for small sample sizes)
de_res = mm.tl.run_de(
    mdata,
    modality="protein",
    category="condition",
    ctrl="control",
    expr="treated",
    stat_method="limma",
)

# Interaction: does the treated-vs-control effect differ between genotypes,
# adjusting for batch?
de_res = mm.tl.run_de(
    mdata,
    modality="protein",
    category="condition",
    ctrl="control",
    expr="treated",
    stat_method="limma",
    interaction="genotype",
    interaction_levels=["wild_type", "knockout"],
    covariates=["batch"],
)

de_res.to_df()  # get results as pandas DataFrame
```

## Permutation test

Passing `welch`, `student`, or `wilcoxon` to `stat_method` opts out of limma and into a non-parametric **label-permutation test**. The statistic is Welch's t (`welch`, suitable for unequal variances between groups), Student's t (`student`), or the Wilcoxon rank-sum W (`wilcoxon`).

These engines **always** permute: `n_resamples` is the number of label shuffles — a positive integer, default `1000` — and **not** an on/off switch. For a parametric analysis, use `stat_method="limma"`.

If the design has fewer distinct label splits than `n_resamples`, every split is enumerated and the p-values are exact (exact test).

`p-value` from the permutation test is computed with the proportion of permuted statistics that are as extreme or more extreme than the observed statistic in null distribution with two-sided test.

`q-value` with `empirical` FDR is calculated by `E[FDR] = pi0 * E[FP] / E[TP]` referred to [Yang Xie et al., Bioinformatics, 2011.](https://academic.oup.com/bioinformatics/article/21/23/4280/194680) and [Storey et al., 2003](https://www.pnas.org/doi/epdf/10.1073/pnas.1530509100).

With very small groups the permutation null has few distinct label splits — only 20 for a 3-vs-3 comparison — and because the observed labelling is itself one of them, the empirical/BH q-value is floored (around `0.068` for 3-vs-3) regardless of effect size. **This is why `limma` is the default.** `run_de()` warns when a permutation design cannot reach `q < 0.05`; the floor drops below 0.05 by roughly 4-vs-4.

```python
de_res = mm.tl.run_de(
    mdata,
    modality="protein",
    category="condition",
    ctrl="control",
    expr="treated",
    stat_method="welch",     # "welch", "student" or "wilcoxon"
    n_resamples=1000,        # number of label shuffles (positive integer)
)
```

## Visualization of DEA Results

`msmu` provides visualization function to explore DEA results with volcano plots.

```python
de_res.plot_volcano(
    log2fc_threshold=None,  # (optional) log2 fold-change line; default None uses the fc_pct_5 guidance line
    pval_threshold=0.05,    # (optional) p-value cutoff line, default 0.05
    label_top=5,            # (optional) number of top significant features to label, default None (no labels)
)
```

With `log2fc_threshold=None` the plot uses the fold-change guidance line (`fc_pct_5`), a label-permutation percentile of the log2 fold change that is computed for every engine. It is deliberately **not** produced for limma interaction or covariate-adjusted contrasts, whose reported `log2FC` (a difference-in-differences or an adjusted coefficient) is not on the scale of the raw two-group null — for those, pass `log2fc_threshold` explicitly.
