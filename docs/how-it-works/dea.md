# Differential expression (DE) analysis

Differential expression (DE) analysis compares protein or peptide abundance between experimental conditions. In `msmu`, [`mm.tl.run_de()`](../reference/tl/run_de.md) performs DE analysis and returns a `DeaResult`.

For a worked example using processed protein data, see the [DE tutorial](../tutorials/dea.ipynb).

## Input and comparison

Select the modality and the `.obs` column containing the condition labels with `modality` and `category`. The analysis reads `.X`, or `.layers[layer]` when a layer is specified. For a simple two-group comparison, `ctrl` and `expr` define the reference and experimental groups; a positive `log2fc` means higher abundance in `expr`.

For example, compare `G2` against `G1` using the condition labels in `mdata["protein"].obs`:

```python
de_res = mm.tl.run_de(
    mdata,
    modality="protein",
    category="factor value[condition]",
    ctrl="G1",
    expr="G2",
    min_pct=0.5,          # require observations in at least 50% of each group
    stat_method="limma",  # "limma", "welch", "student", or "wilcoxon"
    p_adjust="auto",      # multiple-testing correction; see options below
)
```

### Missing values and test eligibility

`min_pct` sets the minimum observed fraction required in every compared group or design cell. A value of `0.5` requires observations in at least 50% of samples in each group. The default, `0.0`, still requires enough observations for the selected test.

`run_de()` does not impute missing values. Returned results retain untestable features with `NaN` in `p_value` and `q_value`.

### Statistical methods

The default, `stat_method="limma"`, moderates per-feature variance estimates using information across features, as described by [Smyth (2004)](#ref-smyth-2004) and [Ritchie et al. (2015)](#ref-ritchie-2015). `msmu` uses [`inmoose`](https://inmoose.readthedocs.io/), described by [Colange et al. (2025)](#ref-colange-2025), for this analysis.

| `stat_method` | Test                                                |
| ------------- | --------------------------------------------------- |
| `"limma"`     | Empirical-Bayes moderated-t                          |
| `"welch"`     | Label permutation of Welch's t                       |
| `"student"`   | Label permutation of Student's t                     |
| `"wilcoxon"`  | Label permutation of the Wilcoxon rank-sum statistic |

The permutation methods compare the observed statistic with a null distribution obtained by rearranging group labels. `n_resamples` sets the requested number of label permutations. For small groups, the number of possible label permutations is limited, so increasing `n_resamples` may not add new information.

### Multiple testing and correction

`p_adjust` selects how the multiple-testing result in `q_value` is calculated. It leaves the test statistic, `log2fc`, and unadjusted `p_value` unchanged.

| `p_adjust`     | Method                         |
| -------------- | ------------------------------ |
| `"auto"`       | Default: `"bh"` for limma; `"empirical"` for permutation methods |
| `"empirical"`  | Permutation-based FDR estimate |
| `"bh"`         | Benjamini–Hochberg             |
| `"by"`         | Benjamini–Yekutieli            |
| `"bonferroni"` | Bonferroni                     |
| `"holm"`       | Holm–Bonferroni                |
| `"hochberg"`   | Hochberg                       |
| `"hommel"`     | Hommel                         |

FDR concerns the expected fraction of false discoveries among selected features. Family-wise error concerns the probability of at least one false discovery across the tested features. All methods write to `q_value`; for family-wise methods, this column contains adjusted p-values rather than FDR-based q-values. The standard correction methods are also described in the [R `p.adjust` documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html).

The standard corrections are applied to the non-missing p-values within the current comparison; untestable features retain `NaN`. To inspect uncorrected significance, use `p_value`. There is no `p_adjust="none"` option. See the [`run_de()` API reference](../reference/tl/run_de.md) for parameter details.

#### Empirical FDR estimation

With `p_adjust="empirical"`, `msmu` estimates FDR from the label-permutation null, following the framework discussed by [Xie et al. (2005)](#ref-xie-2005) and [Storey and Tibshirani (2003)](#ref-storey-2003):

```text
estimated FDR ≈ pi0 * mean null exceedances / observed discoveries
```

`pi0` is the estimated proportion of features with no true difference. Null exceedances and observed discoveries are counted at the same absolute test-statistic cutoff; observed discoveries include both true and false positives. Each `q_value` is the minimum estimated FDR over cutoffs that include that feature.

### Interactions and covariates

Limma also supports more complex designs. Set `interaction` to a second `.obs` factor and `interaction_levels` to two levels to compare the `expr - ctrl` effect between them. The contrast subtracts the effect at the second listed level from the effect at the first. Use `covariates` to adjust for additional `.obs` variables, such as batch; these change the reported model contrast from a raw group difference.

Limma requires an explicit `expr` group. For comparison against all groups other than `ctrl`, the permutation methods support `expr=None`. Design arguments and examples are in the [`run_de()` API reference](../reference/tl/run_de.md).

For a dataset with `condition`, `genotype`, and `batch` columns in the protein modality's `.obs`:

```python
# Two-group moderated-t
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

de_res.to_df()  # get results as a pandas DataFrame
```

## Results

Use `de_res.to_df()` to obtain a `pandas` `DataFrame`:

| Column                   | Interpretation                                                                 |
| ------------------------ | ------------------------------------------------------------------------------ |
| `features`               | Protein or peptide identifier                                                  |
| `repr_ctrl`, `repr_expr` | Group means, or medians for `wilcoxon`, on the input scale                     |
| `pct_ctrl`, `pct_expr`   | Percentage of samples with an observed value in each group                     |
| `log2fc`                 | Log2 fold change; `expr - ctrl` for a simple two-group comparison of log2 data |
| `statistic`              | Test statistic for the selected method                                         |
| `p_value`                | Unadjusted p-value                                                             |
| `q_value`                | Multiple-testing result using the selected `p_adjust` method                   |

Interpret effect size together with the adjusted result and observed coverage. A missing p-value or q-value indicates that significance could not be estimated for that feature.

## References

<a id="ref-colange-2025"></a>

> Colange, M., Appé, G., Meunier, L., Weill, S., Johnson, W. E., Nordor, A., and Behdenna, A. Bridging the gap between R and Python in bulk transcriptomic data analysis with InMoose. _Scientific Reports_ 15, 18104 (2025). [https://doi.org/10.1038/s41598-025-03376-y](https://doi.org/10.1038/s41598-025-03376-y).

<a id="ref-smyth-2004"></a>

> Smyth, G. K. Linear models and empirical Bayes methods for assessing differential expression in microarray experiments. _Statistical Applications in Genetics and Molecular Biology_ 3(1), 1–25 (2004). [https://doi.org/10.2202/1544-6115.1027](https://doi.org/10.2202/1544-6115.1027).

<a id="ref-ritchie-2015"></a>

> Ritchie, M. E., Phipson, B., Wu, D., Hu, Y., Law, C. W., Shi, W., and Smyth, G. K. limma powers differential expression analyses for RNA-sequencing and microarray studies. _Nucleic Acids Research_ 43(7), e47 (2015). [https://doi.org/10.1093/nar/gkv007](https://doi.org/10.1093/nar/gkv007).

<a id="ref-xie-2005"></a>

> Xie, Y., Pan, W., and Khodursky, A. B. A note on using permutation-based false discovery rate estimates to compare different analysis methods for microarray data. _Bioinformatics_ 21(23), 4280–4288 (2005). [https://doi.org/10.1093/bioinformatics/bti685](https://doi.org/10.1093/bioinformatics/bti685).

<a id="ref-storey-2003"></a>

> Storey, J. D. and Tibshirani, R. Statistical significance for genomewide studies. _Proceedings of the National Academy of Sciences_ 100(16), 9440–9445 (2003). [https://doi.org/10.1073/pnas.1530509100](https://doi.org/10.1073/pnas.1530509100).
