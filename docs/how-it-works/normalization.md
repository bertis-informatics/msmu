# Normalization

## Overview

Normalization is a crucial step in proteomics data analysis to correct for systematic biases and ensure comparability across samples. `msmu` provides several normalization methods to address different experimental designs and data characteristics.

## `log2_transform()`

The `log2_transform()` function applies a log2 transformation to the quantification data in the specified modality. This transformation helps stabilize variance and make the data more normally distributed, which is beneficial for downstream statistical analyses. `msmu` assumes that `log2_transform()` is applied on basal level of data before applying normalization methods.

```python
mdata = mm.pp.log2_transform(
    mdata,
    modality="psm",  # or "peptide", "protein"
    layer=None,      # optional; default None transforms .X
)
```

## `normalize()` (or `normalise()`)

The `normalize()` function offers multiple normalization methods: median (`median`), quantile (`quantile`), and total-intensity / constant-sum (`total_sum`, which rescales each sample so its summed intensity equals the median of the per-sample totals). Users can select the method that best suits their data and experimental design. All methods assume log2-transformed input. Normalization can also be performed independently within groups: pass `group_obs` (an `adata.obs` column, e.g. sample batch or type) and/or `group_var` (an `adata.var` column, e.g. `"filename"` for fractionated runs) to normalize within each group.

```python
mdata = mm.pp.normalize(
    mdata,
    modality="psm",           # or "peptide", "protein"
    method="median",          # options: "median", "quantile", "total_sum"; default "median"
    group_obs=None,           # optional adata.obs column: normalize within each sample group
    group_var=None,           # optional adata.var column (e.g. "filename"): normalize within each feature group
    layer=None,               # optional; default None normalizes .X
)
```

## `adjust_ptm_by_protein()`

The `adjust_ptm_by_protein()` function reads each PTM site relative to its parent protein's abundance in a matched `global proteome` dataset, so that a site's change is not simply its protein's change.

For the `ratio` method (the default), the protein's log intensity is subtracted from the site's — the slope-one relationship mass action predicts, and the estimator used by msqrob2PTM. It needs no fitting, which is what keeps it usable at the sample counts proteomics actually has.

For the `ridge` method, a slope is instead fitted per site and the residual is kept. Note that a single-predictor ridge retains only `Sxx / (Sxx + alpha)` of the least-squares slope, where `Sxx = (n - 1) * var(protein)`; a large `ridge_alpha` therefore turns the residual into a plain mean-centring and removes no protein signal at all.

```python
mdata = mm.pp.adjust_ptm_by_protein(
    mdata,
    global_mdata=global_mdata,   # MuData object for global proteome
    modality="phospho_site",     # ptm modality
    method="ratio",              # options: "ratio", "ridge". default "ratio"
    rescale=True,                # whether to rescale adjusted values. default True
    layer=None,                  # optional; default None reads .X
    ridge_alpha=None,            # ridge penalty; only used when method="ridge"
    adjusted_layer="protein_adjusted",  # layer the adjusted values are written to
)
```

### Where the result goes

Adjusted values are written to `layers["protein_adjusted"]` and **`.X` keeps the unadjusted intensities**. Nothing is dropped: a site that could not be adjusted stays in place with `NaN` in that layer, so the unadjusted analysis remains available and the two are never mixed in one matrix.

Each site is annotated with how it was resolved:

- `var["denominator_group"]`: the global protein group used as the denominator, if one was found.
- `var["is_protein_adjusted"]`: whether an adjusted value was produced.
- `var["adjustment_status"]`: why, when it was not — `shared_groups` (the accessions span two or more quantified groups, so the signal is a sum over them and no single denominator is valid), `not_quantified`, `no_global_group`, `not_in_global_fasta` (the two searches used different databases), or `no_estimate` (the denominator was found but the estimator declined, as `ridge` does below three paired observations).

The denominator is located by translating the accessions a site was localized on through the global dataset's `uns["protein_map"]`. Nothing is looked up by peptide, so a PTM peptide the global run never observed — the normal case under enrichment — is still adjustable whenever its protein was quantified there.
