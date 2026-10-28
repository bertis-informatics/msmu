# Normalization

Normalization is a crucial step in proteomics data analysis to correct for systematic biases and ensure comparability across samples. `msmu` provides several normalization methods to address different experimental designs and data characteristics.

## Log-transform intensities

The [`log2_transform()`](../reference/pp/log2_transform.md) function applies a log2 transformation to the quantification data in the specified modality. This transformation helps stabilize variance and make the data more normally distributed, which is beneficial for downstream statistical analyses. `msmu` assumes that [`log2_transform()`](../reference/pp/log2_transform.md) is applied on basal level of data before applying normalization methods.

```python
mdata = mm.pp.log2_transform(
    mdata,
    modality="psm",  # or "peptide", "protein"
    layer=None,      # optional; default None transforms .X
)
```

## Normalize sample intensities

The [`normalize()`](../reference/pp/normalize.md) function (also available as [`normalise()`](../reference/pp/normalise.md)) offers multiple normalization methods: median (`median`), median centering (`median_center`), quantile (`quantile`), total-intensity / constant-sum (`total_sum`, which rescales each sample so its summed intensity equals the median of the per-sample totals), and pairwise median (`pairwise_median`). Users can select the method that best suits their data and experimental design. All methods assume log2-transformed input. Normalization can also be performed independently within groups: pass `group_obs` (an `adata.obs` column, e.g., sample batch or type) and/or `group_var` (an `adata.var` column, e.g., `"filename"` for fractionated runs) to normalize within each group.

```python
mdata = mm.pp.normalize(
    mdata,
    modality="psm",           # or "peptide", "protein"
    method="median",          # required; one of "median", "median_center", "quantile", "total_sum", "pairwise_median"
    group_obs=None,           # optional adata.obs column: normalize within each sample group
    group_var=None,           # optional adata.var column (e.g., "filename"): normalize within each feature group
    layer=None,               # optional; default None normalizes .X
)
```

## Adjust PTM sites by protein abundance

The [`adjust_ptm_by_protein()`](../reference/pp/adjust_ptm_by_protein.md) function reads each PTM site relative to its parent protein's abundance in a matched `global proteome` dataset, so that a site's change is not simply its protein's change.

For the `ratio` method (the default), the protein's log intensity is subtracted from the site's — the slope-one relationship mass action predicts, and the estimator used by msqrob2PTM. It needs no fitting, which is what keeps it usable at the sample counts proteomics actually has.

For the `ridge` method, a slope is instead fitted per site and the residual is kept. Note that a single-predictor ridge retains only `Sxx / (Sxx + alpha)` of the least-squares slope, where `Sxx = (n - 1) * var(protein)`; a large `ridge_alpha` therefore turns the residual into a plain mean-centering and removes no protein signal at all.

```python
mdata = mm.pp.adjust_ptm_by_protein(
    mdata,
    global_mdata=global_mdata,   # the global proteome: a MuData, or the path of its .h5mu
    modality="phospho_site",     # ptm modality
    method="ratio",              # options: "ratio", "ridge". default "ratio"
    rescale=True,                # whether to rescale adjusted values. default True
    layer=None,                  # optional; default None adjusts .X
    ridge_alpha=None,            # ridge penalty; only used when method="ridge"
)
```

The global dataset must hold a `protein` modality and the `.uns["protein_map"]` that
[`infer_protein()`](../reference/pp/infer_protein.md) writes, quantified on the same sample names as the PTM data, in log2 space.
