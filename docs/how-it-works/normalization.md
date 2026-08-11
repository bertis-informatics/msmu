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

The `adjust_ptm_by_protein()` function normalizes PTM site quantifications by their corresponding protein abundances from `global proteome` (if available) data to account for changes in protein expression levels.

For `ridge` regression method, PTM site intensities are adjusted based on the fitted values from a ridge regression model that predicts PTM abundance using protein abundance as a predictor variable. This approach helps to isolate PTM-specific changes from overall protein expression variations.

And for `ratio` method, PTM site intensities are normalized by calculating the ratio of PTM abundance to protein abundance, providing a simple measure of PTM changes relative to protein levels.

```python
mdata = mm.pp.adjust_ptm_by_protein(
    mdata,
    global_mdata=global_mdata,   # MuData object for global proteome
    modality="phospho_site",     # ptm modality
    method="ridge",              # options: "ridge", "ratio". default "ridge"
    rescale=True,                # whether to rescale adjusted values. default True
    layer=None,                  # optional; default None adjusts .X
)
```
