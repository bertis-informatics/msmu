# Normalization

## Overview

Normalization is a crucial step in proteomics data analysis to correct for systematic biases and ensure comparability across samples. `msmu` provides several normalization methods to address different experimental designs and data characteristics.

## `log2_transform()`

The `log2_transform()` function applies a log2 transformation to the quantification data in the specified modality. This transformation helps stabilize variance and make the data more normally distributed, which is beneficial for downstream statistical analyses. `msmu` assumes that `log2_transform()` is applied on basal level of data before other normalization methods.

```python
mdata = mm.pp.log2_transform(
    mdata,
    modality="psm"  # or "peptide", "protein"
)
```

## `normalize()` (or `normalise()`)

The `normalize()` function offers multiple normalization methods, including median (`median`) and quantile (`quantile`) normalization. Users can select the method that best suits their data and experimental design. For fractionated TMT data, setting the `fraction` argument to `True` ensures that normalization is performed within each fraction separately.

```python
mdata = mm.pp.normalize(
    mdata,
    modality="psm",           # or "peptide", "protein"
    method="median",          # options: "median", "quantile", default "median"
    fraction=False            # whether data is fractionated
)
```

## `adjust_ptm_by_protein()`

The `adjust_ptm_by_protein()` function normalizes PTM site quantifications by their corresponding protein abundances from `global proteome` data to account for changes in protein expression levels

For `ridge` regression method, PTM site intensities are adjusted based on the fitted values from a ridge regression model that predicts PTM abundance using protein abundance as a predictor variable. This approach helps to isolate PTM-specific changes from overall protein expression variations.

And for `ratio` method, PTM site intensities are normalized by calculating the ratio of PTM abundance to protein abundance, providing a direct measure of PTM changes relative to protein levels.

```python
mdata = mm.pp.adjust_ptm_by_protein(
    mdata,
    global_mdata=global_mdata,   # MuData object for global proteome
    ptm_mod="phospho_site",      # ptm modality
    method="ridge",              # options: "ridge", "ratio". default "ridge"
    rescale=True                 # whether to rescale adjusted values. default True
)
```
