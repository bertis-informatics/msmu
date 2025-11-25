# Normalisation

## Overview
Normalization is a crucial step in proteomics data analysis to correct for systematic biases and ensure comparability across samples. `msmu` provides several normalization methods to address different experimental designs and data characteristics.

## `log2_transform()`
The `log2_transform()` function applies a log2 transformation to the quantification data in the specified modality. This transformation helps stabilize variance and make the data more normally distributed, which is beneficial for downstream statistical analyses. `msmu` assumes that `log2_transfrom()` is applied on basal level of data before other normalisation methods.

```python
mdata = mm.pp.log2_transform(
    mdata,
    modality="feature"  # or "peptide", "protein"
)
```

## `normalise()`
The `normalise()` function offers multiple normalization methods, including median centering (`median`) and quantile (`quantile`) normalization. Users can select the method that best suits their data and experimental design. For fractionated TMT data, setting the `fraction` argument to `True` ensures that normalization is performed within each fraction separately.

```python
mdata = mm.pp.normalise(
    mdata,
    modality="feature",        # or "peptide", "peptide"
    method="median",          # options: "median", "quantile"
    fraction=False            # whether data is fractionated
)
```

## `scale_feature()`
The `scale_feature()` function standardises the features in the specified modality to have zero median. Or scale features with GIS/IRS method for TMT data to correct for batch effects using Global Internal Standard (GIS) channels.

```python
mdata = mm.pp.scale_feature(
    mdata,
    modality="feature",     # or "peptide", "protein"
    method="gis",           # options: "median", "gis"
    gis_prefix="POOLED_"    # prefix for GIS channels

# or
mdata = mm.pp.scale_feature(
    mdata,
    modality="feature",     # or "peptide", "protein"
    method="median_center",        # options: "median", "gis"
)
