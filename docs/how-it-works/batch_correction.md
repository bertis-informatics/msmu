# Batch Correction

## Overview

Batch effects are unwanted variations in the data that arise from differences in experimental conditions, such as different lots, runs, days, or operators. These variations can obscure true biological signals and lead to misleading conclusions. `msmu` provides functions to correct for discrete batch effects using methods like median centering and GIS/IRS for TMT data. For batch effect from continuous variables (e.g., run order) will be supported in future releases.

## `correct_batch_effect()`

The `correct_batch_effect()` function either:

- Median centering, which standardizes each features to have zero median, or
- GIS/IRS normalization, which rescales and corrects batch effect in TMT data using Global Internal Standard (GIS) channels.

```python
mdata = mm.pp.correct_batch_effect(
    mdata,
    modality="feature",     # or "peptide", "protein"
    method="gis",           # options: "median_center", "gis"
    gis_prefix="POOLED_"    # prefix for GIS channels
    rescale=True             # whether to rescale data
)

# or
mdata = mm.pp.correct_batch_effect(
    mdata,
    modality="feature",            # or "peptide", "protein"
    method="median_center",        # options: "median_center", "gis"
    rescale=True                   # whether to rescale data
)
```
