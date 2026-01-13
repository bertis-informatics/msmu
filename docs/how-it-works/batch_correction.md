# Batch Correction

## Overview

Batch effects are unwanted variations in the data that arise from differences in experimental conditions, such as different lots, runs, days, or operators. These variations can obscure true biological signals and lead to misleading conclusions. `msmu` provides functions to correct for discrete batch effects using methods like median centering and GIS/IRS for TMT data. Support for batch effects arising from continuous variables (e.g., run order) will be added in future releases.

## `correct_batch_effect()`

The `correct_batch_effect()` function either:

- Median centering, which standardizes each features to have zero median (can be re-scaled).
- GIS/IRS normalization, which rescales and corrects batch effect in TMT data using Global Internal Standard (GIS) channels.
- Combat batch effect correction ([pycombat](https://github.com/epigenelabs/pyComBat)).
- Continuous batch effect correction using lowess regression (referred from [Diagnostics and correction of batch effects in large‐scale proteomic studies: a tutorial](https://pmc.ncbi.nlm.nih.gov/articles/PMC8447595/)).
  

```python
mdata = mm.pp.correct_batch_effect(
    mdata,
    modality="feature",                      # or "peptide", "protein"
    layer=None,                              # layer to correct, default is .X
    category="batch",                        # batch information column in .obs
    method="gis",                            # options: "median_center", "gis", "combat", "continuous"
    rescale=True,                            # whether to rescale data with median value of original data. Default is True (for GIS, median_center, continuous)
    gis_sample=["POOLED_1", "POOLED_2"],     # GIS channel names (for TMT data only)
    drop_gis=True,                           # whether to drop GIS channels after correction. Default is True
    log_transformed=True                     # whether data is log-transformed. Default is True
)

# or
mdata = mm.pp.correct_batch_effect(
    mdata,
    modality="feature",            
    category="batch",
    method="median_center",
)

# or
mdata = mm.pp.correct_batch_effect(
    mdata,
    modality="protein",
    category="run_order",
    method="continuous",
)
```
