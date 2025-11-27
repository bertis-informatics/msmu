# Batch Correction

## Overview

## `correct_batch_effect()`
The `correct_batch_effect()` function standardises the features in the specified modality to have zero median. Or scale features with GIS/IRS method for TMT data to correct for batch effects using Global Internal Standard (GIS) channels.

```python
mdata = mm.pp.correct_batch_effect(
    mdata,
    modality="feature",     # or "peptide", "protein"
    method="gis",           # options: "median_center", "gis"
    gis_prefix="POOLED_"    # prefix for GIS channels
)

# or
mdata = mm.pp.correct_batch_effect(
    mdata,
    modality="feature",            # or "peptide", "protein"
    method="median_center",        # options: "median_center", "gis"
)
```