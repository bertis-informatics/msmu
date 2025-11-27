# Precursor Isolation Purity

## Overview

Precursor Isolation Purity (PIP) is a metric that quantifies the proportion of the target precursor ion signal relative to the total signal within the isolation window during MS/MS acquisition. High PIP values indicate that the isolated precursor is relatively free from co-isolated contaminants, which is crucial for accurate quantification, especially in isobaric labeling experiments like TMT.

In `msmu`, PIP is calculated using the `compute_precursor_isolation_purity()` function, which leverages the [`pyopenms`](https://pyopenms.readthedocs.io/en/latest/user_guide/ms_data.html#example-precursor-purity) library to analyze MS1 spectra and determine the purity of each precursor ion.

```python
mdata = mm.pp.compute_precursor_isolation_purity(
    mdata,
    mzml_paths=["/path/to/mzml/files"],  # path to mzML files
    tolerance=20,                        # mass tolerance
    unit_ppm=True                        # mass tolerance in ppm
    )
```