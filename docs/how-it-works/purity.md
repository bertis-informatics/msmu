# Precursor isolation purity

Precursor isolation purity (PIP) is a metric that quantifies the proportion of the target precursor ion signal relative to the total signal within the isolation window during MS/MS acquisition. High PIP values indicate that the isolated precursor is relatively free from co-isolated contaminants, which is crucial for accurate quantification, especially in isobaric labeling experiments like TMT.

In `msmu`, PIP is calculated using the [`compute_precursor_isolation_purity()`](../reference/tl/compute_precursor_isolation_purity.md) function, which uses [`pyopenms`](https://pyopenms.readthedocs.io/en/latest/user_guide/ms_data.html#example-precursor-purity) ([Röst et al., 2014](#ref-rost-2014)) to analyze MS1 spectra and determine the purity of each precursor ion.

```python
mdata = mm.tl.compute_precursor_isolation_purity(
    mdata,
    mzml_paths=["/path/to/mzml/files"],  # path to mzML files
    tolerance=20,                        # mass tolerance
    unit_ppm=True                        # mass tolerance in ppm
    )
```

## References

<a id="ref-rost-2014"></a>

> Röst, H. L., Schmitt, U., Aebersold, R., and Malmström, L. pyOpenMS: a Python-based interface to the OpenMS mass-spectrometry algorithm library. *Proteomics* 14(1), 74–77 (2014). [https://doi.org/10.1002/pmic.201300246](https://doi.org/10.1002/pmic.201300246).
