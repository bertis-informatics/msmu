# Batch correction

Batch effects are unwanted variations in the data that arise from differences in experimental conditions, such as different lots, runs, days, or operators. These variations can obscure true biological signals and lead to misleading conclusions. `msmu` corrects them with median centering, GIS/IRS (for TMT), ComBat, and a continuous (lowess) correction for ordered covariates such as run order.

## Correct batch effects

The [`correct_batch_effect()`](../reference/pp/correct_batch_effect.md) function offers four correction methods:

- Median centering, which removes each batch's per-feature median.
- GIS/IRS normalization, which corrects batch effect in TMT data using global internal standard (GIS) channels ([Plubell et al. (2017)](#ref-plubell-2017)).
- ComBat batch effect correction provided by [`inmoose`](https://github.com/epigenelabs/inmoose) ([Colange et al., 2025](#ref-colange-2025)).
- Continuous batch effect correction using lowess regression (as described by [Čuklina et al. (2021)](#ref-cuklina-2021)).

For `gis`, `median_center`, and `continuous`, the per-feature abundance scale is restored automatically after correction (geometric mean of the per-batch GIS levels for `gis` — classic IRS; per-feature overall median otherwise). The decision is made for the whole modality: if the matrix has cross-batch structure (at least one feature observed in ≥2 batches, as at the peptide/protein level) every feature is restored, so a lone single-batch feature stays on the same abundance scale as the rest; if the matrix is fully block-diagonal (no feature spans ≥2 batches, e.g., a per-plex-split PSM matrix) nothing is restored and the output stays reference-relative, ready to roll up.


```python
mdata = mm.pp.correct_batch_effect(
    mdata,
    modality="peptide",                      # or "protein"
    layer=None,                              # layer to correct, default is .X
    category="batch",                        # batch information column in .obs
    method="gis",                            # options: "median_center", "gis", "combat", "continuous"
    gis_samples=["POOLED_1", "POOLED_2"],    # GIS channel names (for TMT data only)
    drop_gis=True,                           # whether to drop GIS channels after correction. Default is True
    log_transformed=True                     # whether data is log-transformed. Default is True
)

# or
mdata = mm.pp.correct_batch_effect(
    mdata,
    modality="peptide",
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

## Choose a method and interpret the output

- **Level.** Correct **after** summarizing to peptide/protein, not on the raw PSM matrix. At peptide/protein a feature is shared across batches, so the scale restores to abundance and the matrices are dense and small. A per-plex-split PSM matrix is block-diagonal (each feature lives in one plex): the correction still runs, but the output stays reference-relative — roll up first, then correct.
- **`gis`** — TMT/IRS. Needs the pooled reference (GIS) channels named in `gis_samples`, present in every plex. Assumes log-transformed input; normalizes each feature to its plex reference and restores the IRS geometric-mean scale.
- **`median_center`** — removes each batch's per-feature median; assumes a balanced design (batch medians comparable). Works at any level.
- **`combat`** — `inmoose`'s ComBat implementation; models batch location/scale and handles its own scaling (no separate restore).
- **`continuous`** — lowess against an **ordered** covariate (e.g., run / acquisition order in `category`), not a nominal batch label.
- **Output scale.** Abundance on a shared (peptide/protein) matrix; reference-relative (ratio / centered) on a block-diagonal (raw split-PSM) matrix. The restore is a per-feature constant, so it does not change differential-expression contrasts — it only sets the abundance scale.

## References

<a id="ref-plubell-2017"></a>

> Plubell, D. L., Wilmarth, P. A., Zhao, Y., Fenton, A. M., Minnier, J., Reddy, A. P., Klimek, J., Yang, X., David, L. L., and Pamir, N. Extended multiplexing of tandem mass tags (TMT) labeling reveals age and high fat diet specific proteome changes in mouse epididymal adipose tissue. *Molecular & Cellular Proteomics* 16(5), 873–890 (2017). [https://doi.org/10.1074/mcp.M116.065524](https://doi.org/10.1074/mcp.M116.065524).

<a id="ref-colange-2025"></a>

> Colange, M., Appé, G., Meunier, L., Weill, S., Johnson, W. E., Nordor, A., and Behdenna, A. Bridging the gap between R and Python in bulk transcriptomic data analysis with InMoose. *Scientific Reports* 15, 18104 (2025). [https://doi.org/10.1038/s41598-025-03376-y](https://doi.org/10.1038/s41598-025-03376-y).

<a id="ref-cuklina-2021"></a>

> Čuklina, J., Lee, C. H., Williams, E. G., Sajic, T., Collins, B. C., Rodríguez Martínez, M., Sharma, V. S., Wendt, F., Goetze, S., Keele, G. R., Wollscheid, B., Aebersold, R., and Pedrioli, P. G. A. Diagnostics and correction of batch effects in large-scale proteomic studies: a tutorial. *Molecular Systems Biology* 17(8), e10240 (2021). [https://doi.org/10.15252/msb.202110240](https://doi.org/10.15252/msb.202110240).
