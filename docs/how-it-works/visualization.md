# Visualization Overview

`msmu._plotting` wraps Plotly to provide ready-made QC and exploratory plots for MuData objects. The module is structured around data preparation helpers and lightweight plot wrappers so you can compose figures with consistent defaults while still passing Plotly kwargs to tweak layout.

## Common parameters and behaviors

- `mdata`: required `MuData` containing the modality to plot.
- `modality`: defaults vary by plot (`feature`, `peptide`, `protein`)
- `groupby`: observation column used to split traces/groups (e.g., `filename`, `condition`). If omitted, falls back to `obs_column`.
- `obs_column`: observation column used for labeling/group resolution; all elements should be unique. If omitted or no column exists, creates `__obx_idx__` column from the index of `obs`
- `colorby`: optional obs column for coloring; only applied when `groupby` equals `obs_column`.
- `ptype`: plot style selector (`hist`, `box`, `vln`, etc.).
- `**kwargs`: forwarded to `go.Figure.update_layout` for per-plot overrides.

## Example

> Uszkoreit, J., Barkovits, K., Pacharra, S., Pfeiffer, K., Steinbach, S., Marcus, K., & Eisenacher, M. (2022). Dataset containing physiological amounts of spike-in proteins into murine C2C12 background as a ground truth quantitative LC-MS/MS reference. Data in Brief, 43, 108435.

### mdata.obs

| set | sample_id  | sample_name | condition | replicate |
| --- | ---------- | ----------- | --------- | --------- |
| S1  | QExHF04026 | G1-1        | G1        | 1         |
| S1  | QExHF04028 | G2-1        | G2        | 1         |
| S1  | QExHF04030 | G3-1        | G3        | 1         |
| S1  | QExHF04032 | G4-1        | G4        | 1         |
| S1  | QExHF04034 | G5-1        | G5        | 1         |
| S1  | QExHF04036 | G1-2        | G1        | 2         |
| S1  | QExHF04038 | G2-2        | G2        | 2         |
| S1  | QExHF04040 | G3-2        | G3        | 2         |
| S1  | QExHF04042 | G4-2        | G4        | 2         |
| S1  | QExHF04044 | G5-2        | G5        | 2         |
| S1  | QExHF04046 | G1-3        | G1        | 3         |
| S1  | QExHF04048 | G2-3        | G2        | 3         |
| S1  | QExHF04050 | G3-3        | G3        | 3         |
| S1  | QExHF04052 | G4-3        | G4        | 3         |
| S1  | QExHF04054 | G5-3        | G5        | 3         |

### `plot_id`

```python
mm.pl.plot_id(mdata, "protein", groupby="sample_name")
```

![](../assets/images/visualization_id_1.png)

```python
mm.pl.plot_id(mdata, "protein", groupby="condition")
```

![](../assets/images/visualization_id_2.png)

### `plot_intensity`

```python
mm.pl.plot_intensity(mdata, "protein", groupby="sample_name", ptype="hist")
```

![](../assets/images/visualization_intensity_1.png)

### `plot_missingness`

```python
mm.pl.plot_missingness(mdata, "protein")
```

![](../assets/images/visualization_missingness_1.png)

### `plot_var`

```python
mm.pl.plot_var(mdata, "feature", groupby="sample_name", var_column="charge", ptype="stacked_bar")
```

![](../assets/images/visualization_var_1.png)

```python
mm.pl.plot_var(mdata, "feature", groupby="sample_name", var_column="peptide_length", ptype="vln")
```

![](../assets/images/visualization_var_2.png)

### `plot_pca` & `plot_umap`

```python
mm.pl.plot_pca(mdata, "protein", groupby="condition")
```

![](../assets/images/visualization_pca_1.png)

### `plot_correlation`

```python
mm.pl.plot_correlation(mdata, "protein")
```

![](../assets/images/visualization_correlation_1.png)

### `plot_upset`

```python
mm.pl.plot_upset(mdata, "protein", groupby="condition")
```

![](../assets/images/visualization_upset_1.png)
