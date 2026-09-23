# Normalization

Normalization is a crucial step in proteomics data analysis to correct for systematic biases and ensure comparability across samples. `msmu` provides several normalization methods to address different experimental designs and data characteristics.

## Log-transform intensities

The [`log2_transform()`](../reference/pp/log2_transform.md) function applies a log2 transformation to the quantification data in the specified modality. This transformation helps stabilize variance and make the data more normally distributed, which is beneficial for downstream statistical analyses. `msmu` assumes that [`log2_transform()`](../reference/pp/log2_transform.md) is applied on basal level of data before applying normalization methods.

```python
mdata = mm.pp.log2_transform(
    mdata,
    modality="psm",  # or "peptide", "protein"
    layer=None,      # optional; default None transforms .X
)
```

## Normalize sample intensities

The [`normalize()`](../reference/pp/normalize.md) function (also available as [`normalise()`](../reference/pp/normalise.md)) offers multiple normalization methods: median (`median`), median centering (`median_center`), quantile (`quantile`), total-intensity / constant-sum (`total_sum`, which rescales each sample so its summed intensity equals the median of the per-sample totals), and pairwise median (`pairwise_median`, see below). Users can select the method that best suits their data and experimental design. All methods assume log2-transformed input. Normalization can also be performed independently within groups: pass `group_obs` (an `adata.obs` column, e.g. sample batch or type) and/or `group_var` (an `adata.var` column, e.g. `"filename"` for fractionated runs) to normalize within each group.

```python
mdata = mm.pp.normalize(
    mdata,
    modality="psm",           # or "peptide", "protein"
    method="median",          # required; one of "median", "median_center", "quantile", "total_sum", "pairwise_median"
    group_obs=None,           # optional adata.obs column: normalize within each sample group
    group_var=None,           # optional adata.var column (e.g. "filename"): normalize within each feature group
    layer=None,               # optional; default None normalizes .X
)
```

### Pairwise median normalization

`median` centres every sample on the median of the values it observed. When missingness depends on intensity and samples differ in detection depth, that median is biased: a deeper run observes more low-abundance features, its median is pulled down, and after centring it is left too high relative to the same features in a shallower run. `pairwise_median` compares samples only on features they both observe.

- **Estimator.** For every pair of samples, the median log2 difference over their co-observed features. Each sample's shift is the mean of its pairwise medians over all samples (its own pair counting as 0), which is the equal-weight least-squares solution with the shifts summing to zero; the block's mean sample level is kept. This is the construction MaxLFQ uses to build a protein profile from pairwise peptide ratios (Cox et al. 2014), applied to runs instead of proteins; pairwise medians of shared features are also how directLFQ, IonQuant and FlashLFQ align samples. It is not MaxLFQ normalization or directLFQ normalization.
- **Assumption.** The majority of the features *shared* between two samples are unchanged.
- **Observed medians are no longer aligned.** After `pairwise_median`, per-sample medians (and box plots) of observed values can differ between samples with different missingness. This is intended: those differences reflect which features each sample observed, not loading.
- **Limitation.** A true global change between conditions (most features moving in one direction) cannot be told from a loading difference and is removed, as `median` does. Check the loading scheme for secretome, exosome or pull-down designs before relying on either method.
- **Unshared samples.** Every pair of samples in a block must observe at least one feature in common. Otherwise a `ValueError` names the offending sample pairs; this happens for PSM or precursor matrices across runs or TMT plexes. Normalize within `group_obs`, or after summarizing to a level (peptide, protein) where the samples share features. A pair that shares only a few features is aligned on those few, and a warning names it when it shares far fewer than the typical pair. Blocks holding a single sample are left unchanged. Data with little missingness (e.g. TMT within a plex) gain nothing over `median`.

### Inspect the normalization record

Every call writes what it did to each sample of each block to `adata.uns["normalisation"]`:

- `summary`: one row per sample and block with `sample`, `obs_group`, `var_group`, `method`, `layer`, `n_observed_features`, `shift_log2` (the applied shift; for `quantile` the median per-sample change), `location_before` and `location_after` (median of the observed values before and after).
- `blocks["<layer>|<obs_group>|<var_group>"]`: the same per-sample vectors, and for `pairwise_median` the `pair_median_log2` and `pair_shared_count` matrices (samples x samples) the shifts were derived from. The row mean of `pair_median_log2` is the shift; `pair_median_log2[s, t] - (shift[s] - shift[t])` is the pair's residual.

Records for the same layer are replaced on a repeated call; other layers' records are kept.

## Adjust PTM sites by protein abundance

The [`adjust_ptm_by_protein()`](../reference/pp/adjust_ptm_by_protein.md) function reads each PTM site relative to its parent protein's abundance in a matched `global proteome` dataset, so that a site's change is not simply its protein's change.

For the `ratio` method (the default), the protein's log intensity is subtracted from the site's — the slope-one relationship mass action predicts, and the estimator used by msqrob2PTM. It needs no fitting, which is what keeps it usable at the sample counts proteomics actually has.

For the `ridge` method, a slope is instead fitted per site and the residual is kept. Note that a single-predictor ridge retains only `Sxx / (Sxx + alpha)` of the least-squares slope, where `Sxx = (n - 1) * var(protein)`; a large `ridge_alpha` therefore turns the residual into a plain mean-centring and removes no protein signal at all.

```python
mdata = mm.pp.adjust_ptm_by_protein(
    mdata,
    global_mdata=global_mdata,   # the global proteome: a MuData, or the path of its .h5mu
    modality="phospho_site",     # ptm modality
    method="ratio",              # options: "ratio", "ridge". default "ratio"
    rescale=True,                # whether to rescale adjusted values. default True
    layer=None,                  # optional; default None adjusts .X
    ridge_alpha=None,            # ridge penalty; only used when method="ridge"
)
```

The global dataset must hold a `protein` modality and the `uns["protein_map"]` that
[`infer_protein()`](../reference/pp/infer_protein.md) writes, quantified on the same sample names as the PTM data, in log2 space.

### Pass the global dataset as a file to keep the workflow reproducible

Given as a `MuData`, the global container's own history merges into the result, and the adjustment
event has two parents. [`mm.pv.replay()`](../reference/pv/replay.md) and [`mm.pv.to_script()`](../reference/pv/to_script.md) accept a second `MuData` input only
for [`concat`](../reference/dt/concat.md), so the PTM workflow could be reproduced only up to this step. Given as a path, the file is recorded
as an input with its content hash — the way a reader's source file is — and the history stays one
chain, so the whole workflow replays and verifies. Replay needs the original files either way, so
this asks for nothing new.

```python
from pathlib import Path

mdata = mm.pp.adjust_ptm_by_protein(mdata, global_mdata=Path("global/02_processed.h5mu"))
```

Pass a `Path` rather than a `str`: provenance records a string as a file only when the parameter's
name says so, and this one's does not, so a string is read correctly but recorded without a hash.

### Where the result goes

The adjusted values **replace the matrix that was read** — `.X`, or `layers[layer]` when given — the same contract as [`log2_transform()`](../reference/pp/log2_transform.md), [`normalise()`](../reference/pp/normalise.md) and [`correct_batch_effect()`](../reference/pp/correct_batch_effect.md). Nothing is dropped: a site that could not be adjusted is set to `NaN` rather than left holding its raw abundance, so residuals and raw abundances never share a matrix.

To keep the unadjusted values for a side-by-side comparison, copy them into a layer first:

```python
mdata = mm.dt.save_layer(mdata, modality="phospho_site", layer="unadjusted")
mdata = mm.pp.adjust_ptm_by_protein(mdata, global_mdata=global_mdata)
```

Each site is annotated with how it was resolved:

- `var["denominator_group"]`: the global protein group used as the denominator, if one was found.
- `var["is_protein_adjusted"]`: whether an adjusted value was produced.
- `var["adjustment_status"]`: why, when it was not — `shared_groups` (the accessions span two or more quantified groups, so the signal is a sum over them and no single denominator is valid), `not_quantified`, `no_global_group`, `not_in_global_fasta` (the two searches used different databases), or `no_estimate` (the denominator was found but the estimator declined, as `ridge` does below three paired observations).

The denominator is located by translating the accessions a site was localized on through the global dataset's `uns["protein_map"]`. Nothing is looked up by peptide, so a PTM peptide the global run never observed — the normal case under enrichment — is still adjustable whenever its protein was quantified there.
