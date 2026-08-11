# Summarization

## Overview

The term `Summarization` refers to aggregating identification features and quantitative values as data move from one hierarchical level to the next (i.e., PSM/precursor -> peptide -> protein).

Summarization functions are provided as `to_*` methods, such as `to_peptide`, `to_protein`, and `to_ptm`.

The `Summarization` process generally involves:

1. **Feature selection**  
   Selecting features to include in the aggregation based on criteria such as peptide type (unique/shared), precursor isolation purity, or abundance.
2. **Intensity aggregation**  
   Aggregation of quantification values with either a column-wise reduction (`median`, `mean`, `sum`) or, at the protein level, a matrix rollup that models per-peptide response factors (`median_polish`, `directlfq`). See [Aggregation methods](#aggregation_methods) below.
3. **Computing identification confidence scores (PEP, q-value) at the new level when possible.**  
   Calculating PEP and q-values for the aggregated features using appropriate methods.

## Aggregation methods

The `agg_method` argument selects how quantification values are combined. Two families are available.

**Column-wise reductions** — `median` (default), `mean`, and `sum` collapse each sample's features independently. They make no assumption about the relationship between features and are available at every level (`to_peptide`, `to_protein`, `to_ptm`).

**Matrix rollups** — `median_polish` and `directlfq` treat the feature-by-sample block of a protein group as a whole matrix and estimate a per-sample protein profile while accounting for the fact that different peptides ionize with different efficiencies (per-peptide "response factors"):

- `median_polish` fits Tukey's additive model `value = overall + peptide_effect + sample_effect + residual` by iteratively sweeping out row and column medians; the per-sample estimate is `overall + sample_effect`. This follows R's `stats::medpolish`, the protein summarization used by MSstats ([Choi et al., Bioinformatics, 2014](https://doi.org/10.1093/bioinformatics/btu305)).
- `directlfq` applies the DirectLFQ rollup ([Ammar et al., 2023](https://doi.org/10.1016/j.mcpro.2023.100581)): it aligns each peptide's intensity trace onto a common within-group scale and takes the per-sample median of the aligned traces. DirectLFQ is MaxLFQ-inspired but a *distinct* algorithm from the classical MaxLFQ pairwise-ratio least-squares, so the two are correlated, not identical.

Because they model per-peptide response factors — meaningful when combining *distinct* peptides into a protein, but not when combining replicate PSMs of the *same* peptide — the matrix rollups are offered only at the peptide-to-protein step (`to_protein` and `to_ptm`). `to_peptide` accepts only the column-wise reductions.

**Important:** `median_polish` and `directlfq` are additive / log-space methods and must be applied to log2-transformed data. Call `mm.pp.log2_transform()` before summarizing with them.

## `to_peptide()`

`to_peptide()` function takes:

- `MuData` containing `psm` level modality

and returns

- `MuData` with `peptide` level modality

This step aggregates PSMs and their quantification values by `peptide` (non-redundant modified peptide).
Peptide-level PEP is calculated with `best_pep` method by default and peptide-level q-values are computed using a conservative approach when decoy information is available.

For quantification aggregation, the default method is `median`, and an optional `top_n` argument can be used to restrict aggregation using top N (e.g., top 3) features within each peptide. Feature ranking is based on `median_intensity` unless specified otherwise. Only the column-wise reductions (`median`, `mean`, `sum`) are available at the peptide level; the matrix rollups belong to the protein step (see [Aggregation methods](#aggregation_methods)).

In TMT studies, PSMs with low precursor isolation purity may be excluded prior to quantification aggregation to remove spectra with low quantitative accuracy. Precursor isolation purity should be computed with `mm.tl.compute_precursor_isolation_purity()` before calling `to_peptide()`. A `purity_threshold` (commonly `0.7`) can be applied during aggregation.

Note that filtering by `top_n` or `purity_threshold` affects quantification aggregation only and does not modify identification feature aggregation.

```python
mdata = mm.pp.to_peptide(
    mdata,
    agg_method="median",            # default
    purity_threshold=0.7,           # for tmt data
    top_n=None,                     # default
    rank_method="median_intensity",  # default
    layer=None,                     # default; read from .X, or name a layer to summarise instead
    calculate_q=True,               # default; set False to skip peptide-level PEP/q-value
    )
```

## `to_protein()`

`to_protein()` function takes:

- `MuData` containing `peptide` modality with inferred `protein_group` and `peptide_type`

  and returns:

- `MuData` with `protein` level modality

Protein-level summarization requires the `protein_group` and `peptide_type` columns, which are generated by `mm.pp.infer_protein()` from peptide-level data.
Details are provided in the [Protein Inference](../../how-it-works/inference/) section. Briefly:

- `protein_group` contains the inferred proteins for each peptide.
- `peptide_type` indicates whether a peptide is "unique" or "shared".

Only "unique" peptides are used for protein group intensity aggregation; "shared" peptides are excluded.

As in peptide-level aggregation, protein group level `PEP` and `q-value` are computed when possible.

The default settings use `top_n=3` with ranking by `median_intensity`, so only the top three peptides per protein group contribute to quantification.

Beyond the column-wise reductions, `to_protein` also accepts the matrix rollups `median_polish` and `directlfq`, which estimate a per-sample protein profile from the whole peptide-by-sample block while accounting for per-peptide response factors (see [Aggregation methods](#aggregation_methods)). Both require log2-transformed input, so run `mm.pp.log2_transform()` first. They are typically combined with `top_n=None` so that all peptides inform the estimate.

```python
# Infer protein group from mdata (containing peptide modality)
mdata = mm.pp.infer_protein(mdata)

# Summarize peptides to protein group
mdata = mm.pp.to_protein(
    mdata,
    agg_method="median",            # "median" (default), "mean", "sum", "median_polish", "directlfq"
    top_n=3,                        # default; use None with the matrix rollups
    rank_method="median_intensity",  # default
    layer=None,                     # default; read from .X, or name a layer to summarise instead
    calculate_q=True,               # default; set False to skip protein-level PEP/q-value
    )
```

## `to_ptm()`

To summarize modified peptide into post-translational modification (PTM) sites, `to_ptm()` uses the subset of peptides that contain the specified modification and then performs several steps to assign PTM positions at the protein level.

Internally, the function performs:

1. Filtering data with only modified peptides with modi_identifier
2. Extracting modified sites from peptide
3. Assigning peptide-level site labels
4. Exploding peptides to single proteins for per-protein site labeling
5. Mapping the site to the corresponding position in each protein
6. Merging single-protein results back into protein groups
7. Grouping by modified peptide and peptide-site combination
8. Merging site metadata with peptide-level quantification

`to_ptm()` function takes:

- `MuData` containing `peptide` modality and attached FASTA file

and returns:

- `MuData` with `ptm_site` level modality

A FASTA file is required because PTM sites must be mapped to protein-sequence coordinates. FASTA can be attached using `mm.utils.attach_fasta()`.

The argument `modi_name` determines the modality name (e.g., "phospho" -> "phospho_site"), and the `modification` string is used to identify modified peptides.

`agg_method` can be selected among the methods described in [Aggregation methods](#aggregation_methods); the matrix rollups `median_polish` and `directlfq` are available here as well (on log2-transformed data).

```python
mdata = mm.utils.attach_fasta("fasta/file/path.fasta")

mdata = mm.pp.to_ptm(
    mdata,
    modi_name="phospho",
    modification="[+79.9663]",
    agg_method="median",        # default
    top_n=None                  # default
    )
```
