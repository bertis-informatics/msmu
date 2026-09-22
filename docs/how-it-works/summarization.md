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

## Identification q-values and tied PEPs

With `calculate_q=True` and decoy information available, `to_peptide()` and
`to_protein()` use the same target–decoy calculation. Exactly equal PEP values
form one group; distinct PEP values are never rounded together. Groups are
ordered by increasing PEP, and cumulative target (`T`) and decoy (`D`) counts
are evaluated only after including an entire group:

`FDR = min(1, (D + 1) / T)`

The q-value is the reverse cumulative minimum of these group FDRs. Every target
and decoy in a group receives the same q-value, regardless of input row order.
For example, five targets and one decoy with the same PEP all receive
`q = (1 + 1) / 5 = 0.4`, wherever the decoy appears in the input. Boundaries
with no cumulative targets retain undefined (`NaN`) q-values. Missing PEPs
form a final group, preserving the previous NaN-last ordering.

This replaces the previous row-wise treatment of ties. For the same inputs at
one calculation step, q-values can stay the same or increase, so fewer features
may pass a given cutoff. Untied inputs keep their previous q-values. Changes to
earlier filtering can also change the inputs and results of later steps.

Existing provenance histories retain their original expected hashes and may
fail strict replay with this policy. Rerun the original workflow from its source
files with hashing enabled to record a new baseline; keep the old history for
comparison. Do not overwrite its expected hashes to make verification pass.

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

1. Parsing each modified peptide into residues and their modification tags, and keeping the peptidoforms that carry a target modification
2. Assigning peptide-level site labels from the positions of the residues that carry it
3. Exploding peptides to their own accessions for per-protein site labeling
4. Mapping the site to the corresponding position in each protein
5. Grouping by modified peptide and peptide-site combination
6. Merging site metadata with peptide-level quantification

A site's position counts residues only: everything inside a modification tag's brackets is tag
text, however many letters it holds, so `AC(UniMod:4)M(UniMod:35)PSGS(UniMod:21)YTK` and
`AC[+57.0215]M[+15.9949]PSGS[+79.9663]YTK` both put the phosphate on residue 7. A tag written before
the first residue (an N-terminal modification) belongs to residue 1. Every parse is checked against
the peptide's `stripped_peptide`, and a notation msmu cannot read raises rather than yield misplaced
sites.

Sites are localized from the accessions in the peptide's `proteins` column, not from an inferred
`protein_group`, so a site id depends only on the peptide and the FASTA — the same PTM data yields
the same sites whether or not it was processed alongside a global dataset.

`to_ptm()` function takes:

- `MuData` containing `peptide` modality and attached FASTA file

and returns:

- `MuData` with `ptm_site` level modality

A FASTA file is required because PTM sites must be mapped to protein-sequence coordinates. FASTA can be attached using `mm.utils.attach_fasta()`.

Attach **the FASTA the search used**. The search engine found each peptide in the sequence it
reports, so an accession the attached FASTA does not hold — or a sequence of it that does not
contain the peptide, as after a database release change — is a match `to_ptm` cannot reproduce. Such
accessions are dropped: sites are lost, and a site that should have been reported as spanning two
protein groups can be adjusted as if it were unambiguous. `to_ptm` counts these and warns, naming
examples. Contaminant accessions (`Cont_`) are reported at `INFO` instead, since search engines add
contaminant entries a user FASTA routinely lacks. Site positions are always coordinates in the
attached FASTA.

The argument `modi_name` determines the modality name (e.g., "phospho" -> "phospho_site"). The
`modification` argument is the modification tag exactly as it appears in the `peptide` column,
brackets and case included, or a list of tags to summarise into one modality. A tag may be qualified
by its residue (`"S[167]"`) to match that residue only. Tags are matched exactly, not as substrings;
if nothing matches, the error lists the tags the data contains.

| Search engine | Phospho `modification` |
|---|---|
| Sage | `"[+79.9663]"` (decimals follow the search settings) |
| DIA-NN | `"(UniMod:21)"` |
| MaxQuant | `"(Phospho (STY))"` |
| FragPipe | `["S[167]", "T[181]", "Y[243]"]` (the modified residue's total mass) |

### Peptides carrying the modification on several residues

A peptidoform with the target modification on two residues is one measurement of a species that
carries both. Its change between conditions cannot be attributed to either residue — it is the same
problem a shared peptide poses in protein inference, and `to_ptm` gives it the same answer: the
peptidoform is reported as the group it belongs to. With `multisite="combination"` (the default) it
becomes one feature named for its site set, and peptidoforms that differ only in other
modifications or missed cleavages still share a feature:

| Peptidoform | Feature |
|---|---|
| `SPGS[ph]PVLR`, `SPGS[ph]PVLRK` (missed cleavage) | `P1\|S8` |
| `S[ph]PGS[ph]PVLR` | `P1\|S5_S8` |
| `S[ph]PGS[ph]PVLR` matching two proteins | `P1\|S5_S8;P2\|S3_S6` |

Every peptidoform then feeds exactly one feature, so no measurement is tested twice, and no value is
mixed with the change of a neighbouring site. `var["count_site"]` records how many sites a feature
names; the features with `count_site == 1` are the site table built from singly modified
peptidoforms alone, which is what site-centric tools (kinase-activity inference, PhosphoSitePlus
lookups) read:

```python
site = mdata["phospho_site"]
single_site_table = site[:, site.var["count_site"] == 1]
```

A site seen only on multiply modified peptidoforms has no single-site row; it is in the combination
rows. `multisite="pool"` instead copies a multiply modified peptidoform's whole value into each of
its sites, so a site pools singly and multiply modified forms — the interpretation MaxQuant's and
Spectronaut's site tables make, and the previous default. The combination convention is that of
Spectrum Mill's phosphosite tables, TMT-Integrator's multi-site report and MSstatsPTM.

`agg_method` can be selected among the methods described in [Aggregation methods](#aggregation_methods); the matrix rollups `median_polish` and `directlfq` are available here as well (on log2-transformed data).

```python
mdata = mm.utils.attach_fasta("fasta/file/path.fasta")

mdata = mm.pp.to_ptm(
    mdata,
    modi_name="phospho",
    modification="[+79.9663]",
    agg_method="median_polish", # default
    multisite="combination",    # default; "pool" gives a multiply modified peptide to each site
    top_n=None                  # default
    )
```
