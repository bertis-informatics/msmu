# Sample Metadata (SDRF)

## Overview

[SDRF-Proteomics](https://github.com/bigbio/proteomics-sample-metadata) is the community format for
describing what each run and each label in an experiment actually is: the sample, its conditions,
the instrument, the labelling. `msmu` treats an SDRF as the **source of truth** for sample
metadata, in two explicit steps:

1. `attach_sdrf()` stores the whole table at `mdata.uns["sdrf"]`, unchanged.
2. `apply_sdrf_to_obs()` projects the columns you want onto each modality's `.obs`.

The split exists because an SDRF row does not correspond to one observation. Its rows span **both**
axes — `comment[label]` addresses the obs axis (channels) and `comment[data file]` the var axis
(runs and fractions) — so a 144-row SDRF of 12 fractions × 12 channels cannot be reduced to 12
sample rows without discarding the fraction axis. Keeping the original in `.uns`, which is carried
through `split_tmt()` and `collapse_obs()`, means the full table stays available no matter how the
container is reshaped, while `.obs` holds only what is well-defined per observation.

Metadata that is not in SDRF form — a plain DataFrame, csv, tsv, or parquet — is still supported
through [`add_meta()`](#add_meta-metadata-without-an-sdrf).

## `attach_sdrf()`

```python
mdata = mm.read_sage(
    identification_file="data/sage_tmt/sage/results.sage.tsv",
    quantification_file="data/sage_tmt/sage/tmt.tsv",
    label="tmt",
)
mdata = mm.pp.attach_sdrf(
    mdata,
    "data/sage_tmt/meta.sdrf.tsv",  # path, URL, or DataFrame
    validate=True,                  # optional, default True: validate with sdrf-pipelines
    skip_ontology=True,             # optional, default True: skip ontology term lookups
)

mdata.uns["sdrf"].shape   # (144, 17)
```

The SDRF may be a path, a URL, or a DataFrame. Paths and URLs are read with `mm.read_sdrf()`, and
the table is validated with [`sdrf-pipelines`](https://github.com/bigbio/sdrf-pipelines) unless you
pass `validate=False`; ontology term lookups are skipped by default (`skip_ontology=True`). To read
an SDRF into a DataFrame without attaching it, call `mm.read_sdrf(sdrf_file, validate_sdrf=True)`
directly.

`.obs` is untouched by this step, and `uns["sdrf"]` is never rewritten by the steps that follow.

## `apply_sdrf_to_obs()`

Projection reduces the SDRF to the obs axis by a **match key** and copies across the columns that
are a function of that key:

```python
mdata = mm.pp.apply_sdrf_to_obs(mdata)
```
```text
WARNING - apply_sdrf_to_obs: psm columns not projectable under 'comment[label]';
          kept only in uns['sdrf']: ['assay name', 'comment[data file]', 'comment[fraction identifier]']
```
```python
mdata["psm"].obs.head(2)
```
```text
        source name  characteristics[organism]  ...  factor value[condition]  factor value[time point]
TMT126          t0h               Homo sapiens  ...                       t0                       t0h
TMT127          t1h               Homo sapiens  ...                       t1                       t1h
```

All three arguments are optional, and the call above passes none of them:

- **`on`** (optional, default `None`) — the match key. `None` resolves it per modality, from that
  modality's own `uns["label"]`: `comment[label]` where the label is `tmt` (obs are channels), and
  `comment[data file]` otherwise (obs are runs). After `split_tmt()` the composite
  `[comment[label], set_key]` is used for the whole container instead. File names are matched
  ignoring the extension, since readers store bare stems (`QExHF03751`) while an SDRF carries
  `QExHF03751.mzML`. Pass a column name to override, or a list of columns to match a composite key.

    Only the reader's `psm` modality carries `uns["label"]`, so after summarising a TMT experiment
    the derived `peptide` / `protein` modalities fall back to `comment[data file]` even though their
    obs are still channels. Project **before** summarising, or pass `on="comment[label]"`
    explicitly.
- **`columns`** (optional, default `None`) — which columns to project. `None` projects every
  projectable column; pass a name or list to project a subset.
- **`set_index`** (optional, default `None`) — `None` keeps the existing `obs.index`; pass a
  column to rename samples by it (see below).

**Projectable columns only.** A column that takes more than one value within a key group cannot be
represented per observation. Above, each channel spans 12 fractions, so `comment[data file]` and
`comment[fraction identifier]` are skipped with a warning and stay in `uns["sdrf"]`. Naming such a
column explicitly in `columns=[...]` raises instead of skipping, so `.obs` never silently collapses
SDRF data.

### Naming samples with `set_index`

Straight from a reader, samples are named by whatever the instrument produced — TMT channels or
run file names. `set_index` renames them to the SDRF's own sample identifier, so every plot,
`obs` slice and DE result speaks in the experiment's terms rather than the acquisition's:

```python
mdata = mm.pp.apply_sdrf_to_obs(
    mdata,
    on=None,                                             # optional, default None: match key auto-picked per modality
                                                         #   (comment[label] for TMT, comment[data file] otherwise)
    columns=["source name", "factor value[condition]"],  # optional, default None: project every projectable column
    set_index="source name",                             # optional, default None: keep obs.index unchanged
)

mdata["psm"].obs
```
```text
      source name factor value[condition]
t0h           t0h                      t0
t1h           t1h                      t1
t2h           t2h                      t2
t6h           t6h                      t6
```

The projected columns land on every modality's `obs` and on the MuData-level `mdata.obs`, the
index column stays available as a column too, and `uns["sdrf"]` keeps the untouched original —
including the columns that were not projectable.

Two constraints on `set_index`, both raised rather than worked around:

- The `set_index` column must be **among the projected columns** — that is, listed in `columns`, or
  covered by the default `columns=None`, or already present in `obs` from an earlier call.
  `columns=["factor value[condition]"], set_index="source name"` raises, because `source name` was
  never projected.
- Its values must be **unique across obs**, since they become the index.

```python
mm.pp.apply_sdrf_to_obs(mdata, set_index="characteristics[organism]")
# ValueError: set_index 'characteristics[organism]' is not unique across psm.obs; cannot index.

mm.pp.apply_sdrf_to_obs(mdata, set_index="comment[data file]")
# ValueError: set_index 'comment[data file]' is neither a projectable SDRF column nor an obs column
```

On a multi-plex experiment, project and rename **after** `split_tmt()`, not before. The split
builds its `channel_set` obs names from the current obs names and starts the new modality with an
empty `obs`, so renaming first both discards the projected columns and produces names
(`t0h_set1`) that no longer match the SDRF's `comment[label]`. Keep the channel labels through
the split:

```python
mdata = mm.pp.attach_sdrf(mdata, "meta.sdrf.tsv")
mdata = mm.pp.split_tmt(mdata)                                  # obs: TMT126_set1, ...
mdata = mm.pp.apply_sdrf_to_obs(mdata, set_index="source name")  # obs: the SDRF's sample names
```

The projected columns are also merged onto the MuData-level `mdata.obs`, where functions such as
`correct_batch_effect()` read them.

## TMT: channels, plexes and `split_tmt()`

Readers emit TMT channels in the SDRF spelling — `TMT126`, `TMT127N`, … — so `obs.index` lines up
with `comment[label]` without a translation table.

For a multi-plex experiment, `split_tmt()` splits each plex into its own set of samples. With an
SDRF attached it derives the file → set map itself:

```python
mdata = mm.pp.split_tmt(
    mdata,
    map=None,                                     # optional, default None: derive the file -> set map from uns["sdrf"]
    set_key="comment[sample preparation batch]",  # optional, default shown: SDRF column naming each file's set
)
mdata = mm.pp.apply_sdrf_to_obs(mdata)               # obs are now "TMT126_set1", ...
```

Which columns it reads, by default:

| side | column | note |
| --- | --- | --- |
| SDRF | `comment[data file]` | fixed; matched to the data file each plex was acquired in |
| SDRF | `comment[sample preparation batch]` | the set/plex name — override with `set_key` |
| MuData | `psm.var["filename"]` | what the SDRF file names are matched against, extension stripped |

SDRF has no dedicated column for the TMT set, which is why the batch column stands in for it.
`split_tmt()` errors if the column is absent, or if one data file maps to two sets. Any column
that is constant per data file may be named instead:

```python
mdata = mm.pp.split_tmt(mdata, set_key="factor value[plex]")
```

`split_tmt()` records the key it used, so the following `apply_sdrf_to_obs()` knows to match on the
composite `[comment[label], set_key]` that the post-split `channel_set` obs index encodes — no
manual `on` is needed.

Passing `map` explicitly skips the SDRF entirely — a `dict`, `Series`, or two-column `DataFrame`
of file name → set:

```python
mdata = mm.pp.split_tmt(mdata, map={"run_A": "set1", "run_B": "set2"})  # SDRF not consulted
```

## `add_meta()` — metadata without an SDRF

Not every experiment has an SDRF. `add_meta()` is the generic path: it takes a **DataFrame you
already have in memory**, or a `csv`, `tsv`, `parquet`, or `sdrf` file (path or URL), and joins it
onto `obs` by one metadata key and one obs key.

```python
import pandas as pd

meta = pd.DataFrame(
    {
        "channel": ["TMT126", "TMT127", "TMT128", "TMT129", "TMT130", "TMT131"],
        "condition": ["ctrl", "ctrl", "ctrl", "treat", "treat", "treat"],
        "replicate": [1, 2, 3, 1, 2, 3],
    }
)

mdata = mm.pp.add_meta(
    mdata,
    meta,                   # DataFrame, path, or URL
    format=None,            # optional, default None: for files, inferred from the extension
    metadata_on="channel",  # optional, default None: metadata column to match on (default: its index)
    obs_columns=None,       # optional, default None: obs column to match on (default: the obs index)
    validate_sdrf=True,     # optional, default True: only applies to SDRF input
    skip_ontology=True,     # optional, default True: skip ontology term lookups
)

mdata["psm"].obs
```
```text
       channel condition  replicate
TMT126  TMT126      ctrl          1
TMT127  TMT127      ctrl          2
TMT128  TMT128      ctrl          3
TMT129  TMT129     treat          1
TMT130  TMT130     treat          2
TMT131  TMT131     treat          3
```

Every column of the table lands on `obs` (and on the MuData-level `mdata.obs`). Matching is exact
and there is no projectability rule — it is a plain join, so the table must already be one row per
observation. Observations with no matching row are left as `NaN` rather than raising, so check
`obs` after joining a table you are unsure of. For a file input, `format`
is inferred from the extension (`.sdrf.tsv` / `.sdrf` → SDRF, otherwise `.csv` / `.tsv` /
`.parquet`) and only needs passing when the name does not say.

Use `attach_sdrf()` + `apply_sdrf_to_obs()` instead when you do have an SDRF: those keep the
original table on the container, understand its two axes, and refuse to collapse a column that is
not well-defined per observation.
