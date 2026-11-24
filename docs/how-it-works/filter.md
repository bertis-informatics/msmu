# Filter

[`msmu.pp.filter`](../reference/pp/filter.md) filters in one call: it evaluates one condition on a
modality and returns the `MuData` without whatever failed it.

```python
mdata = mm.pp.filter(mdata, modality="psm", on="var", column="q_value", keep="lt", value=0.01)
```

No mask is stored. The call itself, with its condition, is recorded in the
[provenance](provenance.md) log, which is the record of what was filtered. To look at what a
condition would remove before removing it, keep the unfiltered object and compare:

```python
filtered = mm.pp.filter(mdata, modality="psm", on="var", column="q_value", keep="lt", value=0.01)
mdata["psm"].n_vars, filtered["psm"].n_vars
```

## Conditions

`column` names the values to test and `on` says where they are:

| `on` | Values read from | Removed when the condition fails |
| --- | --- | --- |
| `"var"` | `.var[column]` | features |
| `"obs"` | `.obs[column]` | samples |
| `"varm"` | `.varm[key][column]` | features |
| `"obsm"` | `.obsm[key][column]` | samples |
| `"layers"` | `.layers[column]`, one value per cell | cells, then the features left without a cell |

`on` has no default: where the values are depends on the layout of the modality, so every call
names it. `key` is required when `on` is `"varm"` or `"obsm"`.

The `keep` argument accepts conditional operators such as `eq`, `ne`, `lt`, `le`,
`gt`, `ge`, `contains`, and `not_contains`. Missing source values never satisfy a
condition, including `ne` and `not_contains`. `contains` and `not_contains` retain
`pandas` regular-expression semantics: `A.B` also matches `AxB`. Escape regex
metacharacters for literal matching.

```python
# feature-level filter from .var
mdata = mm.pp.filter(mdata, modality="psm", on="var", column="q_value", keep="lt", value=0.01)

# sample-level filter from .obs
mdata = mm.pp.filter(
    mdata, modality="psm", on="obs", column="condition", keep="not_contains", value="BLANK"
)

# feature-level filter from a table in .varm
mdata = mm.pp.filter(
    mdata, modality="psm", on="varm", key="qc", column="score", keep="gt", value=0.01
)
```

Each call applies one condition; call it again for the next one.

## Filtering cells

In a wide `precursor` modality (samples are runs, features are precursors; see
[Data](data.md#wide_precursor_input)) an identification is a cell, not a feature, and its score is
kept per cell in a layer. A filter on a feature would keep or remove a precursor in every run at
once, so the condition is evaluated per cell instead:

```python
mdata = mm.pp.filter(
    mdata, modality="precursor", on="layers", column="q_value", keep="lt", value=0.01
)
```

Cells failing the condition are emptied (set to `NaN`) in `.X` and in every layer, and precursors
left without a passing cell are removed. Samples are kept, and only the numeric comparisons apply.

Every layer is emptied because a failing cell is an identification that is gone, as a removed
feature is in the long layout. An integer or boolean layer cannot hold `NaN` and becomes a
floating-point layer. A layer of strings cannot hold an emptied cell in a form an h5mu file
stores, so the filter refuses a modality that has one. A wide matrix has cells without an
identification from the start, and only `NaN` marks them: keep per-cell values in
floating-point layers.

## Missingness

A quantification filter needs the missing counts on `.var` first.
[`pp.calculate_missingness`](../reference/pp/calculate_missingness.md) records `n_missing` and
`missing_ratio` per feature and, given an `.obs` column, `missing_ratio_<level>` per group, so the
usual filter call applies:

```python
mdata = mm.pp.calculate_missingness(mdata, modality="protein", group_obs="condition")
mdata = mm.pp.filter(
    mdata, modality="protein", on="var", column="missing_ratio", keep="lt", value=0.5
)
```

Decoys carry no quantification, so they receive `n_missing = 0` and `missing_ratio = 0.0`: a
"missing at most" filter keeps every decoy and the target/decoy counts behind the q-values stay
intact, while a filter in the other direction (`ge`/`gt` on missingness) would drop them. The
columns are a snapshot — recompute after removing samples, since filtering does not update them.

## Contaminants and decoys

Readers record identification status rather than acting on it:

- **Decoys** are separated out of `.var` at read time and kept in `.uns["decoy"]`, where the
  q-value machinery needs them. A feature or cell filter is applied to them as well, on their
  own column of the same name, so target and decoy features stay matched.
- **Contaminants** stay in `.var` with a `contaminant` flag (`1` / `0`). `msmu` records the flag and
  leaves the decision to you: a contaminant accession can also be a genuine protein of the sample
  (the parsed group keeps both, e.g., `Cont_P07339;P07339`), so whether such a feature should go is
  a protein-level judgment rather than something to drop at the PSM level.

Because decoys are filtered on their own values, the column a `.var` filter uses, or the column
named like the layer a cell filter uses, must exist in `.uns["decoy"]` too; the filter refuses
otherwise. Give decoys the column with a recorded edit rather than by hand —
[`dt.assign`](../reference/dt/assign.md) with `on="decoy"` for a constant, or
[`dt.map`](../reference/dt/map.md) with `target="<modality>.decoy"` to carry a sample
annotation over by filename — so the step stays in provenance. A `.varm` table has no decoy
rows, so with `on="varm"` a decoy is kept only when it shares its name with a passing feature.

Contaminant markers differ per FASTA (`contam_`, `Cont_`, `CON__`, and each engine's decoy tag in
front of them), so the reader normalizes every accession to one canonical form
`[decoy_][Cont_]<accession>`, takes `decoy_` from the decoy column and sets the contaminant flag
from the parse. If you do filter on contaminant
status, filter on the flag rather than matching a marker on the protein string — a string match
silently stops matching when the search uses a different contaminant FASTA.

## Stored filters

[`msmu.pp.add_filter`](../reference/pp/add_filter.md) and
[`msmu.pp.apply_filter`](../reference/pp/apply_filter.md) split filtering into two steps: the first
stores a boolean mask as a named filter column, the second applies one or more stored masks.
They do not filter cells.

### Define filters

[`add_filter()`](../reference/pp/add_filter.md) supports multiple sources via `on`:

- `on="var"`: read `column` from `.var`, store mask in `.varm["filter"]`
- `on="obs"`: read `column` from `.obs`, store mask in `.obsm["filter"]`
- `on="varm"`: read `column` from `.varm[key]`, store mask in `.varm["filter"]`
- `on="obsm"`: read `column` from `.obsm[key]`, store mask in `.obsm["filter"]`

Use `name` to give a filter a short, explicit identifier:

```python
mdata = mm.pp.add_filter(
    mdata, modality="psm", on="varm", key="qc",
    column="score", keep="gt", value=0.01, name="qc_score",
)
mdata = mm.pp.apply_filter(mdata, modality="psm", columns=["qc_score"])
```

Names must be nonblank strings without `/` so their definitions can be saved in
HDF5. Explicit names are shared across the modality's observation and feature filters.
The source, column, operator, and value are recorded in
`adata.uns["filter_conditions"][name]`. Reusing a name with the same condition
recomputes its mask; a different condition raises an error. An existing mask
without a recorded definition cannot be overwritten with an explicit name.

When `name` is omitted, stored filter column names follow this pattern:
`{column}_{keep}_{value}` for `obs` and `var`. For matrix sources, the name
also includes the source and key, for example `varm['qc'].score_gt_15.0`.
Use that full name when selecting matrix filters with `columns`.

Missing masks introduced by concatenating datasets remain unapplied to rows where
that filter was not recorded.

### Apply filters

[`apply_filter()`](../reference/pp/apply_filter.md) controls target axis with `on`:

- `on="all"` (default): apply both `.varm["filter"]` and `.obsm["filter"]`
- `on="var"`: apply only `.varm["filter"]`
- `on="obs"`: apply only `.obsm["filter"]`
- `columns=[...]` (optional): apply only selected filter columns by name

When `on="all"` and one side does not have a stored filter table, that axis is
skipped. A warning is printed if neither side has filters. When `on="var"` or `on="obs"` and the requested
filter table is missing, an error is raised. Invalid `on` values also raise an error.

The function also prints which filter columns are applied. The provenance logger records
the successful call in `mdata.uns["_log"]`, but does not capture printed output.

```python
mdata = mm.pp.apply_filter(mdata, modality="psm", on="all")

# apply only selected filters
mdata = mm.pp.apply_filter(
    mdata,
    modality="psm",
    on="var",
    columns=["q_value_lt_0.01"],
)
```

### Validation and compatibility

Requested filter names must all exist on the selected axis (either axis for
`on="all"`). Missing names raise an error; filters are not partially applied.
Variable filters are applied to decoys through a parallel mask in
`.uns["decoy_filter"]`, so they must also exist there when decoys are present.
Applying a subset of filters preserves the other decoy conditions for later calls.

Existing stored filter names are not renamed. Newly recorded matrix filters use
the source-qualified names above; repeating the same condition replaces its mask.
Old workflows that regenerate matrix filters using the former names, relied on
missing values passing, or requested nonexistent columns may no longer replay
unchanged. Verified replay can report a hash mismatch, and explicit old matrix
filter names can raise an error. Re-record those workflows with the current API;
do not disable verification to hide a changed result.
