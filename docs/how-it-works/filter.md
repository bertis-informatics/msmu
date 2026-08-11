# Filter

Filtering in `msmu` is split into two steps and is implemented in
[`msmu.pp.add_filter`](../../reference/pp/add_filter/) and
[`msmu.pp.apply_filter`](../../reference/pp/apply_filter/).

1. `add_filter()` creates a boolean mask and stores it as a named filter column.
2. `apply_filter()` applies one or more stored masks to subset the modality.

## `add_filter`

`add_filter` supports multiple sources via `on`:

- `on="var"`: read `column` from `.var`, store mask in `.varm["filter"]`
- `on="obs"`: read `column` from `.obs`, store mask in `.obsm["filter"]`
- `on="varm"`: read `column` from `.varm[key]`, store mask in `.varm["filter"]`
- `on="obsm"`: read `column` from `.obsm[key]`, store mask in `.obsm["filter"]`

`key` is required when `on` is `"varm"` or `"obsm"`.

The `keep` argument accepts conditional operators such as `eq`, `ne`, `lt`, `le`,
`gt`, `ge`, `contains`, and `not_contains`.

Stored filter column names follow this pattern:
`{column}_{keep}_{value}`.

```python
# feature-level filter from .var
mdata = mm.pp.add_filter(
    mdata,
    modality="psm",
    on="var",
    column="q_value",
    keep="lt",
    value=0.01,
)

# sample-level filter from .obs
mdata = mm.pp.add_filter(
    mdata,
    modality="psm",
    on="obs",
    column="condition",
    keep="not_contains",
    value="BLANK",
)
```

## `apply_filter`

`apply_filter` controls target axis with `on`:

- `on="all"` (default): apply both `.varm["filter"]` and `.obsm["filter"]`
- `on="var"`: apply only `.varm["filter"]`
- `on="obs"`: apply only `.obsm["filter"]`
- `columns=[...]` (optional): apply only selected filter columns by name

When `on="all"` and one side does not have a stored filter table, a warning is
printed and that axis is skipped. When `on="var"` or `on="obs"` and the requested
filter table is missing, an error is raised.

The function also prints which filter columns are applied, and this printed output
is captured into `mdata.uns["_cmd"]` by the command logger.

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

## Contaminants and decoys

Readers record identification status rather than acting on it:

- **Decoys** are separated out of `.var` at read time and kept in `.uns["decoy"]`, where the
  q-value machinery needs them. Filters you add are applied to them in parallel, through
  `.uns["decoy_filter"]`, so target and decoy features stay matched.
- **Contaminants** stay in `.var` with a `contaminant` flag (`1` / `0`). msmu records the flag and
  leaves the decision to you: a contaminant accession can also be a genuine protein of the sample
  (the parsed group keeps both, e.g. `Cont_P07339;P07339`), so whether such a feature should go is
  a protein-level judgement rather than something to drop at the PSM level.

Contaminant markers differ per FASTA (`contam_`, `Cont_`, `CON__`, and each engine's decoy tag in
front of them), so the reader normalises every accession to one canonical form
`[rev_][Cont_]<accession>` and sets the flag from the parse. If you do filter on contaminant
status, filter on the flag rather than matching a marker on the protein string — a string match
silently stops matching when the search uses a different contaminant FASTA.
