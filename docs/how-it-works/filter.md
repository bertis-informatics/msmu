# Filter

Filtering in `msmu` is one call, [`msmu.pp.filter`](../reference/pp/filter.md): it evaluates one
condition on a modality and returns the `MuData` without whatever failed it.

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

In a wide `precursor` modality (see [Data](data.md#wide_precursor_input)) an identification is a
cell, with its score in a layer, so the condition is evaluated per cell:

```python
mdata = mm.pp.filter(
    mdata, modality="precursor", on="layers", column="q_value", keep="lt", value=0.01
)
```

Cells failing the condition are emptied (set to `NaN`) in `.X` and in every layer, and precursors
left without a passing cell are removed. Samples are kept, and only the numeric comparisons apply.
Integer and boolean layers become floating-point to hold `NaN`; a modality with a layer of strings
is refused.

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

Decoys are counted from their own quantification, so a missingness filter removes them when the
engine does not quantify decoys. The columns are a snapshot — recompute after removing samples,
since filtering does not update them.

## Contaminants and decoys

Readers record identification status rather than acting on it:

- **Decoys** stay in `.var`, marked by `decoy == 1`. Filters apply to them as to targets;
  quantification analyses and plots leave them out.
- **Contaminants** stay in `.var` with a `contaminant` flag (`1` / `0`). Whether to remove them is
  left to you: a contaminant accession can also be a genuine protein of the sample (e.g.,
  `Cont_P07339;P07339`).

Filter on identification scores before quantification, since a quantification filter can remove
decoys and make the next q-values too small. `include_decoy=False` keeps every decoy through a
variable or layer filter.

To drop a run from `psm`, remove its identifications (`filename` in `.var`), then its sample
(`.obs`); an `.obs` filter alone keeps the identifications. For an exact result, search again
without the run.

```python
mdata = mm.pp.filter(mdata, modality="psm", on="var", column="filename", keep="ne", value="run_A")
mdata = mm.pp.filter(mdata, modality="psm", on="obs", column="run", keep="ne", value="run_A")
```

Readers write every accession as `[decoy_][Cont_]<accession>`, whatever markers the FASTA used.
To filter on contaminant status, use the flag rather than a marker in the protein string.

## Stored filters (deprecated)

[`add_filter`](../reference/pp/add_filter.md) stores a condition as a boolean mask in
`.varm["filter"]` or `.obsm["filter"]`, and [`apply_filter`](../reference/pp/apply_filter.md)
removes what the stored masks reject. Both still work and emit a `DeprecationWarning`; use
[`filter`](../reference/pp/filter.md) instead. They do not filter cells.

A mask is named `{column}_{keep}_{value}` unless `name` is given; matrix sources add the source
and key, for example `varm['qc'].score_gt_15.0`. Workflows recorded with older matrix filter
names may not replay unchanged: re-record them rather than disabling verification.
