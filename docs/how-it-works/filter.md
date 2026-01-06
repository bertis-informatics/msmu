# Filter

## on `.var`

Functions related to filtering features (`.var`) are implemented in [`msmu.pp.add_filter`](/reference/pp/add_filter/) and [`msmu.pp.apply_filter`](/reference/pp/apply_filter/).

In `msmu`, filtering features consists of 2 stages.

1. `add_filter()` to modality

    - making boolean mask for features in `mdata[modality].varm["filter"]`
    - a column to filter should be in `.var`
    - The `keep` argument accepts general expressions for test, such as `lt`, `le`, `gt`, `ge`, `equal`, etc.

2. `apply_filter()`

    - filter features based on boolean masks from `mdata[modality].varm["filter"]`

```python
# filter PSM with q_value < 0.01
mdata = mm.pp.add_filter(
    mdata,
    modality="psm",
    column="q_value", # a column in .var
    keep="lt",
    value=0.01
    )

mdata = mm.pp.apply_filter(mdata, modality="psm")
```

## on `.obs`

Filtering on `.obs` is not implemented as an utility function. `.obs` can be filtered with slicing function on `mudata`

```python
# filter BLANK channels for TMT studies
mdata = mdata[mdata.obs["group"] != "BLANK", ]
```
