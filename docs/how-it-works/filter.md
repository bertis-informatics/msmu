# Filter

## on `.var`

Functions related to filtering features (`.var`) are implemented in [`msmu.pp.add_filter`](../../reference/pp/add_filter/) and [`msmu.pp.apply_filter`](../../reference/pp/apply_filter/).

In `msmu`, filtering features consists of 2 stages.

1. `add_filter()` to modality

    - The column name should be present in `.var`
    - The `keep` argument accepts general expressions for condition, such as `lt`, `le`, `gt`, `ge`, `equal`, etc.
    - Boolean masking will be stored in `mdata[modality].varm["filter"]` by the column name of `column_keep_value` from parameters, e.g. `q_value_lt_0.01`

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

Filtering on `.obs` is not implemented as an utility function. `.obs` can be filtered with slicing function on `MuData`

```python
# filter BLANK channels for TMT studies
mdata = mdata[mdata.obs["group"] != "BLANK", ]
```
