# Command Provenance (`uns_logger`)

`msmu` functions decorated with `uns_logger` automatically append execution logs to
`mdata.uns["_cmd"]`.

At runtime, `mdata.uns["_cmd"]` is managed as `dict[str, dict]`, where each entry
contains:

- `function`: function name
- `timestamp`: execution timestamp
- `msmu_version`: installed `msmu` version
- `python_version`: runtime Python version
- `payload`: bound function arguments (including default values for omitted args)
- `stdout`: captured printed messages from the function (if any)
  (both `print(...)` and `logging` messages under the `msmu` logger)
- `input_dimensions`: input MuData dimensions (`n_obs`, `n_vars`, per-modality dimensions, `layers`)
- `output_dimensions`: output MuData dimensions (`n_obs`, `n_vars`, per-modality dimensions, `layers`)

## Behavior

- `print(...)` output and `logging` output (from `msmu` loggers) are shown on
  screen and captured into `_cmd` (`stdout`).
- Reader functions (`read_*`, `read_h5mu`) also append `_cmd` entries.
- `mm.read_sdrf(...)` and `mm.pp.add_meta(..., format="sdrf")` share the same
  SDRF parsing and validation flow.
- `mm.pp.add_meta(..., format="sdrf")` records the SDRF source and matching
  arguments in `_cmd` instead of persisting SDRF-specific payloads in `.uns`.
- When reading `.h5mu`, `_cmd` is normalized back to runtime `dict[str, dict]` format.

## Example

```python
mdata = mm.pp.add_filter(
    mdata,
    modality="psm",
    on="var",
    column="q_value",
    keep="lt",
    value=0.01,
)

last = mdata.uns["_cmd"][max(mdata.uns["_cmd"], key=lambda k: int(k))]
print(last["function"])   # add_filter
print(last["payload"])    # arguments (including defaults)
```
