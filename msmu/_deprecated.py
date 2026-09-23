"""Compatibility wrappers for deprecated public functions."""

import warnings

import mudata as md


def merge_mudata(mdatas: dict[str, md.MuData]) -> md.MuData:
    """Deprecated alias of [`msmu.dt.concat`][msmu.dt.concat].

    Parameters:
        mdatas: Dictionary mapping dataset labels to at least two sample-aligned MuData objects.

    Returns:
        A concatenated MuData, as described by [`concat`][msmu.dt.concat].

    Notes:
        Calling this alias emits `DeprecationWarning`. Replace `mm.merge_mudata(...)` with [`mm.dt.concat(...)`][msmu.dt.concat].

    Examples:
        ```python
        import msmu as mm
        combined = mm.dt.concat({"batch_a": first, "batch_b": second})
        ```
    """
    from ._data import concat

    warnings.warn(
        "merge_mudata is deprecated; use msmu.dt.concat instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return concat(mdatas)
