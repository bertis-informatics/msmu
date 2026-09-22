"""Compatibility wrappers for deprecated public functions."""

import warnings

import mudata as md


def merge_mudata(mdatas: dict[str, md.MuData]) -> md.MuData:
    """Deprecated alias for :func:`msmu.dt.concat`."""
    from ._data import concat

    warnings.warn(
        "merge_mudata is deprecated; use msmu.dt.concat instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return concat(mdatas)
