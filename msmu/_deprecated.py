"""Compatibility wrappers for deprecated public functions."""

import warnings

import mudata as md


def merge_mudata(mdatas: dict[str, md.MuData]) -> md.MuData:
    """Deprecated alias for :func:`msmu.concat`."""
    from ._read_write._reader_utils import concat

    warnings.warn(
        "merge_mudata is deprecated; use msmu.concat instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return concat(mdatas)
