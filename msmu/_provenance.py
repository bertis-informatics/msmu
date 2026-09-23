"""Inspect and replay execution history, and configure optional content hashing."""

from ._core._provenance import get_log, get_options, options as _options, set_options, to_env, log_provenance as log
from ._core._hashing import _hash_msmu_v1 as compute_hash
from ._core._replay import replay, to_script

__all__ = ["get_log", "get_options", "options", "set_options", "log", "compute_hash", "replay", "to_script", "to_env"]


def options(*, hashing: bool):
    """Temporarily enable/disable hashing, restoring the previous setting on exit."""
    return _options(hashing=hashing)
