"""CPTAC reader -- currently unsupported.

The CPTAC reader is not part of the pipeline and is intentionally disabled: it is
disconnected at the public API (``read_cptac`` is not exported and is absent from
``_reader_registry.py`` / ``__init__.py``) and its implementation has been removed
from this module.

To re-enable CPTAC, add a reader here following the fresh-build contract used by
the other readers (see ``DelpiReader`` / ``DiannReader`` / ``MaxQuantReader``):
build the feature frame on a fresh DataFrame in
``_make_needed_columns_for_identification`` (do not mutate the raw frame), and
implement ``_extract_quant_from_raw`` for the merged (TMT) variant. The base
``SearchResultReader`` no longer copies the raw frame before normalisation, so an
in-place ``_make_needed_columns_for_identification`` would corrupt the raw frame.
The previous implementation lives in git history.
"""
