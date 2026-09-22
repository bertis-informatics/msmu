import logging

import anndata

from .logging_utils import ensure_null_handler, setup_logger

from . import _plotting as pl
from . import _preprocessing as pp
from . import _tools as tl
from . import _read_write as io
from ._deprecated import merge_mudata
from ._read_write._reader_registry import (
    read_h5mu,
    read_sage,
    read_diann,
    read_maxquant,
    read_fragpipe,
    read_sdrf,
    read_delpi,
    # read_cptac,
)
from . import _utils as utils
from . import _provenance as pv
from . import _data as dt

try:
    from ._version import version as __version__
except ImportError:
    __version__ = version = "0.0.0"
else:
    version = __version__

logger = ensure_null_handler()
if logger.level == logging.NOTSET and not any(
    not isinstance(handler, logging.NullHandler) for handler in logger.handlers
):
    setup_logger()

# msmu serialises proteomics MuData to .h5mu. The reader frames (polars -> pandas on the pandas-3
# stack) carry pandas nullable / Arrow-backed string columns -- including the obs/var index -- which
# anndata 0.13 refuses to write when pd.options.future.infer_string is False unless this is opted
# in. Enable it so mdata.write_h5mu(...) succeeds regardless of the ambient infer_string setting;
# the nullable-string-array on-disk format round-trips losslessly (values + null mask). See BID-103.
if hasattr(anndata.settings, "allow_write_nullable_strings"):
    anndata.settings.allow_write_nullable_strings = True

del anndata

__all__ = [
    "read_h5mu",
    "read_sage",
    "read_diann",
    "read_maxquant",
    "read_fragpipe",
    "read_sdrf",
    "read_delpi",
    # "read_cptac",
    "merge_mudata",
    "pp",
    "dt",
    "pl",
    "tl",
    "utils",
    "io",
    "pv",
    "setup_logger",
]
