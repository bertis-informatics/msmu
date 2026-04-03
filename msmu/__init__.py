import mudata

from .logging_utils import ensure_null_handler, setup_logger

from . import _plotting as pl
from . import _preprocessing as pp
from . import _tools as tl
from . import _read_write as io
from ._read_write._reader_utils import merge_mudata
from ._read_write._reader_registry import (
    read_h5mu,
    read_sage,
    read_diann,
    read_maxquant,
    read_fragpipe,
    read_delpi,
    # read_cptac,
)
from . import _utils as utils

try:
    from ._version import version as __version__
except ImportError:
    __version__ = version = "0.0.0"
else:
    version = __version__

logger = ensure_null_handler()

mudata.set_options(pull_on_update=False)
pl.set_templates()

del mudata

__all__ = [
    "read_h5mu",
    "read_sage",
    "read_diann",
    "read_maxquant",
    "read_fragpipe",
    "read_delpi",
    # "read_cptac",
    "merge_mudata",
    "pp",
    "pl",
    "tl",
    "utils",
    "io",
    "setup_logger",
]
