import mudata

import logging
from .logging_utils import setup_logger, LogLevel

from . import _plotting as pl
from . import _preprocessing as pp
from . import _tools as tl
from . import _read_write as io
from ._read_write._reader_utils import merge_mudata
from ._read_write._reader_registry import read_h5mu, read_sage, read_diann, read_maxquant, read_fragpipe
from . import _utils as utils

try:
    from ._version import version as __version__
except ImportError:
    __version__ = version = "0.0.0"
else:
    version = __version__

logger = logging.getLogger("msmu")
logger.setLevel(LogLevel.INFO)
setup_logger(level=LogLevel.INFO)

mudata.set_options(pull_on_update=False)
pl.set_templates()

del LogLevel, logging, mudata

__all__ = [
    "read_h5mu",
    "read_sage",
    "read_diann",
    "read_maxquant",
    "read_fragpipe",
    "merge_mudata",
    "pp",
    "pl",
    "tl",
    "utils",
    "io",
]
