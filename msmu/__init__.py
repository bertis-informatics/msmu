import mudata

import logging
from .logging_utils import setup_logger, LogLevel

from . import _plotting as pl
from . import _preprocessing as pp
from . import _tools as tl
from ._read_write._reader_utils import merge_mudata
from ._read_write._reader_registry import read_h5mu, read_sage, read_diann, read_maxquant
from . import _utils as utils
from ._utils import get_fasta_meta, get_label, map_fasta, subset, add_quant, split_tmt, rename_obs

__version__ = "0.1.8"

logger = logging.getLogger("msmu")
logger.setLevel(LogLevel.INFO)
setup_logger(level=LogLevel.INFO)

mudata.set_options(pull_on_update=False)
pl._set_templates()

del LogLevel, logging, mudata

__all__ = [
    "pp",
    "pl",
    "tl",
    "utils"
]
