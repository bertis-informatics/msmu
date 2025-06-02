import mudata

from . import _plotting as pl
from . import _preprocessing as pp
from . import _read_write as rw
from . import _tools as tl
from ._read_write._readers import (merge_mudata, read_diann, read_h5mu,
                                   read_sage)
from ._utils import get_fasta_meta, get_label, map_fasta, subset

__version__ = "0.1.0_dev"

mudata.set_options(pull_on_update=False)
pl._set_templates()

__all__ = [
    "rw",
    "pp",
    "pl",
    "tl",
]
