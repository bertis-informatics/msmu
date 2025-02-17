import mudata

from . import _read_write as rw
from . import _preprocessing as pp
from . import _plotting as pl
from . import _tools as tl

from ._read_write._readers import mask_obs, merge_mudata, read_sage, read_h5mu

__version__ = "0.1.0_dev"

mudata.set_options(pull_on_update=False)
pl._set_templates()

__all__ = [
    "rw",
    "pp",
    "pl",
    "tl",
]
