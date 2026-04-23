from typing import Any, cast

from anndata import AnnData
from mudata import MuData


def get_adata(mdata: MuData, modality: str) -> AnnData:
    """Return the modality-specific AnnData object with proper typing."""
    return mdata.mod[modality]


def get_mdata(mdata: Any) -> MuData:
    """Return the MuData object with proper typing."""
    return cast(MuData, mdata)
