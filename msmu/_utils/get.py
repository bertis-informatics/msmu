from anndata import AnnData
from mudata import MuData
from typing import cast


def get_adata(mdata: MuData, modality: str) -> AnnData:
    """Returns the modality-specific AnnData object with proper typing."""
    return cast(AnnData, mdata[modality])


def get_mdata(mdata) -> MuData:
    """Returns the MuData object with proper typing."""
    return cast(MuData, mdata)
