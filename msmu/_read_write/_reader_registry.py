from pathlib import Path
from typing import Any, Literal
import mudata as md

from ._diann import DiannReader, DiannProteinGroupReader
from ._sage import LfqSageReader, TmtSageReader
from ._maxquant import MaxTmtReader, MaxLfqReader, MaxDiaReader


def read_sage(
    search_dir: str | Path,
    label: Literal["tmt", "label_free"],
    _quantification: bool = True,
) -> md.MuData:
    """
    Reads Sage output and returns a MuData object.
    Args:
        search_dir (str | Path): Path to the Sage output directory.
        label (Literal["tmt", "label_free"]): Label for the Sage output ('tmt' or 'label_free').
        _quantification (bool): Whether to include quantification data. Default is True.
    Returns:
        md.MuData: A MuData object containing the Sage data.
    """
    if label == "tmt":
        reader = TmtSageReader(
            search_dir=search_dir,
        )
    elif label == "label_free":
        reader = LfqSageReader(
            search_dir=search_dir,
            _quantification=_quantification,
        )
    else:
        raise ValueError("Argument label should be one of 'tmt', 'label_free'.")

    mdata:md.MuData = reader.read()

    return mdata


class _ReadDiannFacade:
    """
    Facade class for reading DIA-NN data.
    Provides methods to read data at different levels (precursor and protein group).
    """
    def __call__(self, search_dir: str | Path) -> md.MuData:
        """
        Reads DIA-NN output and returns a MuData object.
        Args:
            search_dir (str | Path): Path to the DIA-NN output directory.
        Returns:
            md.MuData: A MuData object containing the DIA-NN data at precursor level.
        """
        return DiannReader(search_dir=search_dir).read()
    
    def from_pg(self, search_dir: str | Path) -> md.MuData:
        """
        Reads DIA-NN protein group output and returns a MuData object.
        Args:
            search_dir (str | Path): Path to the DIA-NN output directory.
        Returns:
            md.MuData: A MuData object containing the DIA-NN data at protein group level.
        """
        return DiannProteinGroupReader(search_dir=search_dir).read()

read_diann: _ReadDiannFacade = _ReadDiannFacade()


# Working on it
class _MaxQuantFacade:
    """
    Facade class for reading MaxQuant data.
    Provides methods to read data with different labels and acquisition methods.
    """
    def __call__(
        self, 
        search_dir: str | Path, 
        label: Literal["tmt", "label_free"], 
        acquisition: Literal["dda", "dia"], 
        _quantification: bool = True
        ) -> md.MuData:
        """
        Reads MaxQuant output and returns a MuData object.
        Args:
            search_dir (str | Path): Path to the MaxQuant output directory.
            label (Literal["tmt", "label_free"]): Label for the MaxQuant output ('tmt' or 'label_free').
            acquisition (Literal["dda", "dia"]): Acquisition method ('dda' or 'dia').
            _quantification (bool): Whether to include quantification data. Default is True.
        Returns:
            md.MuData: A MuData object containing the MaxQuant data.
        """
        if label == "tmt" and acquisition == "dda":
            reader = MaxTmtReader(
                search_dir=search_dir,
            )
        elif label == "label_free" and acquisition == "dda":
            reader = MaxLfqReader(
                search_dir=search_dir,
                _quantification=_quantification,
            )
        elif label == "label_free" and acquisition == "dia":
            # reader = MaxDiaReader(
            #     search_dir=search_dir,
            # )
            raise NotImplementedError("MaxQuant DIA reader is not implemented yet.")
        else:
            raise ValueError("Argument label should be one of 'tmt', 'label_free' and acquisition should be one of 'dda', 'dia'.")
        return reader.read()

    def from_pg(self, *args: Any, **kwds: Any) -> md.MuData:
        """
        Reads MaxQuant protein group output and returns a MuData object.
        """
        raise NotImplementedError("MaxQuant protein group reader is not implemented yet.")

read_maxquant: _MaxQuantFacade = _MaxQuantFacade()


def read_h5mu(h5mu_file: str | Path) -> md.MuData:
    """
    Reads an h5mu file (HDF5) and returns a MuData object.

    Args:
        h5mu_file (str | Path): Path to the H5MU file.

    Returns:
        md.MuData: A MuData object.
    """
    return md.read_h5mu(h5mu_file)



#######################################################################
# Placeholder functions for future implementations
########################################################################
def read_fragpipe():
    raise NotImplementedError


def read_comet():
    raise NotImplementedError


def read_protdiscov():
    raise NotImplementedError
