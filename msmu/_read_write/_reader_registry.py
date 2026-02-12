from pathlib import Path
from typing import Any, Literal
import mudata as md

from ._diann import DiannReader, DiannProteinGroupReader
from ._sage import LfqSageReader, TmtSageReader
from ._maxquant import MaxTmtReader, MaxLfqReader, MaxDiaReader
from ._fragpipe import TmtFragPipeReader, LfqFragPipeReader
from .._utils._provenance import append_cmd_log, capture_provenance_output, get_bound_call_kwargs, normalize_cmd_for_runtime


def read_sage(
    identification_file: str | Path,
    label: Literal["tmt", "label_free"],
    quantification_file: str | Path | None = None,
) -> md.MuData:
    """
    Reads Sage output and returns a MuData object.

    Parameters:
        identificaton_file: Path to the results.sage.tsv.
        label: Label for the Sage output ('tmt' or 'label_free').
        quantification_file: Whether to include quantification data. Default is None.

    Returns:
        A MuData object containing the Sage data.
    """
    if label == "tmt":
        reader = TmtSageReader(
            identification_file=identification_file,
            quantification_file=quantification_file,
        )
    elif label == "label_free":
        reader = LfqSageReader(
            identification_file=identification_file,
            quantification_file=quantification_file,
        )
    else:
        raise ValueError("Argument label should be one of 'tmt', 'label_free'.")

    with capture_provenance_output() as stdout_buffer:
        mdata: md.MuData = reader.read()
    return append_cmd_log(
        mdata,
        function="read_sage",
        payload=get_bound_call_kwargs(
            read_sage,
            identification_file,
            label,
            quantification_file=quantification_file,
        ),
        stdout=stdout_buffer.getvalue().strip() or None,
    )


class _ReadDiannFacade:
    """
    Facade class for reading DIA-NN data.
    Provides methods to read data at different levels (precursor and protein group).
    """

    __name__ = "read_diann"

    def __call__(self, identification_file: str | Path) -> md.MuData:
        """
        Reads DIA-NN output and returns a MuData object.

        Parameters:
            identification_file: Path to the DIA-NN output directory.

        Returns:
            A MuData object containing the DIA-NN data at precursor level.
        """
        with capture_provenance_output() as stdout_buffer:
            mdata = DiannReader(identification_file=identification_file).read()
        return append_cmd_log(
            mdata,
            function="read_diann",
            payload=get_bound_call_kwargs(self.__call__, identification_file),
            stdout=stdout_buffer.getvalue().strip() or None,
        )

    def from_pg(self, identification_file: str | Path) -> md.MuData:
        """
        Reads DIA-NN protein group output and returns a MuData object.

        Parameters:
            identification_file: Path to the DIA-NN output directory.

        Returns:
            A MuData object containing the DIA-NN data at protein group level.
        """
        with capture_provenance_output() as stdout_buffer:
            mdata = DiannProteinGroupReader(identification_file=identification_file).read()
        return append_cmd_log(
            mdata,
            function="read_diann.from_pg",
            payload=get_bound_call_kwargs(self.from_pg, identification_file),
            stdout=stdout_buffer.getvalue().strip() or None,
        )


read_diann: _ReadDiannFacade = _ReadDiannFacade()
"""Alias for :class:`_ReadDiannFacade`.

Parameters:
    identification_file: Path to the DIA-NN output directory.

Returns:
    A MuData object containing the DIA-NN data at precursor level

Usage:
    mdata_precursor = mm.read_diann(search_dir)
    mdata_protein_group = mm.read_diann.from_pg(search_dir)
"""


# Working on it
class _MaxQuantFacade:
    """
    Facade class for reading MaxQuant data.
    Provides methods to read data with different labels and acquisition methods.
    """

    __name__ = "read_maxquant"

    def __call__(
        self,
        identification_file: str | Path,
        label: Literal["tmt", "label_free"],
        acquisition: Literal["dda", "dia"],
        _quantification: bool = True,
    ) -> md.MuData:
        """
        Reads MaxQuant output and returns a MuData object.

        Parameters:
            identification_file: Path to the MaxQuant output directory.
            label: Label for the MaxQuant output ('tmt' or 'label_free').
            acquisition: Acquisition method ('dda' or 'dia').
            _quantification: Whether to include quantification data. Default is True.

        Returns:
            A MuData object containing the MaxQuant data.
        """
        if label == "tmt" and acquisition == "dda":
            reader = MaxTmtReader(
                identification_file=identification_file,
            )
        elif label == "label_free" and acquisition == "dda":
            reader = MaxLfqReader(
                identification_file=identification_file,
                _quantification=_quantification,
            )
        elif label == "label_free" and acquisition == "dia":
            # reader = MaxDiaReader(
            #     identification_file=identification_file,
            # )
            raise NotImplementedError("MaxQuant DIA reader is not implemented yet.")
        else:
            raise ValueError(
                "Argument label should be one of 'tmt', 'label_free' and acquisition should be one of 'dda', 'dia'."
            )
        with capture_provenance_output() as stdout_buffer:
            mdata = reader.read()
        return append_cmd_log(
            mdata,
            function="read_maxquant",
            payload=get_bound_call_kwargs(
                self.__call__,
                identification_file,
                label,
                acquisition,
                _quantification=_quantification,
            ),
            stdout=stdout_buffer.getvalue().strip() or None,
        )

    def from_pg(self, *args: Any, **kwds: Any) -> md.MuData:
        """
        Reads MaxQuant protein group output and returns a MuData object.
        """
        raise NotImplementedError("MaxQuant protein group reader is not implemented yet.")


read_maxquant: _MaxQuantFacade = _MaxQuantFacade()
"""Alias for :class:`_MaxQuantFacade`.

Parameters:
    identification_file: Path to the MaxQuant output directory.
    label: Label for the MaxQuant output ('tmt' or 'label_free').
    acquisition: Acquisition method ('dda' or 'dia').

Returns:
    A MuData object containing the MaxQuant data at precursor level

Usage:
    mdata_precursor = mm.read_maxquant(search_dir)
    mdata_protein_group = mm.read_maxquant.from_pg(search_dir)
"""


class FragPipeFacade:

    __name__ = "read_fragpipe"

    def __call__(
        self,
        identification_file: str | Path,
        label: Literal["tmt", "label_free"],
        acquisition: Literal["dda", "dia"],
        quantification_file: str | Path | None = None,
    ) -> md.MuData:
        if label == "tmt" and acquisition == "dda":
            reader = TmtFragPipeReader(identification_file=identification_file, quantification_file=quantification_file)
        elif label == "label_free" and acquisition == "dda":
            reader = LfqFragPipeReader(identification_file=identification_file, quantification_file=quantification_file)
        else:
            raise ValueError(
                "Argument label should be one of 'tmt', 'label_free' and acquisition should be one of 'dda', 'dia'."
            )

        with capture_provenance_output() as stdout_buffer:
            mdata = reader.read()
        return append_cmd_log(
            mdata,
            function="read_fragpipe",
            payload=get_bound_call_kwargs(
                self.__call__,
                identification_file,
                label,
                acquisition,
                quantification_file=quantification_file,
            ),
            stdout=stdout_buffer.getvalue().strip() or None,
        )

    def from_pg(self):
        raise NotImplementedError("FragPipe protein group reader is not implemented yet.")


read_fragpipe: FragPipeFacade = FragPipeFacade()
"""Alias for :class:`FragPipeFacade`.

Parameters:
    identification_file: Path to the FragPipe output directory.
    quantification_file: Path to the FragPipe quantification file (if applicable, for LFQ).
    label: Label for the FragPipe output ('tmt' or 'label_free').
    acquisition: Acquisition method ('dda' or 'dia').

Returns:
    A MuData object containing the FragPipe data at PSM level

Usage:
    mdata = mm.read_fragpipe(identification_file, quantification_file, label, acquisition)
"""


def read_h5mu(h5mu_file: str | Path) -> md.MuData:
    """
    Reads an h5mu file (HDF5) and returns a MuData object.

    Parameters:
        h5mu_file: Path to the H5MU file.

    Returns:
        A MuData object.
    """
    mdata = md.read_h5mu(h5mu_file)
    mdata = normalize_cmd_for_runtime(mdata)
    return append_cmd_log(
        mdata,
        function="read_h5mu",
        payload=get_bound_call_kwargs(read_h5mu, h5mu_file),
    )


#######################################################################
# Placeholder functions for future implementations
########################################################################
def read_comet():
    raise NotImplementedError


def read_protdiscov():
    raise NotImplementedError
