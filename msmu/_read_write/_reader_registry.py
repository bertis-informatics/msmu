from pathlib import Path
from typing import Any, Literal
import mudata as md
import logging

from ._base_reader import SearchResultDataFrameConverter
from ._diann import DiannReader, DiannProteinGroupReader
from ._sage import LfqSageReader, TmtSageReader
from ._maxquant import MaxTmtReader, MaxLfqReader, MaxDiaReader
from ._fragpipe import TmtFragPipeReader, LfqFragPipeReader
from ._cptac import TmtCPTACReader, LfqCPTACReader, CPTACDataFrameConverter

from .._utils._provenance import (
    append_cmd_log,
    capture_provenance_output,
    get_bound_call_kwargs,
    normalize_cmd_for_runtime,
)

logger = logging.getLogger(__name__)


def read_sage(
    identification_file: str | Path,
    label: Literal["tmt", "label_free"],
    quantification_file: str | Path | None = None,
) -> md.MuData:
    """
    Reads Sage output and returns a MuData object.

    Parameters:
        identification_file: Path to the results.sage.tsv.
        label: Label for the Sage output ('tmt' or 'label_free').
        quantification_file: Whether to include quantification data. Default is None.

    Returns:
        A MuData object containing the Sage data.
    """

    logger.info(f"Reading SAGE Identification data: {len(identification_file)} file(s)")
    identification_file_, identification_df_ = SearchResultDataFrameConverter().convert(identification_file)
    if quantification_file is not None:
        logger.info(f"Reading SAGE Quantification data: {len(quantification_file)} file(s)")
        quantification_file_, quantification_df_ = SearchResultDataFrameConverter().convert(quantification_file)
    else:
        quantification_file_, quantification_df_ = None, None

    if label == "tmt":
        reader = TmtSageReader(
            identification_file=identification_file_,
            identification_df=identification_df_,
            quantification_file=quantification_file_,
            quantification_df=quantification_df_,
        )
    elif label == "label_free":
        reader = LfqSageReader(
            identification_file=identification_file_,
            identification_df=identification_df_,
            quantification_file=quantification_file_,
            quantification_df=quantification_df_,
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
        identification_files = []
        if isinstance(identification_file, list):
            identification_files = identification_file
        elif isinstance(identification_file, (str, Path)):
            identification_files = [identification_file]
        else:
            raise ValueError("Argument identification_file should be a string, Path, or list of strings/Paths.")

        logger.info(f"Reading MaxQuant data from {len(identification_files)} file(s)")
        identification_file_, identification_df_ = SearchResultDataFrameConverter().convert(identification_files)

        if label == "tmt" and acquisition == "dda":
            reader = MaxTmtReader(
                identification_file=identification_file_,
                identification_df=identification_df_,
            )
        elif label == "label_free" and acquisition == "dda":
            reader = MaxLfqReader(
                identification_file=identification_file_,
                identification_df=identification_df_,
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
        identification_file: str | Path | list,
        label: Literal["tmt", "label_free"],
        acquisition: Literal["dda", "dia"],
        quantification_file: str | Path | None = None,
    ) -> md.MuData:

        identification_files = []
        if isinstance(identification_file, list):
            identification_files = identification_file
        elif isinstance(identification_file, (str, Path)):
            identification_files = [identification_file]
        else:
            raise ValueError("Argument identification_file should be a string, Path, or list of strings/Paths.")

        if quantification_file is not None:
            quantification_files = []
            if isinstance(quantification_file, list):
                quantification_files = quantification_file
            elif isinstance(quantification_file, (str, Path)):
                quantification_files = [quantification_file]
            else:
                raise ValueError("Argument quantification_file should be a string, Path, or list of strings/Paths.")

        logger.info(f"Reading FragPipe data from {len(identification_files)} identification file(s)")
        identification_file_, identification_df_ = SearchResultDataFrameConverter().convert(identification_files)

        if label == "tmt" and acquisition == "dda":
            reader = TmtFragPipeReader(identification_file=identification_file_, identification_df=identification_df_)
        elif label == "label_free" and acquisition == "dda":
            if quantification_file is not None:
                logger.info(
                    f"Reading FragPipe quantification data from {len(quantification_files)} quantification file(s)"
                )
                quantification_file_, quantification_df_ = SearchResultDataFrameConverter().convert(
                    quantification_files
                )
            else:
                quantification_file_, quantification_df_ = None, None

            reader = LfqFragPipeReader(
                identification_file=identification_file_,
                identification_df=identification_df_,
                quantification_file=quantification_file_,
                quantification_df=quantification_df_,
            )
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


def read_cptac(
    identification_file: str | Path | list,
    label: Literal["tmt"],
    max_workers: int = 4,
    drop_search_result: bool = False,
) -> md.MuData:
    """
    Reads a CPTAC output file and returns a MuData object.

    Parameters:
        identification_file: Path to the CPTAC output file (mzid format).
        label: Label for the CPTAC output ('tmt'). Currently, only 'tmt' is supported.
        max_workers: Maximum number of worker processes to use for reading multiple files. Default is 4.
        drop_search_result: Whether to drop the search result after reading. Default is False.

    Returns:
        A MuData object.
    """
    mzid_files = []
    if isinstance(identification_file, list):
        mzid_files = identification_file
    elif isinstance(identification_file, (str, Path)):
        mzid_files = [identification_file]
    else:
        raise ValueError("Argument identification_file should be a string, Path, or list of strings/Paths.")

    if label not in ["tmt"]:
        raise ValueError("Argument label should be one of 'tmt'.")

    logger.info(f"Reading CPTAC data from {len(mzid_files)} mzid file(s)")
    identification_file_, identification_df_ = CPTACDataFrameConverter().convert(
        file_paths=mzid_files, max_workers=max_workers
    )

    if label == "tmt":
        reader = TmtCPTACReader(
            identification_file=identification_file_,
            identification_df=identification_df_,
            _drop_search_result=drop_search_result,
        )
    elif label == "label_free":
        raise NotImplementedError("LFQ CPTAC reader is not implemented yet.")
        # reader = LfqCPTACReader(identification_file=identification_file_, identification_df=identification_df_)

    mdata: md.MuData = reader.read()

    return mdata


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
