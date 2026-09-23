from pathlib import Path
from typing import Literal
import mudata as md
import pandas as pd

from ..logging_utils import get_logger
from ._base_reader import SearchResultDataFrameConverter
from ._diann import DiannReader
from ._sage import LfqSageReader, TmtSageReader
from ._maxquant import MaxTmtReader, MaxLfqReader, MaxQuantDataFrameConverter
from ._fragpipe import TmtFragPipeReader, LfqFragPipeReader
from ._delpi import DelpiReader
from .._preprocessing._meta import read_sdrf as _read_sdrf

from .._core._provenance import log_provenance

logger = get_logger(__name__)


@log_provenance
def read_sage(
    identification_file: str | Path,
    label: Literal["tmt", "label_free"],
    quantification_file: str | Path | None = None,
    drop_search_result: bool = False,
) -> md.MuData:
    """
    Reads Sage output and returns a MuData object.

    Parameters:
        identification_file: Path or URL to results.sage.tsv, or a list of input files.
        label: Label for the Sage output ('tmt' or 'label_free').
        quantification_file: Path or URL to tmt.tsv (required for TMT) or lfq.tsv.
            Omit for an identification-only label-free import. A list of files is also accepted.
        drop_search_result: If True, omit the raw search table from varm["search_result"].

    Returns:
        A MuData object containing the Sage data.
    """
    identification_files = []
    if isinstance(identification_file, list):
        identification_files = identification_file
    elif isinstance(identification_file, (str, Path)):
        identification_files = [identification_file]
    else:
        raise ValueError("Argument identification_file should be a string, Path, or list of strings/Paths.")

    if quantification_file is not None:
        if isinstance(quantification_file, list):
            quantification_files = quantification_file
        elif isinstance(quantification_file, (str, Path)):
            quantification_files = [quantification_file]
        else:
            raise ValueError("Argument quantification_file should be a string, Path, or list of strings/Paths.")
    else:
        if label == "tmt":
            logger.error("Quantification file is required for TMT-labeled Sage data.")
            raise ValueError("Quantification file is required for TMT-labeled Sage data.")

        quantification_files = []
        quantification_file_, quantification_df_ = None, None
        logger.debug("No Sage quantification file provided for label-free input.")

    logger.info(f"Reading SAGE Identification data: {len(identification_files)} file(s)")
    identification_file_, identification_df_ = SearchResultDataFrameConverter().convert(identification_files)

    if quantification_files:
        logger.info(f"Reading SAGE Quantification data: {len(quantification_files)} file(s)")
        quantification_file_, quantification_df_ = SearchResultDataFrameConverter().convert(quantification_files)
    else:
        logger.debug("Skipping Sage quantification import because no files were provided.")

    if label == "tmt":
        reader = TmtSageReader(
            identification_file=identification_file_,
            identification_df=identification_df_,
            quantification_file=quantification_file_,
            quantification_df=quantification_df_,
            drop_search_result=drop_search_result,

        )
    elif label == "label_free":
        reader = LfqSageReader(
            identification_file=identification_file_,
            identification_df=identification_df_,
            quantification_file=quantification_file_,
            quantification_df=quantification_df_,
            drop_search_result=drop_search_result,
        )
    else:
        raise ValueError("Argument label should be one of 'tmt', 'label_free'.")
    logger.debug("Selected Sage reader: %s", type(reader).__name__)

    mdata: md.MuData = reader.read()
    return mdata


@log_provenance
def read_diann(
    identification_file: str | Path | list,
    drop_search_result: bool = False,
    level: Literal["precursor", "protein_group"] = "precursor",
) -> md.MuData:
    """
    Reads DIA-NN output and returns a MuData object.

    The block-diagonal precursor quantification is stored as a SciPy sparse ``.X``: the precursor
    pivot is ``(n_precursor_obs x n_run)`` with ~one non-null per row, so only the observed cells
    are built (no dense pivot). Downstream tools handle the sparse ``.X`` transparently (absent
    cells restored as NaN).

    Parameters:
        identification_file: Path or URL to report.tsv/report.parquet, or a list of files.
        drop_search_result: If True, omit the raw search table from varm["search_result"].
        level: Level of the output to read ('precursor' or 'protein_group').
            Note: 'protein_group' is not yet implemented.

    Returns:
        A MuData object containing the DIA-NN data.
    """
    if level == "protein_group":
        raise NotImplementedError("Protein group level reading is not yet implemented.")

    identification_files = []
    if isinstance(identification_file, list):
        identification_files = identification_file
    elif isinstance(identification_file, (str, Path)):
        identification_files = [identification_file]
    else:
        raise ValueError("Argument identification_file should be a string, Path, or list of strings/Paths.")

    identification_file_, identification_df_ = SearchResultDataFrameConverter().convert(identification_files)

    mdata = DiannReader(
        identification_file=identification_file_,
        identification_df=identification_df_,
        drop_search_result=drop_search_result,
    ).read()
    return mdata


@log_provenance
def read_maxquant(
    identification_file: str | Path | list,
    label: Literal["tmt", "label_free"],
    acquisition: Literal["dda", "dia"],
    drop_search_result: bool = False,
    _quantification: bool = True,
) -> md.MuData:
    """
    Reads MaxQuant output and returns a MuData object.

    Parameters:
        identification_file: Path or URL to evidence.txt, or a list of files.
        label: Label type ('tmt' or 'label_free').
        acquisition: Acquisition method ('dda' or 'dia'). Note: 'dia' is not yet implemented.
        drop_search_result: Whether to drop the raw search result after reading. Default is False.
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
    identification_file_, identification_df_ = MaxQuantDataFrameConverter().convert(identification_files)

    if label == "tmt" and acquisition == "dda":
        reader = MaxTmtReader(
            identification_file=identification_file_,
            identification_df=identification_df_,
            drop_search_result=drop_search_result,
        )
    elif label == "label_free" and acquisition == "dda":
        reader = MaxLfqReader(
            identification_file=identification_file_,
            identification_df=identification_df_,
            _quantification=_quantification,
            drop_search_result=drop_search_result,
        )
    elif label == "label_free" and acquisition == "dia":
        raise NotImplementedError("MaxQuant DIA reader is not yet implemented.")
    else:
        raise ValueError(
            "Argument label should be one of 'tmt', 'label_free' and acquisition should be one of 'dda', 'dia'."
        )
    logger.debug("Selected MaxQuant reader: %s", type(reader).__name__)

    mdata = reader.read()
    return mdata


@log_provenance
def read_fragpipe(
    identification_file: str | Path | list,
    label: Literal["tmt", "label_free"],
    acquisition: Literal["dda", "dia"],
    quantification_file: str | Path | list | None = None,
) -> md.MuData:
    """
    Reads FragPipe output and returns a MuData object.

    Parameters:
        identification_file: Path to the FragPipe PSM output file(s).
        label: Label type ('tmt' or 'label_free').
        acquisition: Acquisition method ('dda' or 'dia'). Note: 'dia' is not yet implemented.
        quantification_file: Path to combined_modified_peptide.tsv file(s) for LFQ intensities.
            Omit for an identification-only LFQ import. TMT intensities are read from the PSM file.

    Returns:
        A MuData object containing the FragPipe data.
    """
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
        reader = TmtFragPipeReader(
            identification_file=identification_file_,
            identification_df=identification_df_,
        )
    elif label == "label_free" and acquisition == "dda":
        if quantification_file is not None:
            logger.info(f"Reading FragPipe quantification data from {len(quantification_files)} quantification file(s)")
            quantification_file_, quantification_df_ = SearchResultDataFrameConverter().convert(quantification_files)
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

    mdata = reader.read()
    return mdata


@log_provenance
def read_delpi(identification_file: str | Path, drop_search_result: bool = False) -> md.MuData:
    """
    Reads a DELPI output file and returns a MuData object.

    Parameters:
        identification_file: Path to the DELPI output file.
        drop_search_result: If True, the raw search result is not stored in varm.

    Returns:
        A MuData object.
    """
    identification_files = []
    if isinstance(identification_file, list):
        identification_files = identification_file
    elif isinstance(identification_file, (str, Path)):
        identification_files = [identification_file]
    else:
        raise ValueError("Argument identification_file should be a string, Path, or list of strings/Paths.")

    identification_file_, identification_df_ = SearchResultDataFrameConverter().convert(identification_files)

    reader = DelpiReader(
        identification_file=identification_file_,
        identification_df=identification_df_,
        drop_search_result=drop_search_result,
    )

    mdata: md.MuData = reader.read()

    return mdata


def read_sdrf(
    sdrf_file: str | Path,
    *,
    validate_sdrf: bool = True,
) -> pd.DataFrame:
    """
    Reads tab-delimited SDRF metadata and returns an obs-ready DataFrame.

    Parameters:
        sdrf_file: Path or URL to the SDRF file.
        validate_sdrf: Validate the SDRF with sdrf-pipelines if installed. Default is True.

    Parsed SDRF columns preserve normalized SDRF header text, including spaces
    and square brackets.
    """
    return _read_sdrf(sdrf_file, validate=validate_sdrf)


@log_provenance
def read_h5mu(h5mu_file: str | Path) -> md.MuData:
    """
    Reads an h5mu file (HDF5) and returns a MuData object.

    Parameters:
        h5mu_file: Path to the H5MU file.

    Returns:
        A MuData object.
    """
    mdata = md.read_h5mu(h5mu_file)
    return mdata


#######################################################################
# Placeholder functions for future implementations
########################################################################
def read_comet():
    raise NotImplementedError


def read_protdiscov():
    raise NotImplementedError
