from pathlib import Path

import mudata as md
import pandas as pd

from ._diann_reader import DiannReader
from ._sage_reader import LfqSageReader, TmtSageReader


def read_sage(
    sage_output_dir: str | Path,
    sample_name: list[str],
    label: str,
    channel: list[str] | None = None,
    filename: list[str] | None = None,
) -> md.MuData:
    if label == "tmt":
        reader_cls = TmtSageReader
    elif label == "lfq":
        reader_cls = LfqSageReader
    else:
        raise ValueError("Argument label should be one of 'Tmt', 'lfq'.")

    reader = reader_cls(
        sage_output_dir=sage_output_dir,
        sample_name=sample_name,
        channel=channel,
        filename=filename,
    )
    mdata = reader.read()

    return mdata


def read_diann():
    return NotImplementedError


def merge_mudata(): ...


def make_sample_annotation(): ...


# TODO: MARK IS_BLANK and IS_IRS function for TMT
