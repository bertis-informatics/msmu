"""Shared Polars text ingestion for search results, quantification and metadata."""

import csv
import bz2
import gzip
import lzma
import zipfile
from contextlib import ExitStack
from io import TextIOWrapper
from os import PathLike

import polars as pl

from ._sources import open_source


# Keep the missing-value vocabulary used by the existing search-result readers.
_PANDAS_NA_VALUES = [
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
]


def _pandas_column_names(header):
    """Retain existing duplicate suffixes, reserving explicitly named columns first."""
    names = [name or f"Unnamed: {i}" for i, name in enumerate(header)]
    reserved = set(names)
    used = set()
    for i in [i for i, name in enumerate(header) if name] + [i for i, name in enumerate(header) if not name]:
        name = names[i]
        if name in used:
            suffix = 1
            while f"{name}.{suffix}" in reserved or f"{name}.{suffix}" in used:
                suffix += 1
            names[i] = f"{name}.{suffix}"
        used.add(names[i])
    return names


def read_delimited(source, *, separator="\t", strings_only=False, pandas_headers=False) -> pl.DataFrame:
    """Read file/URL values with Polars, retaining each caller's text-input contract."""
    settings = dict(
        separator=separator,
        infer_schema_length=0 if strings_only else None,
        null_values=[""] if strings_only else _PANDAS_NA_VALUES,
    )
    with open_source(source) as buffer:
        if not pandas_headers:
            return pl.read_csv(buffer, **settings)

        # Parse only the header as text; all data values and inference stay in Polars.
        with ExitStack() as stack:
            binary = stack.enter_context(open(buffer, "rb")) if isinstance(buffer, (str, PathLike)) else buffer
            magic = binary.read(6)
            binary.seek(0)
            for prefix, opener in [(b"\x1f\x8b", gzip.open), (b"BZh", bz2.open), (b"\xfd7zXZ\x00", lzma.open)]:
                if magic.startswith(prefix):
                    binary = stack.enter_context(opener(binary))
                    break
            if magic.startswith(b"PK\x03\x04"):
                archive = stack.enter_context(zipfile.ZipFile(binary))
                entries = archive.namelist()
                if len(entries) != 1:
                    raise ValueError("Expected exactly one file in the ZIP archive.")
                binary = stack.enter_context(archive.open(entries[0]))
            text = TextIOWrapper(binary, encoding="utf-8-sig", newline="")
            try:
                skipped_rows = 0
                while True:
                    position = text.tell()
                    line = text.readline()
                    if not line:
                        raise pl.exceptions.NoDataError("empty CSV")
                    if line.strip() or separator in line:
                        text.seek(position)
                        header = next(csv.reader(text, delimiter=separator))
                        break
                    skipped_rows += 1
            finally:
                text.detach()
                binary.seek(0)
            names = _pandas_column_names(header)
            frame = pl.read_csv(binary, has_header=False, skip_rows=skipped_rows + 1, raise_if_empty=False, **settings)
            if frame.width == 0:
                return pl.DataFrame(schema={name: pl.String for name in names})
            # Omitted trailing fields still belong to the header's columns.
            frame.columns = names[:frame.width]
            if frame.width < len(names):
                frame = frame.with_columns(
                    pl.lit(None, dtype=pl.String if strings_only else pl.Float64).alias(name)
                    for name in names[frame.width:]
                )
            return frame
