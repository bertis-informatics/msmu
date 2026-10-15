import bz2
import gzip
from io import BytesIO
import lzma
from zipfile import ZipFile

import polars as pl
import pytest

from msmu._core._tabular import read_delimited


@pytest.mark.parametrize("compression", [None, "gzip", "bz2", "xz", "zip"])
def test_delimited_compression_preserves_decimal_values(tmp_path, compression):
    content = b"Sequence\tIntensity_s1\nAA\t2927.7894287109375\n"
    if compression == "zip":
        buffer = BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("quant.tsv", content)
        content = buffer.getvalue()
    elif compression:
        content = {"gzip": gzip.compress, "bz2": bz2.compress, "xz": lzma.compress}[compression](content)
    path = tmp_path / "quant.tsv"
    path.write_bytes(content)

    frame = read_delimited(path, pandas_headers=True)

    assert frame.columns == ["Sequence", "Intensity_s1"]
    assert frame["Intensity_s1"].dtype == pl.Float64
    assert frame["Intensity_s1"].to_list() == [2927.7894287109375]


@pytest.mark.parametrize("strings_only", [False, True])
def test_delimited_preserves_headers_and_missing_trailing_fields(strings_only):
    content = b' \n\n"a\na"\ta\ta\ta_duplicated_0\t\n1\t2\t3\n'

    frame = read_delimited(BytesIO(content), strings_only=strings_only, pandas_headers=True)

    assert frame.columns == ["a\na", "a", "a.1", "a_duplicated_0", "Unnamed: 4"]
    expected = [("1", "2", "3", None, None)] if strings_only else [(1, 2, 3, None, None)]
    assert frame.rows() == expected


def test_delimited_header_only_file_retains_columns():
    frame = read_delimited(BytesIO(b"a\ta\ta.1\n"), pandas_headers=True)

    assert frame.shape == (0, 3)
    assert frame.columns == ["a", "a.2", "a.1"]


@pytest.mark.parametrize("separator", [",", "\t"])
def test_delimited_all_unnamed_headers_keep_first_data_row(separator):
    content = f"{separator}\nA{separator}1\n".encode()

    frame = read_delimited(BytesIO(content), separator=separator, pandas_headers=True)

    assert frame.columns == ["Unnamed: 0", "Unnamed: 1"]
    assert frame.rows() == [("A", 1)]


def test_delimited_rejects_ambiguous_zip_members(tmp_path):
    path = tmp_path / "quant.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("one.tsv", "a\n1\n")
        archive.writestr("two.tsv", "a\n2\n")

    with pytest.raises(ValueError, match="exactly one file"):
        read_delimited(path, pandas_headers=True)
