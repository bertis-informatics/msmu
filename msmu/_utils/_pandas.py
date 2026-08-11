import pandas as pd


def split_delimited_strings(values: pd.Series, delimiter: str) -> pd.Series:
    """Split a delimited string column into lists, tolerating categorical dtype.

    ``write_h5mu`` stores var string columns as categorical, so any column read back from disk is
    categorical rather than ``str``. Applying ``.str.split`` to a categorical in pandas 3 yields the
    *repr* of the split list (``"['A', 'B']"``) instead of the list itself, which turns the following
    ``explode`` into a no-op and silently corrupts the values. Decategorising first keeps the result
    a real list column, missing values included.

    Parameters:
        values: string column to split
        delimiter: separator to split on

    Returns:
        column of lists aligned to the input index
    """
    if isinstance(values.dtype, pd.CategoricalDtype):
        values = values.astype(str)

    return values.str.split(delimiter)
