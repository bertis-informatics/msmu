import pandas as pd

from .._utils._mudata import add_modality as add_modality


def to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts object- and string-type columns in a DataFrame to categorical.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with object and string columns converted to categorical.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = pd.Categorical(df[col])

    return df
