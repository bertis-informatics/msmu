import pandas as pd
import mudata as md
from pathlib import Path


def write_flashlfq_input(mdata: md.MuData, filename: str | Path) -> None:
    """
    Exports MuData psm object to FlashLFQ format.

    Parameters:
        mdata: MuData object containing the data to export.
        filename: Path to the output FlashLFQ file.
    """
    required_column_dict: dict[str, str] = {
        "filename": "File Name",
        "rt": "Scan Retention Time",
        "charge": "Precursor Charge",
        "stripped_peptide": "Base Sequence",
        "peptide": "Full Sequence",
        "calcmass": "Peptide Monoisotopic Mass",
        "proteins": "Protein Accession",
    }

    source_df: pd.DataFrame = mdata["psm"].var.copy()

    source_df = source_df[required_column_dict.keys()]
    source_df = source_df.rename(columns=required_column_dict)

    source_df.to_csv(filename, sep="\t", index=False)


def write_csv(
    mdata: md.MuData,
    modality: str,
    filename: str | Path,
    sep: str,
    include: str | list[str] | None = None,
    exclude: str | list[str] | None = None,
    quantification: bool = True,
) -> None:
    """
    Exports MuData modalities to CSV/TSV files.

    Parameters:
        mdata: MuData object containing the data to export.
        modality: The modality to export (e.g., 'psm', 'peptide', 'protein').
        filename: Path to the output file.
        sep: Separator for the output file (e.g., ',', '\t').
        include: List of columns to include.
        exclude: List of columns to exclude.
        quantification: Whether to include quantification data.
    """
    df = to_readable(
        mdata,
        modality=modality,
        include=include,
        exclude=exclude,
        quantification=quantification,
    )
    df.to_csv(filename, sep=sep, index=False)


def to_readable(
    mdata: md.MuData,
    modality: str,
    include: str | list[str] | None = None,
    exclude: str | list[str] | None = None,
    quantification: bool = True,
) -> pd.DataFrame:
    """Convert MuData modality to a human-readable format.

    Parameters:
        mdata: MuData object containing the data to convert.
        modality: The modality to convert (e.g., 'psm', 'peptide', 'protein').
        include: List of columns to include.
        exclude: List of columns to exclude.
        quantification: Whether to include quantification data.

    Returns:
        A pandas DataFrame in a human-readable format.
    """
    df = mdata[modality].var.copy()

    if include is None and exclude is None and not quantification:
        return df

    if include:
        if isinstance(include, str):
            include = [include]
        df = df[include]
    if exclude:
        if isinstance(exclude, str):
            exclude = [exclude]
        df = df.drop(columns=exclude)
    if quantification:
        quant_df = mdata[modality].to_df().T
        df = pd.concat([df, quant_df], axis=1)

    return df
