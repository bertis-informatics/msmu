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

    source_df: pd.DataFrame = mdata.mod["psm"].var.copy()

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


def write_pin(
    mdata: md.MuData,
    filename: str | Path | None = None,
) -> pd.DataFrame | None:
    """
    Exports MuData psm object to Percolator input format.

    Parameters:
        mdata: MuData object containing the data to export.
        filename: Path to the output Percolator input file. If None, the function will return a DataFrame instead of writing to a file.

    Returns:
        A pandas DataFrame in Percolator input format if filename is None, otherwise None.
    """
    var_columns = ["filename", "scan_num", "charge", "peptide", "proteins", "calcmass", "expmass"]

    target_df: pd.DataFrame = mdata.mod["psm"].var[var_columns].copy()
    target_df["decoy"] = 0

    if "decoy" in mdata.mod["psm"].uns:
        decoy_df = mdata.mod["psm"].uns["decoy"].copy()

        pin_df = pd.concat([target_df, decoy_df], axis=0)
    else:
        pin_df = target_df
    pin_df["SpecId"] = pin_df.index.astype(str)
    pin_df["Spectra"] = pin_df.index.astype(str)
    pin_df["Label"] = pin_df["decoy"].apply(lambda x: -1 if x == 1 else 1)
    pin_df["ScanNr"] = pin_df["scan_num"].astype(int)
    pin_df["PepLen"] = pin_df["peptide_length"].astype(int)

    pin_df = pin_df.rename(
        columns={
            "filename": "FileName",
            "charge": "Charge",
            "peptide": "Peptide",
            "proteins": "Proteins",
            "calcmass": "CalcMass",
            "expmass": "ExpMass",
            "score": "XCorr",
        }
    )

    pin_req_columns = [
        "SpecId",
        "Label",
        "Peptide",
        "Proteins",
        "Charge",
        "ScanNr",
        "PepLen",
        "CalcMass",
        "ExpMass",
        "XCorr",
    ]
    pin_df = pin_df[pin_req_columns]

    if filename is None:
        return pin_df
    else:
        pin_df.to_csv(filename, sep="\t", index=False)


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
    df = mdata.mod[modality].var.copy()

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
        quant_df = mdata.mod[modality].to_df().T
        df = pd.concat([df, quant_df], axis=1)

    return df
