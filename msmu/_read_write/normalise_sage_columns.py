import re

import pandas as pd


def rename_sage_columns(sage_result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns in the Sage result DataFrame to standardized names.

    Args:
        sage_result_df (pd.DataFrame): Input DataFrame with original column names.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    rename_dict: dict[str, str] = {
        "Spectrum File": "filename",
        "Observed Mass": "expmass",
        "Calculated Peptide Mass": "calcmass",
        "Retention": "rt",
        "Charge": "charge",
        "Peptide Length": "peptide_len",
        "Number of Missed Cleavage": "missed_cleavages",
    }

    return sage_result_df.rename(columns=rename_dict)


def normalise_sage_columns(sage_result_df) -> pd.DataFrame:
    """
    Normalizes the Sage result DataFrame by selecting relevant columns and adding derived columns.

    Args:
        sage_result_df (pd.DataFrame): Input DataFrame with Sage results.

    Returns:
        pd.DataFrame: Normalized DataFrame with additional columns.
    """
    used_cols: list[str] = [
        "proteins",
        "peptide",
        "filename",
        "scan_num",
        "charge",
        "missed_cleavages",
        "spectrum_q",
        "peptide_q",
        "protein_q",
    ]
    normalised_sage_result_df = sage_result_df[used_cols].copy()

    # Extract stripped peptide and modifications
    normalised_sage_result_df[["stripped_peptide", "modifications"]] = sage_result_df[
        "peptide"
    ].apply(lambda x: pd.Series(make_peptide(x)))

    # Calculate observed m/z
    normalised_sage_result_df["observed_mz"] = sage_result_df.apply(
        lambda x: make_observed_mz(x["expmass"], x["charge"]), axis=1
    )

    return normalised_sage_result_df


def make_peptide(peptide_input: str) -> tuple[str, str]:
    """
    Parses a peptide string to extract the stripped peptide and modifications.

    Args:
        peptide_input (str): Input peptide string with modifications.

    Returns:
        Tuple[str, str]: Stripped peptide and a string of modifications.
    """
    pattern = r"([A-Z]+)|(\[\+\d+\.\d+\])"
    splited_peptide = re.findall(pattern, peptide_input)

    stripped_peptide = "".join([item[0] for item in splited_peptide if item[0]])

    mod_pos = 0
    AA_mod_pos = "N-term"
    modifications: list[str] = []

    for item in splited_peptide:
        if item[1]:  # Modification found
            pos = mod_pos if mod_pos > 0 else ""
            mod_mass = float(item[1].strip("[]+"))
            mod = f"{pos}{AA_mod_pos}({mod_mass})"
            modifications.append(mod)
        else:  # Amino acid found
            mod_pos += len(item[0])
            AA_mod_pos = item[0][-1]

    return stripped_peptide, ", ".join(modifications)


def make_observed_mz(observed_mass: float, charge: int) -> float:
    """
    Calculates the observed m/z value.

    Args:
        observed_mass (float): Observed mass of the peptide.
        charge (int): Charge of the peptide.

    Returns:
        float: Observed m/z value.
    """
    if charge == 0:
        raise ValueError("Charge cannot be zero when calculating observed m/z.")
    return observed_mass / charge
