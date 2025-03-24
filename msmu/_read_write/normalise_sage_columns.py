import re

import pandas as pd


def rename_sage_columns(sage_result_df: pd.DataFrame) -> pd.DataFrame:
    rename_dict: dict[str, str] = {
        "Spectrum File": "filename",
        "Observed Mass": "expmass",
        "Calculated Peptide Mass": "calcmass",
        "Retention": "rt",
        "Charge": "charge",
        "Peptide Length": "peptide_len",
        "Number of Missed Cleavage": "missed_cleavages",
    }

    sage_result_df.rename(columns=rename_dict)

    return sage_result_df


def normalise_sage_columns(sage_result_df):
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

    (
        normalised_sage_result_df["stripped_peptide"],
        normalised_sage_result_df["modifications"],
    ) = zip(*sage_result_df["peptide"].apply(lambda x: make_peptide(x)))

    normalised_sage_result_df["observed_mz"] = sage_result_df["expmass"] / sage_result_df["charge"]

    return normalised_sage_result_df


# def make_protein(protein_input:str) -> tuple[str]:
#    protein_split = protein_input.split(';')
#
#    proteins = [protein for protein in protein_split if not protein.startswith("rev_")]
#    rev_proteis = [protein for protein in protein_split if protein.startswith("rev_")]
#
#    if proteins:
#        return proteins[0]
#
#
#
#    return protein, protein_group, rev_protein


def make_peptide(peptide_input):
    pattern = r"([A-Z]+)|(\[\+\d+\.\d+\])"
    splited_peptide = re.findall(pattern, peptide_input)

    stripped_peptide = "".join([item[0] for item in splited_peptide if item[0]])

    mod_pos = 0
    AA_mod_pos = "N-term"
    modifications: list = list()
    for item in splited_peptide:
        if item[1]:
            pos = mod_pos if mod_pos > 0 else ""
            mod_mass = float(item[1].strip("[").strip("]"))
            mod = f"{pos}{AA_mod_pos}({mod_mass})"
            modifications.append(mod)
        else:
            mod_pos += len(item[0])
            AA_mod_pos = item[0][-1]

    return stripped_peptide, ", ".join(modifications)


def make_observed_mz(observed_mass: float, charge: int) -> float:
    return observed_mass / charge
