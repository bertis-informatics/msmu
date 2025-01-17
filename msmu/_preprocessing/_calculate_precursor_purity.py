import pandas as pd
import numpy as np
from pathlib import Path
import re

from anndata import AnnData
from pymzml.run import Reader as MzmlReader

from ..tools._mzml import read_mzml, get_frame_df, get_ms2_isolation_window
from ..tools._bmskit import get_spectrum, get_composition, get_envelope, get_peptide_mass, get_isotope_peak


def calculate_precursor_purity(
    adata: AnnData,
    mzml_files: list[str | Path],
) -> pd.DataFrame:
    psm_df: pd.DataFrame = adata.varm["search_result"]  # SHOULD CHANGE TO VAR

    purities: list[pd.DataFrame] = []
    for mzml_file in mzml_files:
        mzml_file_name: str = Path(mzml_file).name
        psm_df_by_file: pd.DataFrame = psm_df.loc[psm_df["filename"] == mzml_file_name]

        mzml: MzmlReader = read_mzml(mzml_file)
        frame_df: pd.DataFrame = get_frame_df(mzml)

        purities_by_file: pd.DataFrame = _calculate_precursor_purity_by_file(mzml, frame_df, psm_df_by_file)
        purities.append(purities_by_file)

    return pd.concat(purities)["purity"]


def _calculate_precursor_purity_by_file(
    mzml: MzmlReader,
    frame_df: pd.DataFrame,
    psm_df_by_file: pd.DataFrame,
) -> pd.DataFrame:
    purity_colname = "purity"
    purity_df_by_file = pd.DataFrame(columns=["psm_id", "filename", purity_colname])
    purity_df_by_file["psm_id"] = psm_df_by_file["psm_id"]
    purity_df_by_file["filename"] = psm_df_by_file["filename"]

    for row_idx, psm_row in psm_df_by_file.iterrows():
        ms2_scan_id: list[int] = [s for s in psm_row["scannr"].split(" ") if "scan=" in s][0].split("=")[1]
        charge = psm_row["charge"]
        pep_seq = _get_peptide(psm_row)  # NOT NEEDED FOR FUTURE VERSION
        modif_info = _get_modifications(psm_row)  # NOT NEEDED FOR FUTURE VERSION

        ms1_scan_id: int = frame_df.loc[frame_df["ID"].astype(int) == int(ms2_scan_id), "ms1_scan_id"].values[0]
        ms1_spectrum = get_spectrum(mzml, ms1_scan_id)
        ms2_frame = mzml[int(ms2_scan_id)]
        isolation_window = get_ms2_isolation_window(ms2_frame)

        composition = get_composition(pep_seq)
        envelope = get_envelope(composition)
        pep_mass = get_peptide_mass(composition=composition, modif_info=modif_info)

        isotope_peaks = get_isotope_peak(
            ms1_spectrum=ms1_spectrum,
            pep_mass=pep_mass,
            charge=charge,
            envelope=envelope,
        )

        purity = _calculate_precursor_purity_by_psm(
            ms1_spectrum=ms1_spectrum,
            isotope_peaks=isotope_peaks,
            isolation_window=isolation_window,
        )
        purity_df_by_file.loc[row_idx, purity_colname] = purity

    return purity_df_by_file


def _get_peptide(psm_row):
    peptide_with_modi = psm_row["peptide"]
    pattern = r"([A-Z]+)|(\[\+\d+\.\d+\])"
    result = re.findall(pattern, peptide_with_modi)

    # 대문자와 숫자를 분리하여 각각 리스트로 저장
    peptide = [item[0] for item in result if item[0]]

    return "".join(peptide)


def _get_modifications(psm_row):
    pattern = r"([A-Z]+)|(\[\+\d+\.\d+\])"
    result = re.findall(pattern, psm_row["peptide"])

    # label modified AA & position
    mod_pos = 0
    mod_pos_AA = "N-term"
    modified_peptide = []
    for item in result:
        if item[1]:
            pos = mod_pos if mod_pos > 0 else ""
            mod_mass = float(item[1][1:-1])
            mod = f"{pos}{mod_pos_AA}({mod_mass})"
            modified_peptide.append(mod)
        else:
            mod_pos += len(item[0])
            mod_pos_AA = item[0][-1]

    return ", ".join(modified_peptide)


def _calculate_precursor_purity_by_psm(
    ms1_spectrum,
    isotope_peaks,
    isolation_window,
) -> float:
    if isotope_peaks is None:  # No isotope peak in ms1 spectrum
        purity = -1.0
    else:
        iso_min, iso_max = isolation_window
        isotype_peak_condition = (isotope_peaks.mz_array >= iso_min) & (isotope_peaks.mz_array <= iso_max)
        is_peak_in_index = np.where(isotype_peak_condition)[0]

        isotope_peaks_within_ab = isotope_peaks.ab_array[is_peak_in_index]
        isotope_peaks_sum = isotope_peaks_within_ab.sum()

        _, all_isolation_peaks_ab = ms1_spectrum.get_peaks_within(iso_min, iso_max)
        all_peaks_sum = all_isolation_peaks_ab.sum()

        if all_peaks_sum == 0:  # zero division # No isolation peak in isolation window
            purity = -2.0

        else:
            purity = (isotope_peaks_sum / all_peaks_sum).round(4)

    return purity
