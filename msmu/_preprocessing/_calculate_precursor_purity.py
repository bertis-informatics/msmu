import pandas as pd
import numpy as np
from pathlib import Path

from mudata import MuData
from pymzml.run import Reader as MzmlReader

from ..tools._mzml import read_mzml, get_frame_df, get_ms2_isolation_window
from ..tools._bmskit import get_spectrum, get_composition, get_envelope, get_peptide_mass, get_isotope_peak


def calculate_precursor_purity(
    mdata: MuData,
    level: str,
) -> pd.DataFrame:
    psm_df: pd.DataFrame = mdata[level].varm["search_result"]  # SHOULD CHANGE TO VAR
    mzml_files: list[str | Path] = mdata[level].uns["mzml_files"]

    purities: list[pd.DataFrame] = []
    for mzml_file in mzml_files:
        mzml_file_name: str = Path(mzml_file).name
        psm_df_by_file: pd.DataFrame = psm_df.loc[psm_df["filename"] == mzml_file_name]

        mzml: MzmlReader = read_mzml(mzml_file)
        frame_df: pd.DataFrame = get_frame_df(mzml)

        purities_by_file: pd.DataFrame = _calculate_precursor_purity_by_file(mzml, frame_df, psm_df_by_file)
        purities.append(purities_by_file)

    return merge_purity(purities)


def _calculate_precursor_purity_by_file(
    mzml: MzmlReader,
    frame_df: pd.DataFrame,
    psm_df: pd.DataFrame,
) -> pd.DataFrame:
    purity_colname = "purity"
    purity_df_by_file = pd.DataFrame(columns=["psm_id", "filename", purity_colname])
    purity_df_by_file["psm_id"] = psm_df["psm_id"]
    purity_df_by_file["filename"] = psm_df["filename"]

    for row_idx, psm_row in psm_df.iterrows():
        ms2_scan_id = psm_row["scannr"].str.split("scan=").str[1].astype(int)
        charge = psm_row["charge"]
        pep_seq = psm_row["peptide"]
        modif_info = psm_row["modifications"]

        ms1_scan_id: int = frame_df.loc[frame_df["ID"] == ms2_scan_id, "ms1_scan_id"]
        ms1_spectrum = get_spectrum(mzml, ms1_scan_id)
        ms2_frame = mzml[ms2_scan_id]
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


def _calculate_precursor_purity_by_psm(
    ms1_spectrum,
    isotope_peak,
    isolation_window,
) -> float:
    if isotope_peak is None:  # No isotope peak in ms1 spectrum
        purity = -1
    else:
        iso_min, iso_max = isolation_window
        isotype_peak_condition = (isotope_peak.mz_array >= iso_min) & (isotope_peak.mz_array <= iso_max)
        is_peak_in_index = np.where(isotype_peak_condition)[0]

        isotope_peaks_within_ab = isotope_peak.ab_array[is_peak_in_index]
        isotope_peaks_sum = isotope_peaks_within_ab.sum()

        _, all_isolation_peaks_ab = ms1_spectrum.get_peaks_within(iso_min, iso_max)
        all_peaks_sum = all_isolation_peaks_ab.sum()

        if all_peaks_sum == 0:  # zero division # No isolation peak in isolation window
            purity = -2

        else:
            purity = (isotope_peaks_sum / all_peaks_sum).round(4)

    return purity


def merge_purity(psm_df: pd.DataFrame, purities: list[pd.DataFrame]) -> pd.DataFrame:
    purities_df = pd.concat(purities)["psm_id", "purity"]
    psm_df = psm_df.merge(purities_df, on="psm_id", how="left")
    return purities
