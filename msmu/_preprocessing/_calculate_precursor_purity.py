import pandas as pd
import numpy as np
from pathlib import Path
import concurrent.futures

from anndata import AnnData
from pymzml.run import Reader as MzmlReader

from ..tools._mzml import read_mzml, get_frame_df, get_ms2_isolation_window
from ..tools._bmskit import get_spectrum, get_composition, get_envelope, get_peptide_mass, get_isotope_peak


def calculate_precursor_purity(
    adata: AnnData,
    mzml_files: list[str | Path],
    n_cores: int = 1,
) -> pd.DataFrame:
    psm_df: pd.DataFrame = adata.var.copy()

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [executor.submit(_calculate_precursor_purity_worker, mzml_file, psm_df) for mzml_file in mzml_files]
        purities = [future.result() for future in concurrent.futures.as_completed(futures)]

    return pd.concat(purities)["purity"].astype(float)


def _calculate_precursor_purity_worker(mzml_file, psm_df):
    mzml_file_name: str = Path(mzml_file).name
    psm_df_by_file: pd.DataFrame = psm_df.loc[psm_df["filename"] == mzml_file_name]
    if psm_df_by_file.empty:
        return pd.DataFrame(columns=["purity"])

    mzml: MzmlReader = read_mzml(mzml_file)
    frame_df: pd.DataFrame = get_frame_df(mzml)

    purities_by_file: pd.DataFrame = _calculate_precursor_purity_by_file(mzml, frame_df, psm_df_by_file)
    return purities_by_file


def _calculate_precursor_purity_by_file(
    mzml: MzmlReader,
    frame_df: pd.DataFrame,
    psm_df_by_file: pd.DataFrame,
) -> pd.DataFrame:
    purity_colname = "purity"
    purity_df_by_file = pd.DataFrame(columns=[purity_colname])

    for row_idx, psm_row in psm_df_by_file.iterrows():
        ms2_scan_id: int = int(psm_row["scan_num"])
        charge = psm_row["charge"]
        pep_seq = psm_row["stripped_peptide"]
        modif_info = psm_row["modifications"]

        ms1_scan_id: int = frame_df.loc[frame_df["ID"].astype(int) == ms2_scan_id, "ms1_scan_id"].values[0]
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
    ms1_spectrum: MzmlReader,
    isotope_peaks: pd.DataFrame,
    isolation_window: tuple[float, float],
) -> float:
    if isotope_peaks is None:  # No isotope peak in ms1 spectrum
        purity: float = -1.0
    else:
        iso_min, iso_max = isolation_window
        isotype_peak_condition = (isotope_peaks.mz_array >= iso_min) & (isotope_peaks.mz_array <= iso_max)
        is_peak_in_index = np.where(isotype_peak_condition)[0]

        isotope_peaks_within_ab = isotope_peaks.ab_array[is_peak_in_index]
        isotope_peaks_sum = isotope_peaks_within_ab.sum()

        _, all_isolation_peaks_ab = ms1_spectrum.get_peaks_within(iso_min, iso_max)
        all_peaks_sum = all_isolation_peaks_ab.sum()

        if all_peaks_sum == 0:  # zero division # No isolation peak in isolation window
            purity: float = -2.0

        else:
            purity: float = (isotope_peaks_sum / all_peaks_sum).round(4)

    return purity
