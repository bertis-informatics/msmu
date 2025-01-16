from bmskit.composition.averagine import Averagine
from bmskit.composition.composition import Composition
from bmskit.sequence.sequence import Sequence
from bmskit.spectrometry.spectrum import Spectrum
from bmskit.spectrometry.tolerance import Tolerance


from pymzml.run import Reader as MzmlReader
from pymzml.file_classes import standardMzml as StandardMzml


def get_spectrum(mzml: MzmlReader, scan_id: int) -> StandardMzml:
    frame = mzml[scan_id]
    return Spectrum(mz_arr=frame.mz, ab_array=frame.i)


def get_composition(pep_seq):
    sequence = Sequence.from_string(pep_seq)
    composition: Composition = sequence.get_precursor_composition()

    return composition


def get_envelope(composition):
    return composition.get_isotopomer_envelope()


def get_peptide_mass(composition, modif_info):
    modif_mass = _get_modif_mass(modif_info)
    peptide_mass = float(composition.mass + modif_mass)

    return peptide_mass


def _get_modif_mass(modif_info: str) -> float:
    modif_mass = 0.0
    pattern = r"\(([\d.]+)\)"

    matches = re.findall(pattern, modif_info)
    for match in matches:
        modif_mass += float(match)

    return modif_mass


def get_isotope_peak(ms1_spectrum: dict, pep_mass, charge, envelope):
    tol = Tolerance(40)
    relative_intensity_threshold = 0.01

    isotope_peak = ms1_spectrum.get_all_isotope_peaks(pep_mass, charge, envelope, tol, relative_intensity_threshold)

    return isotope_peak
