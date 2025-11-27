import os
from tqdm import tqdm
from pathlib import Path
from threading import Lock
import pandas as pd
import numpy as np

# import concurrent.futures

import anndata as ad
import mudata as md
import pyopenms as oms

from .._plotting._plots import plot_var
import plotly.graph_objects as go


class PurityResult:
    def __init__(self, purity: float, scan_num: int, filename: str):
        self.filename: list[str] = filename
        self.scan_num: list[int] = scan_num
        self.purity: list[float] = purity

    def __repr__(self) -> str:
        return f"PurityResult(purity={self.purity}, scan_num={self.scan_num}, filename='{self.filename}')"

    def to_df(self) -> pd.DataFrame:
        """
        Convert the PurityResult to a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing purity, scan_num, and filename.
        """
        return pd.DataFrame({"purity": self.purity, "scan_num": self.scan_num, "filename": self.filename})

    @property
    def _dummy_mdata(self) -> md.MuData:
        purity_df = self.to_df()
        purity_df.index = purity_df["filename"].str.strip(".mzML") + purity_df["scan_num"].astype(str)

        purity_adata = ad.AnnData(purity_df[["purity"]].T, var=purity_df)
        purity_mdata = md.MuData({"psm": purity_adata})

        purity_mdata["psm   "].uns["filter"] = {"mdata": {"filter_purity": np.nan}}

        return purity_mdata

    def hist(self) -> go.Figure:
        return plot_var(mdata=self._dummy_mdata, groupby="filename", ptype="hist", var_column="purity")

    def box(self) -> go.Figure:
        return plot_var(mdata=self._dummy_mdata, groupby="filename", ptype="box", var_column="purity")


class PrecursorPurityCalculator:
    """
    A class to calculate precursor isolation purity from mzML files or MuData objects.
    This class can be initialized with a tolerance value and whether to use ppm for the calculation.
    This class is a wrapper around the OpenMS PrecursorPurity class and provides methods to calculate.
    pyopenms: https://pyopenms.readthedocs.io/en/latest/py-modindex.html#module-pyopenms.PrecursorPurity

    Attributes:
        tolerance: Tolerance for precursor purity calculation.
        unit_ppm: Whether to use ppm for tolerance.
        mzml: Path to the mzML file.
        exp: OpenMS MSExperiment object loaded from the mzML file.
        exp_src: Source path of the loaded MSExperiment.
        exp_mtime: Last modified time of the mzML file.
        lock: Thread lock for thread-safe access to the MSExperiment.
    """

    def __init__(self, tolerance: float = 20, unit_ppm: bool = True):
        self._tolerance = tolerance
        self._unit_ppm = unit_ppm

        self._mzml: Path | None = None
        self._exp: oms.MSExperiment | None = None
        self._exp_src: Path | None = None
        self._exp_mtime: float | None = None
        self._lock: Lock = Lock()
        self._var_df: pd.DataFrame | None = None

    @classmethod
    def from_mudata(cls, mdata: md.MuData, tolerance: float = 20.0, unit_ppm: bool = True):
        """
        Initialize PrecursorPurityCalculator from a MuData object.

        Parameters:
            mdata: MuData object containing PSM data.
            tolerance: Tolerance for precursor purity calculation.
            unit_ppm: Whether to use ppm for tolerance.
        """
        instance = cls(tolerance=tolerance, unit_ppm=unit_ppm)
        if "psm" not in mdata.mod_names:
            raise ValueError("MuData object must contain 'psm' layer with PSM data.")

        if "filename" not in mdata["psm"].var.columns:
            raise ValueError("MuData object must contain 'filename' in the psm variable data.")
        if "scan_num" not in mdata["psm"].var.columns:
            raise ValueError("MuData object must contain 'scan_num' in the psm variable data.")

        instance._var_df = mdata["psm"].var.copy()

        return instance

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: float):
        if value <= 0:
            raise ValueError("Tolerance must be a positive number.")
        self._tolerance = value

    @property
    def mzml(self) -> Path | None:
        return self._mzml

    @mzml.setter
    def mzml(self, value: str | Path):
        if isinstance(value, str):
            value = Path(value)
        if not value.exists():
            raise FileNotFoundError(f"The specified mzML file does not exist: {value}")

        with self._lock:
            self._exp = None
            self._exp_src = None
            self._exp_mtime = None

        self._mzml = value

    @property
    def exp(self) -> oms.MSExperiment:
        with self._lock:
            mtime = os.path.getmtime(self._mzml)
            if (
                self._exp is None
                or self._exp_src != self._mzml
                or (self._exp_mtime is not None and self._exp_mtime != mtime)
            ):
                self._exp = self._import_mzml(self._mzml)
                self._exp_src = self._mzml
                self._exp_mtime = mtime

            return self._exp

    @property
    def ms2_indices(self) -> list[int]:
        if self._var_df is not None:
            ms2_indices = self._get_ms2_spectra_indices_from_mdata()
        else:
            ms2_indices = self._get_ms2_spectra_indices_from_mzml()

        return ms2_indices

    @property
    def ms2_scan_num(self) -> list[int]:
        return np.array(self.ms2_indices) + 1

    @staticmethod
    def _import_mzml(mzml_path: str | Path) -> oms.MSExperiment:
        exp: oms.MSExperiment = oms.MSExperiment()
        oms.MzMLFile().load(str(mzml_path), exp)

        return exp

    def _get_ms2_spectra_indices_from_mdata(self) -> list[int]:
        ms2_spectra: list[int] = (
            self._var_df.loc[self._var_df["filename"] == self.mzml.name.strip(".mzML"), "scan_num"].astype(int).tolist()
        )
        ms2_spectra = [i - 1 for i in ms2_spectra]
        return ms2_spectra

    def _get_ms2_spectra_indices_from_mzml(self) -> list[int]:
        ms2_spectra: list = []
        for i in range(0, self.exp.getNrSpectra()):
            if self.exp[i].getMSLevel() == 2:
                ms2_spectra.append(i)

        return ms2_spectra

    def _get_ms1_spectrum(self, ms2_index):
        is_ms1 = False
        i = ms2_index
        while is_ms1 is False:
            if self.exp[i].getMSLevel() == 1:
                is_ms1 = True
            else:
                i -= 1

        ms1_spectrum = self.exp[i]

        return ms1_spectrum

    def _calculate_precursor_isolation_purity(self, ms2_index) -> float:
        precursor = self.exp[ms2_index].getPrecursors()[0]

        ms1_spectrum = self._get_ms1_spectrum(ms2_index=ms2_index)

        purity_score = oms.PrecursorPurity().computePrecursorPurity(
            ms1_spectrum, precursor, self.tolerance, self._unit_ppm
        )

        return purity_score.signal_proportion

    def calculate_precursor_isolation_purities(self) -> pd.DataFrame:
        purities: pd.DataFrame = pd.DataFrame(
            {"scan_num": self.ms2_scan_num, "scan_index": self.ms2_indices, "filename": self.mzml.name}
        )
        purities["purity"] = purities["scan_index"].apply(self._calculate_precursor_isolation_purity)

        return purities[["filename", "scan_num", "purity"]].reset_index(drop=True)


def compute_precursor_isolation_purity_from_mzml(
    mzml_paths: str | Path | list, tolerance: float = 20.0, unit_ppm: bool = True
) -> PurityResult:
    """
    Calculate precursor isolation purity for all MS2 scans in the given mzML file.

    Parameters:
        mzml_paths: Full path(s) to the mzML file.
        tolerance: Tolerance for precursor purity calculation.
        unit_ppm: Whether to use ppm for tolerance.

    Returns:
        pd.DataFrame: DataFrame with scan numbers and their corresponding purity scores.
    """
    if isinstance(mzml_paths, (str, Path)):
        mzml_paths = [mzml_paths]
    if not isinstance(mzml_paths, list):
        raise TypeError("mzml_paths must be a string, Path, or list of strings/Paths.")

    calculator: PrecursorPurityCalculator = PrecursorPurityCalculator(tolerance=tolerance, unit_ppm=unit_ppm)
    purity_list: list = list()
    tqdm_iter = tqdm(mzml_paths, total=len(mzml_paths))
    for mzml_path in mzml_paths:
        tqdm_iter.set_description(f"Compute for {mzml_path}")
        if not isinstance(mzml_path, (str, Path)):
            raise TypeError("Each mzml_path must be a string or Path.")
        if not Path(mzml_path).exists():
            raise FileNotFoundError(f"The specified mzML file does not exist: {mzml_path}")

        calculator.mzml = mzml_path
        purities: pd.DataFrame = calculator.calculate_precursor_isolation_purities()
        purity_list.append(purities)

    purity_concatenated: pd.DataFrame = pd.concat(purity_list, ignore_index=True)
    purity_result: PurityResult = PurityResult(
        purity=purity_concatenated["purity"].to_list(),
        scan_num=purity_concatenated["scan_num"].tolist(),
        filename=purity_concatenated["filename"].tolist(),
    )

    return purity_result


def compute_precursor_isolation_purity(
    mdata: md.MuData, mzml_paths: str | Path | list, tolerance: float = 20.0, unit_ppm: bool = True
) -> md.MuData:
    """
    Calculate precursor isolation purity for PSMs in the given MuData object and mzML file.

    Parameters:
        mdata: MuData object containing PSM data.
        mzml_paths: Full path(s) to the mzML file.
        tolerance: Tolerance for precursor purity calculation. Default is 20.
        unit_ppm: Whether to use ppm for tolerance. Default is True.

    Returns:
        md.MuData: MuData object containing purity results.
    """

    if isinstance(mzml_paths, (str, Path)):
        mzml_paths: list = [mzml_paths]
    if not isinstance(mzml_paths, list):
        raise TypeError("mzml_paths must be a string, Path, or list of strings/Paths.")

    calculator: PrecursorPurityCalculator = PrecursorPurityCalculator.from_mudata(
        mdata, tolerance=tolerance, unit_ppm=unit_ppm
    )
    file_dict: dict = dict()
    for file in mdata["psm"].var["filename"].unique():
        full_mzml = [x for x in mzml_paths if Path(x).name == f"{file}.mzML"]
        if not full_mzml:
            raise ValueError(f"File {file} not found in provided mzML paths.")
        if len(full_mzml) > 1:
            raise ValueError(f"Multiple mzML files found for {file}. Please provide unique paths.")
        file_dict[file] = full_mzml[0]

    purity_list: list = list()
    tqdm_iter = tqdm(file_dict.items(), total=len(file_dict))
    for filename, full_ in tqdm_iter:
        tqdm_iter.set_description(f"Compute for {filename}")

        if not isinstance(full_, (str, Path)):
            raise TypeError("Each mzml_path must be a string or Path.")
        calculator.mzml = full_
        purities: pd.DataFrame = calculator.calculate_precursor_isolation_purities()
        purity_list.append(purities)

    purity_concatenated: pd.DataFrame = pd.concat(purity_list, ignore_index=True)
    purity_result: PurityResult = PurityResult(
        purity=purity_concatenated["purity"].to_list(),
        scan_num=purity_concatenated["scan_num"].tolist(),
        filename=purity_concatenated["filename"].tolist(),
    )

    purity_result_df = purity_result.to_df()
    purity_result_df["index"] = purity_result_df["filename"].astype(str).str.strip("mzML") + purity_result_df[
        "scan_num"
    ].astype(str)
    purity_result_df = purity_result_df.set_index("index", drop=True)
    purity_result_df = purity_result_df.rename_axis(None)
    purity_result_df["scan_num"] = purity_result_df["scan_num"].astype(int)

    purity_mdata = mdata.copy()
    purity_mdata["psm"].var["purity"] = purity_result_df["purity"]

    return purity_mdata
