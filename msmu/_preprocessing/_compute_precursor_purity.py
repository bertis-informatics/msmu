from pathlib import Path
from tqdm import tqdm
import pandas as pd
import mudata as md

from .._tools._precursor_purity import PrecursorPurityCalculator, PurityResult

def compute_precursor_purity(
    mdata: md.MuData, 
    mzml_paths: str | Path | list, 
    tolerance: float = 20.0, 
    unit_ppm: bool = True
    ) -> md.MuData:
    """
    Calculate precursor isolation purity for PSMs in the given MuData object and mzML file.

    Parameters:
        mdata (md.MuData): MuData object containing PSM data.
        mzml_paths (str | Path | list): Full path(s) to the mzML file.
        tolerance (float): Tolerance for precursor purity calculation. Default is 20.
        unit_ppm (bool): Whether to use ppm for tolerance. Default is True.

    Returns:
        md.MuData: MuData object containing purity results.
    """

    if isinstance(mzml_paths, (str, Path)):
        mzml_paths:list = [mzml_paths]
    if not isinstance(mzml_paths, list):
        raise TypeError("mzml_paths must be a string, Path, or list of strings/Paths.")
    
    calculator:PrecursorPurityCalculator = PrecursorPurityCalculator.from_mudata(
        mdata, tolerance=tolerance, unit_ppm=unit_ppm
        )
    file_dict:dict = dict()
    for file in mdata["feature"].var["filename"].unique():
        full_mzml = [x for x in mzml_paths if Path(x).name == file]
        if not full_mzml:
            raise ValueError(f"File {file} not found in provided mzML paths.")
        if len(full_mzml) > 1:
            raise ValueError(f"Multiple mzML files found for {file}. Please provide unique paths.")
        file_dict[file] = full_mzml[0]

    purity_list:list = list()
    tqdm_iter = tqdm(file_dict.items(), total=len(file_dict))
    for filename, full_ in tqdm_iter:
        tqdm_iter.set_description(f"Compute for {filename}")

        if not isinstance(full_, (str, Path)):
            raise TypeError("Each mzml_path must be a string or Path.")
        calculator.mzml = full_
        purities:pd.DataFrame = calculator.calculate_precursor_isolation_purities()
        purity_list.append(purities)

    purity_concatenated:pd.DataFrame = pd.concat(purity_list, ignore_index=True)
    purity_result:PurityResult = PurityResult(
        purity=purity_concatenated["purity"].to_list(),
        scan_num=purity_concatenated["scan_num"].tolist(),
        filename=purity_concatenated["filename"].tolist()
    )

    purity_result_df = purity_result.to_df()
    purity_result_df["index"] = purity_result_df["filename"].str.strip("mzML") + purity_result_df["scan_num"].astype(str)
    purity_result_df = purity_result_df.set_index("index", drop=True)
    purity_result_df = purity_result_df.rename_axis(None)
    purity_result_df["scan_num"] = purity_result_df["scan_num"].astype(int)

    purity_mdata = mdata.copy()
    purity_mdata["feature"].var["purity"] = purity_result_df["purity"]

    return purity_mdata