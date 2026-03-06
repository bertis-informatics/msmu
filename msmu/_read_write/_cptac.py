from pathlib import Path
import pandas as pd
import numpy as np

from ._base_reader import SearchResultReader, SearchResultSettings, SearchResultDataFrameConverter
from ._sage import SageReader
from . import label_info


class CPTACDataFrameConverter(SearchResultDataFrameConverter):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _read_file(file_path):
        from pyteomics import mzid

        identification_df = pd.DataFrame(mzid.DataFrame(str(file_path)))
        filename = file_path.stem
        identification_df["filename"] = filename

        return file_path, identification_df


class CPTACReader(SearchResultReader):
    """
    Reader for CPTAC output files.

    Parameters:
        identification_file: Path to the CPTAC output file (mzid format).
    """

    def __init__(
        self,
        identification_file: list[str | Path | None],
        identification_df: pd.DataFrame,
        _drop_search_result: bool = True,
    ) -> None:
        super().__init__(_drop_search_result=_drop_search_result)
        self.search_settings: SearchResultSettings = SearchResultSettings(
            search_engine="mgfplus",
            quantification=None,
            label=None,
            acquisition="dda",
            identification_file=identification_file,
            identification_df=identification_df,
            identification_level="psm",
            quantification_file=None,
            quantification_df=None,
            quantification_level=None,
            ident_quant_merged=False,
            has_decoy=True,
        )
        self._feature_rename_dict: dict = {
            "MS-GF:QValue": "q_value",
            "MS-GF:EValue": "PEP",
            "retention time": "rt",
            "chargeState": "charge",
            "PeptideSequence": "stripped_peptide",
            "MS-GF:RawScore": "score",
            "CPTAC-CDAP:PrecursorArea": "precursor_area",
            "MS-GF:DeNovoScore": "de_novo_score",
        }
        self.used_feature_cols.extend(
            [
                "calcmass",
                "expmass",
                "PEP",
                "rt",
                "score",
                "q_value",
            ]
        )

    @staticmethod
    def _label_decoy(label: int) -> int:
        if label == False:
            return 0
        else:
            return 1

    def _make_unique_index(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()
        df["tmp_index"] = df["filename"] + "." + df["scan_num"].astype(str)
        df = df.set_index("tmp_index", drop=True).rename_axis(index=None)

        return df

    def _make_peptide(self, stripped_peptide: str, modifications: list[dict]) -> str:
        if modifications == "":
            return stripped_peptide
        else:
            splited_peptide = [""] + [x for x in stripped_peptide]
            modification_positions = ["" for _ in range(len(splited_peptide))]
            for mod in modifications:
                mod_pos = mod["location"]
                mod_mass = mod["monoisotopicMassDelta"]

                modification_positions[mod_pos] = f"[+{mod_mass}]-" if mod_pos == 0 else f"[+{mod_mass}]"

            modified_peptide = "".join(f"{aa}{mod}" for aa, mod in zip(splited_peptide, modification_positions))
            return modified_peptide

    def _make_needed_columns_for_identification(self, identification_df):
        identification_df["scan_num"] = identification_df["spectrumID"].apply(SageReader._extract_scan_number)
        identification_df["decoy"] = identification_df["isDecoy"].apply(self._label_decoy)
        identification_df["peptide"] = identification_df.apply(
            lambda row: self._make_peptide(row["PeptideSequence"], row["Modification"]), axis=1
        )
        identification_df["peptide_length"] = identification_df["PeptideSequence"].apply(len)
        identification_df["proteins"] = identification_df["accession"].apply(";".join)

        identification_df["calcmass"] = (
            identification_df["calculatedMassToCharge"] * identification_df["chargeState"]
            - identification_df["chargeState"] * 1.007276466621
        )
        identification_df["expmass"] = (
            identification_df["experimentalMassToCharge"] * identification_df["chargeState"]
            - identification_df["chargeState"] * 1.007276466621
        )

        return identification_df


class TmtCPTACReader(CPTACReader):
    """
    Reader for CPTAC output files with TMT quantification.

    Parameters:
        identification_file: Path to the CPTAC output file (mzid format).
        identification_df: DataFrame containing the identification data.
    """

    def __init__(
        self,
        identification_file: str | Path | None,
        identification_df: pd.DataFrame,
        _drop_search_result: bool = True,
    ) -> None:
        super().__init__(identification_file, identification_df, _drop_search_result=_drop_search_result)
        self.search_settings.quantification = "ReAdW4Mascot2"
        self.search_settings.quantification_level = "psm"
        self.search_settings.label = "tmt"
        self.search_settings.ident_quant_merged = True

    def _make_rename_dict_for_obs(self, quantification_df: pd.DataFrame) -> dict:
        plex_len = len(quantification_df.columns)
        # need validation with TMTXX format

        tmt_labels = getattr(label_info, f"Tmt{plex_len}").label
        quant_cols = quantification_df.columns
        quant_col_dict = {quant_col: tmt for quant_col, tmt in zip(quant_cols, tmt_labels)}

        return quant_col_dict

    def _split_merged_identification_quantification(
        self, identification_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_identification_df = identification_df.copy()

        quant_cols = [
            x
            for x in identification_df.columns
            if ("TMT" in x)
            and (x.endswith("Flags") == False)
            and (x.endswith("FractionOfTotalAb") == False)
            and (x.endswith("TotalAb") == False)
        ]
        split_quant_df = split_identification_df[quant_cols]

        split_identification_df = split_identification_df.drop(columns=quant_cols)

        cols = split_quant_df.columns
        split_quant_df[cols] = (
            split_quant_df[cols].stack().str.split("/").str[0].unstack().astype(float).replace(0, np.nan)
        )

        return split_identification_df, split_quant_df


class LfqCPTACReader(CPTACReader):
    """
    Reader for CPTAC output files with label-free quantification.

    Parameters:
        identification_file: Path to the CPTAC output file (mzid format).
    """

    def __init__(
        self,
        identification_file: str | Path | None,
        identification_df: pd.DataFrame,
        _drop_search_result: bool = True,
    ) -> None:
        super().__init__(identification_file, identification_df, _drop_search_result=_drop_search_result)
        self.search_settings.label = "label_free"
        self.search_settings.ident_quant_merged = False
