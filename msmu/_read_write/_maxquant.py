from pathlib import Path
import pandas as pd

from ._base_reader import SearchResultReader, SearchResultSettings, SearchResultDataFrameConverter
from . import label_info


class MaxQuantDataFrameConverter(SearchResultDataFrameConverter):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _read_file(file_path):
        file_path, identification_df = SearchResultDataFrameConverter._read_file(file_path)
        identification_df = identification_df.loc[~identification_df["Type"].isin(["MULTI-SECPEP"])]

        return file_path, identification_df


class MaxQuantReader(SearchResultReader):
    """
    Reader for MaxQuant output files.
    Args:
        identification_file (str | Path): Path to the MaxQuant output file.
        label (Literal["tmt", "label_free"]): Label for the MaxQuant output ('tmt' or 'label_free').
    """

    def __init__(
        self, identification_file: str | Path, identification_df: pd.DataFrame, drop_search_result: bool = False
    ) -> None:
        super().__init__(_drop_search_result=drop_search_result)
        self.search_settings: SearchResultSettings = SearchResultSettings(
            search_engine="maxquant",
            quantification="maxquant",
            label=None,
            acquisition=None,
            identification_file=identification_file,
            identification_df=identification_df,
            identification_level="psm",
            quantification_file=None,
            quantification_df=None,
            quantification_level="psm",
            ident_quant_merged=True,
            has_decoy=True,
        )

        self.used_feature_cols.extend(
            [
                "rt",
                "missed_cleavages",
                "contaminant",
                "PEP",
            ]
        )

        self._feature_rename_dict: dict = {
            "Sequence": "stripped_peptide",
            "Modified sequence": "peptide",
            "Length": "peptide_length",
            "Missed cleavages": "missed_cleavages",
            "Charge": "charge",
            "Raw file": "filename",
            "MS/MS Scan Number": "scan_num",
            "Retention time": "rt",
        }

    def _make_needed_columns_for_identification(self, identification_df: pd.DataFrame) -> pd.DataFrame:
        identification_df["decoy"] = identification_df["Reverse"].apply(lambda x: 1 if x == "+" else 0)
        identification_df["contaminant"] = identification_df["Potential contaminant"].apply(
            lambda x: 1 if x == "+" else 0
        )

        identification_df["proteins"] = identification_df["Proteins"]
        identification_df.loc[identification_df["decoy"] == 1, "proteins"] = identification_df.loc[
            identification_df["decoy"] == 1, "Leading proteins"
        ]
        # identification_df["proteins"] = identification_df["proteins"].apply(lambda x: x.replace("REV__", "rev_"))
        # identification_df["proteins"] = identification_df["proteins"].apply(lambda x: x.replace("CON__", "contam_"))

        # identification_df = identification_df.loc[~identification_df["Type"].isin(["MULTI-SECPEP"])]

        return identification_df


class MaxTmtReader(MaxQuantReader):
    def __init__(
        self,
        identification_file: str | Path,
        identification_df: pd.DataFrame,
        drop_search_result: bool = False,
    ) -> None:
        super().__init__(identification_file, identification_df)
        self.search_settings.label = "tmt"
        self.search_settings.acquisition = "dda"

    def _split_merged_identification_quantification(
        self, feature_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_identification_df = feature_df.copy()

        quant_cols = [x for x in feature_df.columns if x.startswith("Reporter intensity corrected")]
        split_quant_df = split_identification_df[quant_cols]

        split_identification_df = split_identification_df.drop(columns=quant_cols)

        return split_identification_df, split_quant_df

    def _make_rename_dict_for_obs(self, quantification_df: pd.DataFrame) -> dict:
        plex = len(quantification_df.columns)
        tmt_labels = getattr(label_info, f"Tmt{plex}").label
        mq_labels = [f"Reporter intensity corrected {x}" for x in range(1, plex + 1)]

        channel_dict = {mq_col: tmt for mq_col, tmt in zip(mq_labels, tmt_labels)}

        return channel_dict


class MaxLfqReader(MaxQuantReader):
    def __init__(
        self,
        identification_file: str | Path,
        identification_df: pd.DataFrame,
        _quantification: bool = True,
        drop_search_result: bool = False,
    ) -> None:
        super().__init__(
            identification_file=identification_file,
            identification_df=identification_df,
            drop_search_result=drop_search_result,
        )
        self.search_settings.label = "label_free"
        self.search_settings.quantification_level = "peptide" if _quantification else None
        self.search_settings.acquisition = "dda"

    def _make_peptide_quantification(self, split_identification_df: pd.DataFrame) -> pd.DataFrame:
        # make quantification dataframe from identification dataframe by grouping by peptide
        # (summing intensities of PSMs with the same peptide sequence)
        pep_quant_df = split_identification_df[["filename", "peptide", "Intensity"]].copy()
        pep_quant_df = pep_quant_df.pivot_table(index="peptide", columns="filename", values="Intensity", aggfunc="sum")
        pep_quant_df = pep_quant_df.rename_axis(index=None, columns=None)

        return pep_quant_df

    def _split_merged_identification_quantification(
        self, identification_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_identification_df = identification_df.copy()
        split_identification_df = split_identification_df.drop(columns=["Intensity"])

        split_quant_df = identification_df[["filename", "peptide", "Intensity"]].reset_index()
        split_quant_df = self._make_peptide_quantification(split_quant_df)

        return split_identification_df, split_quant_df


class MaxDiaReader(MaxQuantReader):
    def __init__(
        self, identification_file: str | Path, identification_df: pd.DataFrame, drop_search_result: bool = False
    ):
        super().__init__(identification_file, identification_df, drop_search_result=drop_search_result)
        self.search_settings.label = "label_free"
        self.search_settings.acquisition = "dia"
