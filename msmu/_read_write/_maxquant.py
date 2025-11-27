from pathlib import Path
import pandas as pd

from ._base_reader import SearchResultReader, SearchResultSettings
from . import label_info


class MaxQuantReader(SearchResultReader):
    """
    Reader for MaxQuant output files.
    Args:
        identification_file (str | Path): Path to the MaxQuant output file.
        label (Literal["tmt", "label_free"]): Label for the MaxQuant output ('tmt' or 'label_free').
    """

    def __init__(
        self,
        identification_file: str | Path,
    ) -> None:
        super().__init__()
        self.search_settings: SearchResultSettings = SearchResultSettings(
            search_engine="maxquant",
            quantification="maxquant",
            label=None,
            acquisition=None,
            identification_file=identification_file,
            identification_level="psm",
            quantification_file=None,
            quantification_level="psm",
            ident_quant_merged=True,
            has_decoy=True,
        )

        self.used_feature_cols.extend(
            [
                "missed_cleavages",
                "decoy",
                "contaminant",
                "score",
            ]
        )

        self._feature_rename_dict: dict = {
            "Sequence": "stripped_peptide",
            "Modified sequence": "peptide",
            "Length": "peptide_length",
            "Missed cleavages": "missed_cleavages",
            "Charge": "charge",
            "Raw file": "filename",
            "MSMS scan number": "scan_num",
            "Retention time": "rt",
            "hyperscore": "score",
        }

        self._cols_to_stringify: list[str] = [
            "Proteins",
            "Gene names",
            "Protein names",
            "Reverse",
            "Potential contaminant",
            "Taxonomy names",
        ]

    def _read_config_file(self):
        config = pd.read_csv(self.search_settings.config_path, sep="\t")
        config["Value"] = config["Value"].astype(str)
        return config

    def _read_identification_file(self) -> pd.DataFrame:
        tmp_sep = self._get_separator(self.search_settings.identification_path)
        identification_df = pd.read_csv(self.search_settings.identification_path, sep=tmp_sep)
        identification_df = identification_df.loc[~identification_df["Type"].isin(["MULTI-SECPEP"])]

        identification_df.columns = [x.replace("/", "") for x in identification_df.columns]
        prob_cols = [x for x in identification_df.columns if x.endswith("Probabilities")]
        score_diff_cols = [x for x in identification_df.columns if x.endswith("Score Diffs")]
        site_id_cols = [x for x in identification_df.columns if x.endswith("site IDs")]
        self._cols_to_stringify = self._cols_to_stringify + prob_cols + score_diff_cols + site_id_cols

        identification_df = self._stringify_cols(identification_df)

        return identification_df

    def _make_needed_columns_for_identification(self, identification_df: pd.DataFrame) -> pd.DataFrame:
        identification_df["decoy"] = identification_df["Reverse"].apply(lambda x: 1 if x == "+" else 0)
        identification_df["contaminant"] = identification_df["Potential contaminant"].apply(
            lambda x: 1 if x == "+" else 0
        )

        identification_df["proteins"] = identification_df["Proteins"]
        identification_df.loc[identification_df["decoy"] == 1, "proteins"] = identification_df.loc[
            identification_df["decoy"] == 1, "Leading proteins"
        ]
        identification_df["proteins"] = identification_df["proteins"].apply(lambda x: x.replace("REV__", "rev_"))
        identification_df["proteins"] = identification_df["proteins"].apply(lambda x: x.replace("CON__", "contam_"))

        return identification_df


class MaxTmtReader(MaxQuantReader):
    def __init__(
        self,
        search_dir: str | Path,
    ) -> None:
        super().__init__(search_dir)
        self.search_settings.label = "tmt"
        self.search_settings.acquisition = "dda"

    def _split_merged_feature_quantification(self, feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    def __init__(self, identification_file: str | Path, _quantification: bool = True) -> None:
        super().__init__(identification_file=identification_file)
        self.search_settings.label = "label_free"
        self.search_settings.quantification_level = "peptide" if _quantification else None
        self.search_settings.acquisition = "dda"

    def _make_peptide_quantification(self, split_identification_df: pd.DataFrame) -> pd.DataFrame:
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
    def __init__(self, search_dir):
        super().__init__(search_dir)
        self.search_settings.label = "label_free"
        self.search_settings.acquisition = "dia"
