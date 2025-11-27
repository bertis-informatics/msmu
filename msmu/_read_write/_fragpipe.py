from pathlib import Path
from typing import Literal
import pandas as pd
import numpy as np

from ._base_reader import SearchResultReader, SearchResultSettings

# from . import label_info


class FragPipeReader(SearchResultReader):
    def __init__(self, identification_file: str | Path, label: Literal["tmt", "label_free"] | None = None) -> None:
        super().__init__()
        self.search_settings: SearchResultSettings = SearchResultSettings(
            search_engine="fragpipe",
            quantification="fragpipe",
            label=label,
            acquisition="dda",
            identification_file=identification_file,
            identification_level="psm",
            quantification_file=None,
            quantification_level=None,
            ident_quant_merged=True,
        )
        self._feature_rename_dict: dict = {
            "Charge": "charge",
            "Peptide Length": "peptide_length",
            "Number of Missed Cleavages": "missed_cleavages",
            "Peptide": "stripped_peptide",
        }

        self.desc_cols = [
            "Spectrum",
            "Spectrum File",
            "Peptide",
            "Modified Peptide",
            "Extended Peptide",
            "Prev AA",
            "Next AA",
            "Peptide Length",
            "Charge",
            "Retention",
            "Observed Mass",
            "Calibrated Observed Mass",
            "Observed M/Z",
            "Calibrated Observed M/Z",
            "Calculated Peptide Mass",
            "Calculated M/Z",
            "Delta Mass",
            "Expectation",
            "Hyperscore",
            "Nextscore",
            "PeptideProphet Probability",
            "Number of Enzymatic Termini",
            "Number of Missed Cleavages",
            "Protein Start",
            "Protein End",
            "Intensity",
            "Assigned Modifications",
            "Observed Modifications",
            "Compensation Voltage",
            "Purity",
            "Is Unique",
            "Protein",
            "Protein ID",
            "Entry Name",
            "Gene",
            "Protein Description",
            "Mapped Genes",
            "Mapped Proteins",
            "Quan Usage",
            "stripped_peptide",
            "peptide_length",
            "missed_cleavages",
            "charge",
            "decoy",
            "filename",
            "scan_num",
            "proteins",
            "peptide",
        ]

        self.used_feature_cols.extend(
            [
                "missed_cleavages",
                "decoy",
            ]
        )

    def _make_needed_columns_for_identification(self, identification_df: pd.DataFrame) -> pd.DataFrame:
        identification_df["filename"] = identification_df["Spectrum"].apply(lambda x: x.split(".")[0])
        identification_df["scan_num"] = identification_df["Spectrum"].apply(lambda x: int(x.split(".")[1]))

        identification_df["proteins"] = (
            identification_df["Protein"].astype(str) + "," + identification_df["Mapped Proteins"].astype(str)
        )
        identification_df["proteins"] = identification_df["proteins"].apply(
            lambda x: [y.strip() for y in x.split(",") if y != "nan"]
        )
        identification_df["proteins"] = identification_df["proteins"].apply(lambda x: ",".join(x))
        identification_df["proteins"] = identification_df["proteins"].apply(lambda x: x.replace(",", ";"))

        identification_df["peptide"] = identification_df["Modified Peptide"]
        identification_df.loc[identification_df["peptide"].isna(), "peptide"] = identification_df.loc[
            identification_df["peptide"].isna(), "Peptide"
        ]

        identification_df["decoy"] = identification_df["proteins"].apply(lambda x: 1 if "rev_" in str(x) else 0)

        return identification_df


class TmtFragPipeReader(FragPipeReader):
    def __init__(self, search_dir: str | Path) -> None:
        super().__init__(search_dir, label="tmt")
        self.search_settings.quantification_level = "psm"

    def _split_merged_identification_quantification(
        self, identification_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_identification_df = identification_df.copy()

        quant_cols = [x for x in identification_df.columns if x not in self.desc_cols]
        split_quant_df = split_identification_df[quant_cols]
        split_identification_df = split_identification_df.drop(columns=quant_cols)

        return split_identification_df, split_quant_df


class LfqFragPipeReader(FragPipeReader): ...
