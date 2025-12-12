from pathlib import Path
from typing import Literal
import pandas as pd

from ._base_reader import SearchResultReader, SearchResultSettings


class FragPipeReader(SearchResultReader):
    def __init__(
        self,
        identification_file: str | Path,
        quantification_file: str | Path | None,
        label: Literal["tmt", "label_free"] | None = None,
    ) -> None:
        super().__init__()
        self.search_settings: SearchResultSettings = SearchResultSettings(
            search_engine="fragpipe",
            quantification="fragpipe",
            label=label,
            acquisition="dda",
            identification_file=identification_file,
            identification_level="psm",
            quantification_file=quantification_file if quantification_file is not None else None,
            quantification_level=None,
            ident_quant_merged=True,
        )
        self._feature_rename_dict: dict = {
            "Charge": "charge",
            "Peptide Length": "peptide_length",
            "Number of Missed Cleavages": "missed_cleavages",
            "Peptide": "stripped_peptide",
            "Calculated Peptide Mass": "calcmass",
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

    @staticmethod
    def _label_decoy(label: int) -> int:
        if "rev_" in str(label):
            return 1
        else:
            return 0

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

        identification_df["decoy"] = identification_df["proteins"].apply(self._label_decoy)
        if identification_df["decoy"].unique().tolist() == [0]:
            self.search_settings.has_decoy = False

        identification_df["rt"] = identification_df["Retention"] / 60.0  # convert to minutes

        return identification_df


class TmtFragPipeReader(FragPipeReader):
    def __init__(self, identification_file: str | Path) -> None:
        super().__init__(identification_file=identification_file, label="tmt")
        self.search_settings.quantification_level = "psm"

    def _split_merged_identification_quantification(
        self, identification_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_identification_df = identification_df.copy()

        quant_cols = [x for x in identification_df.columns if x not in self.desc_cols]
        split_quant_df = split_identification_df[quant_cols]
        split_identification_df = split_identification_df.drop(columns=quant_cols)

        return split_identification_df, split_quant_df


class LfqFragPipeReader(FragPipeReader):
    def __init__(self, identification_file: str | Path, quantification_file: str | Path | None) -> None:
        super().__init__(identification_file, quantification_file)
        self.search_settings.label = "label_free"
        self.search_settings.ident_quant_merged = False

        self.used_feature_cols.extend(
            [
                "rt",
                "calcmass",
            ]
        )

        if quantification_file is not None:
            self.search_settings.quantification_level = "peptide"
        else:
            self.search_settings.quantification = None

    def _make_needed_columns_for_quantification(self, quantification_df: pd.DataFrame) -> pd.DataFrame:
        quantification_df = quantification_df.set_index("Modified Sequence", drop=True).rename_axis(index=None).copy()
        intensity_cols = [col for col in quantification_df.columns if col.endswith(" Intensity")]
        quantification_df = quantification_df[intensity_cols]

        return quantification_df

    def _make_rename_dict_for_obs(self, quantification_df):
        original_cols = quantification_df.columns.tolist()
        rename_dict = {col: col.removesuffix(" Intensity") for col in original_cols}

        return rename_dict
