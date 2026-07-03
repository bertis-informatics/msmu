from pathlib import Path
from typing import Literal
import pandas as pd

from ._base_reader import SearchResultReader, SearchResultSettings


class FragPipeReader(SearchResultReader):
    def __init__(
        self,
        identification_file: str | Path,
        identification_df: pd.DataFrame,
        drop_search_result: bool = False,
        quantification_file: str | Path | None = None,
        quantification_df: pd.DataFrame | None = None,
        label: Literal["tmt", "label_free"] | None = None,
    ) -> None:
        super().__init__(_drop_search_result=drop_search_result)
        self.search_settings: SearchResultSettings = SearchResultSettings(
            search_engine="fragpipe",
            quantification="fragpipe",
            label=label,
            acquisition="dda",
            identification_file=identification_file,
            identification_df=identification_df,
            identification_level="psm",
            quantification_file=quantification_file if quantification_file is not None else None,
            quantification_df=quantification_df if quantification_df is not None else None,
            quantification_level=None,
            ident_quant_merged=True,
        )
        self._feature_rename_dict: dict = {
            "Charge": "charge",
            "Peptide Length": "peptide_length",
            "Number of Missed Cleavages": "missed_cleavages",
            "Peptide": "stripped_peptide",
            "Calculated Peptide Mass": "calcmass",
            "observed mass": "expmass",
            "Hyperscore": "score",
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
                "rt",
                "calcmass",
                "expmass",
                "missed_cleavages",
                "decoy",
                "score",
            ]
        )

    @staticmethod
    def _label_decoy(label: int) -> int:
        if "rev_" in str(label):
            return 1
        else:
            return 0

    def _make_needed_columns_for_identification(self, identification_df: pd.DataFrame) -> pd.DataFrame:
        # Build the feature frame on a FRESH DataFrame (read the raw frame read-only so it
        # stays intact for varm or to be freed). Quantification (TMT channels) is taken from
        # the raw frame by TmtFragPipeReader._extract_quant_from_raw. Carry-through columns
        # keep their raw names and are renamed by _normalise_identification_df.
        feature_df = pd.DataFrame(index=identification_df.index)

        feature_df["filename"] = identification_df["Spectrum"].apply(lambda x: x.split(".")[0])
        feature_df["scan_num"] = identification_df["Spectrum"].apply(lambda x: int(x.split(".")[1]))

        proteins = identification_df["Protein"].astype(str) + "," + identification_df["Mapped Proteins"].astype(str)
        proteins = proteins.apply(lambda x: [y.strip() for y in x.split(",") if y != "nan"])
        proteins = proteins.apply(lambda x: ",".join(x))
        proteins = proteins.apply(lambda x: x.replace(",", ";"))
        feature_df["proteins"] = proteins

        peptide = identification_df["Modified Peptide"].copy()
        peptide.loc[peptide.isna()] = identification_df.loc[peptide.isna(), "Peptide"]
        feature_df["peptide"] = peptide

        feature_df["decoy"] = feature_df["proteins"].apply(self._label_decoy)
        if feature_df["decoy"].unique().tolist() == [0]:
            self.search_settings.has_decoy = False

        feature_df["rt"] = identification_df["Retention"] / 60.0  # convert to minutes

        feature_df["Peptide"] = identification_df["Peptide"]  # -> stripped_peptide
        feature_df["Charge"] = identification_df["Charge"]  # -> charge
        feature_df["Peptide Length"] = identification_df["Peptide Length"]  # -> peptide_length
        feature_df["Number of Missed Cleavages"] = identification_df["Number of Missed Cleavages"]  # -> missed_cleavages
        feature_df["Calculated Peptide Mass"] = identification_df["Calculated Peptide Mass"]  # -> calcmass
        feature_df["observed mass"] = identification_df["observed mass"]  # -> expmass
        feature_df["Hyperscore"] = identification_df["Hyperscore"]  # -> score

        return feature_df


class TmtFragPipeReader(FragPipeReader):
    def __init__(
            self, 
            identification_file: str | Path, 
            identification_df: pd.DataFrame,
            drop_search_result: bool = False,
            ) -> None:
        super().__init__(
            identification_file=identification_file,
            identification_df=identification_df,
            label="tmt",
            drop_search_result=drop_search_result,
        )
        self.search_settings.quantification_level = "psm"

    def _extract_quant_from_raw(self, raw_identification_df: pd.DataFrame) -> pd.DataFrame:
        # Real TMT channels only: raw columns that are neither FragPipe descriptors
        # (desc_cols) nor identification columns the reader renames (_feature_rename_dict
        # keys, e.g. "observed mass"). Sourcing from the raw frame -- where those columns
        # still carry their descriptor names -- avoids the old split's leak of renamed
        # identification cols (rt/calcmass/expmass/score) into quant. Re-indexed via
        # _make_unique_index (filename.scan_num) to align with the fresh feature frame.
        non_quant_cols = set(self.desc_cols) | set(self._feature_rename_dict.keys())
        quant_cols = [c for c in raw_identification_df.columns if c not in non_quant_cols]

        quant = pd.DataFrame(index=raw_identification_df.index)
        quant["filename"] = raw_identification_df["Spectrum"].apply(lambda x: x.split(".")[0])
        quant["scan_num"] = raw_identification_df["Spectrum"].apply(lambda x: int(x.split(".")[1]))
        for col in quant_cols:
            quant[col] = raw_identification_df[col]
        quant = self._make_unique_index(quant)
        quant = quant.drop(columns=["filename", "scan_num"])

        return quant


class LfqFragPipeReader(FragPipeReader):
    def __init__(
        self,
        identification_file: str | Path,
        identification_df: pd.DataFrame,
        drop_search_result: bool = False,
        quantification_file: str | Path | None = None,
        quantification_df: pd.DataFrame | None = None,
    ) -> None:
        super().__init__(
            identification_file=identification_file,
            identification_df=identification_df,
            quantification_file=quantification_file,
            quantification_df=quantification_df,
            drop_search_result=drop_search_result,
        )
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
