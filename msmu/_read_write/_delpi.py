from pathlib import Path
import pandas as pd

from ._base_reader import SearchResultReader, SearchResultSettings
from .._utils.fasta import parse_uniprot_accession


class DelpiReader(SearchResultReader):
    """
    Reader for DELPI output files.

    Parameters:
        identification_file (str | Path): Path to the DELPI output file.
    """

    def __init__(
        self,
        identification_file: str | Path,
        identification_df: pd.DataFrame,
    ) -> None:
        super().__init__()
        self.search_settings: SearchResultSettings = SearchResultSettings(
            search_engine="delpi",
            quantification="delpi",
            label="label_free",
            acquisition="dia",
            identification_file=identification_file,
            identification_df=identification_df,
            identification_level="precursor",
            quantification_file=None,
            quantification_df=None,
            quantification_level="precursor",
            ident_quant_merged=True,
            has_decoy=True,
        )

        self._cols_to_stringify: list[str] = []

        self._feature_rename_dict = {
            "is_decoy": "decoy",
            "posterior_error": "PEP",
            "global_precursor_q_value": "q_value",
            "precursor_charge": "charge",
            "run_name": "filename",
            "frame_num": "scan_num",
            "peptide": "stripped_peptide",
            "modified_sequence": "peptide",
            "sequence_length": "peptide_length",
        }

        self.used_feature_cols.extend(["PEP", "q_value", "score"])

        # self.used_feature_cols.remove("scan_num")

    @staticmethod
    def _make_unique_index(input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()
        df["tmp_index"] = df["filename"].astype(str) + "." + df["pmsm_index"].astype(str)
        df = df.set_index("tmp_index", drop=True).rename_axis(index=None)

        return df

    def _split_merged_identification_quantification(
        self, identification_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_indentification_df = identification_df.copy()
        split_indentification_df = split_indentification_df.drop(columns=["ms2_area"])

        split_quantification_df = identification_df[["filename", "ms2_area"]].reset_index()
        split_quantification_df = split_quantification_df.pivot(index="index", columns="filename", values="ms2_area")
        split_quantification_df = split_quantification_df.rename_axis(index=None, columns=None)

        return split_indentification_df, split_quantification_df

    def _make_needed_columns_for_identification(self, identification_df):
        identification_df["rt"] = identification_df["observed_rt"] / 60.0  # convert to minutes
        identification_df["proteins"] = parse_uniprot_accession(identification_df["fasta_id"])
        identification_df["peptide"] = (
            identification_df["peptide"].str.strip("<").str.strip(">").str.strip(".").str.strip("_")
        )
        identification_df["modified_sequence"] = (
            identification_df["modified_sequence"].str.strip("<").str.strip(">").str.strip(".").str.strip("_")
        )

        return identification_df
