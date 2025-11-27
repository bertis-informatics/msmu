from pathlib import Path
import pandas as pd
import numpy as np

from ._base_reader import SearchResultReader, SearchResultSettings
from .._utils.fasta import parse_uniprot_accession


class DiannReader(SearchResultReader):
    """
    Reader for DIA-NN output files.

    Parameters:
        search_dir (str | Path): Path to the DIA-NN output directory.
    """

    def __init__(
        self,
        evidence_file: str | Path,
    ) -> None:
        super().__init__()
        self.search_settings: SearchResultSettings = SearchResultSettings(
            search_engine="diann",
            quantification="diann",
            label="label_free",
            acquisition="dia",
            evidence_file=Path(evidence_file),
            evidence_level="precursor",
            quantification_file=None,
            quantification_level="precursor",
            feat_quant_merged=True,
            has_decoy=False,
        )

        self.used_feature_cols.extend(
            [
                "PEP",
                "q_value",
            ]
        )

        self._cols_to_stringify: list[str] = [
            "Protein.Names",
            "Protein.Group",
            "Genes",
            "Genes.Quantity",
            "Genes.Normalised",
            "Genes.MaxLFQ",
            "Genes.MaxLFQ.Unique",
            "First.Protein.Description",
        ]

        self._mbr: bool | None = None

    @property
    def _feature_rename_dict(self):
        if self._mbr:
            q_value_prefix = "Lib"
        else:
            q_value_prefix = "Global"

        rename_dict = {
            "Protein.Group": "protein_group",
            "Modified.Sequence": "peptide",
            "Stripped.Sequence": "stripped_peptide",
            "Run": "filename",
            "MS2.Scan": "scan_num",
            "Precursor.Charge": "charge",
            "Decoy": "decoy",
            f"{q_value_prefix}.Q.Value": "q_value",
        }

        return rename_dict

    @staticmethod
    def _make_unique_index(input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()
        df["tmp_index"] = df["filename"] + "." + df["scan_num"].astype(str) + "." + df["Precursor.Id"]
        df = df.set_index("tmp_index", drop=True).rename_axis(index=None)

        return df

    def _split_merged_evidence_quantification(self, evidence_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_evidence_df = evidence_df.copy()
        split_evidence_df = split_evidence_df.drop(columns=["Precursor.Quantity"])

        split_quant_df = evidence_df[["filename", "Precursor.Quantity"]].reset_index()
        split_quant_df = split_quant_df.pivot(index="index", columns="filename", values="Precursor.Quantity")
        split_quant_df = split_quant_df.rename_axis(index=None, columns=None)
        split_quant_df = split_quant_df.replace(0, np.nan)

        return split_evidence_df, split_quant_df

    def _make_needed_columns_for_evidence(self, evidence_df: pd.DataFrame) -> pd.DataFrame:
        evidence_df = evidence_df.copy()
        self._set_mbr(evidence_df)  # set self._mbr for _feature_rename_dict
        self._set_decoy(evidence_df)

        evidence_df["proteins"] = evidence_df["Protein.Ids"]
        evidence_df["proteins"] = parse_uniprot_accession(evidence_df["proteins"])
        evidence_df["missed_cleavages"] = evidence_df["Stripped.Sequence"].apply(self._count_missed_cleavages)
        evidence_df["peptide_length"] = evidence_df["Stripped.Sequence"].apply(self._get_peptide_length)
        if not self.search_settings.has_decoy:
            evidence_df["decoy"] = 0

        return evidence_df

    def _set_mbr(self, evidence_df: pd.DataFrame) -> None:
        if evidence_df["Lib.Q.Value"].sum() == 0:
            self._mbr = False
        else:
            self._mbr = True

    def _set_decoy(self, evidence_df: pd.DataFrame) -> None:
        if "Decoy" in evidence_df.columns:
            self.search_settings.has_decoy = True
        else:
            self.search_settings.has_decoy = False


class DiannProteinGroupReader(SearchResultReader):
    def __init__(self, search_dir: str | Path) -> None:
        super().__init__()
        raise NotImplementedError("DIA-NN protein group reader is not implemented yet.")
