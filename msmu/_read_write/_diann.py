from pathlib import Path
import pandas as pd
import numpy as np

from ._base_reader import SearchResultReader, SearchResultSettings

class DiannReader(SearchResultReader):
    """
    Reader for DIA-NN output files.
    Args:
        search_dir (str | Path): Path to the DIA-NN output directory.
    """
    def __init__(
        self,
        search_dir: str | Path,
    ) -> None:
        super().__init__()
        self.search_settings: SearchResultSettings = SearchResultSettings(
            search_engine="diann",
            quantification="diann",
            label="label_free",
            acquisition="dia",
            output_dir=Path(search_dir).absolute(),
            feature_file="report.tsv",
            feature_level="precursor",
            quantification_file=None,
            quantification_level="precursor",
            config_file=None,
            feat_quant_merged=True,
        )

        self.used_feature_cols.extend([
            # "protein_group",
            "spectrum_q",
            "protein_q",
        ])

        self._cols_to_stringify:list[str] = [
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
        self._decoy: bool | None = None

    @property
    def _feature_rename_dict(self):
        if self._mbr:
            q_value_prefix = "Lib"
        else:
            q_value_prefix = "Global"

        rename_dict = {
            "Protein.Ids": "proteins",
            "Protein.Group": "protein_group",
            "Modified.Sequence": "peptide",
            "Stripped.Sequence": "stripped_peptide",
            "Run": "filename",
            "MS2.Scan": "scan_num",
            "Precursor.Charge": "charge",
            "Decoy": "decoy",
            f"{q_value_prefix}.Q.Value": "spectrum_q",
            f"{q_value_prefix}.PG.Q.Value": "protein_q",
        }
        
        return rename_dict
    
    @staticmethod
    def _make_unique_index(input_df:pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()
        df['tmp_index'] = df["filename"] + "." + df["scan_num"].astype(str) + "." + df["Precursor.Id"]
        df = df.set_index("tmp_index", drop=True).rename_axis(index=None)

        return df

    def _split_merged_feature_quantification(self, feature_df:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_feature_df = feature_df.copy()
        split_feature_df = split_feature_df.drop(columns=["Precursor.Quantity"])

        split_quant_df = feature_df[["filename", "Precursor.Quantity"]].reset_index()
        split_quant_df = split_quant_df.pivot(
            index="index",
            columns="filename",
            values="Precursor.Quantity"
        )
        split_quant_df = split_quant_df.rename_axis(index=None, columns=None)
        split_quant_df = split_quant_df.replace(0, np.nan)

        return split_feature_df, split_quant_df

    def _make_needed_columns_for_feature(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        feature_df = feature_df.copy()
        self._set_mbr(feature_df) # set self._mbr for _feature_rename_dict
        self._set_decoy(feature_df)

        feature_df["missed_cleavages"] = feature_df["Stripped.Sequence"].apply(self._count_missed_cleavages)
        feature_df["peptide_length"] = feature_df["Stripped.Sequence"].apply(self._get_peptide_length)

        if not self._decoy:
            feature_df["decoy"] = 0

        return feature_df

    def _set_mbr(self, feature_df: pd.DataFrame) -> None:
        if feature_df["Lib.Q.Value"].sum() == 0:
            self._mbr = False
        else:
            self._mbr = True
    
    def _set_decoy(self, feature_df: pd.DataFrame) -> None:
        if "Decoy" in feature_df.columns:
            self._decoy = True
        else:
            self._decoy = False


class DiannProteinGroupReader(SearchResultReader):
    def __init__(self, search_dir: str | Path) -> None:
        super().__init__()
        raise NotImplementedError("DIA-NN protein group reader is not implemented yet.")
