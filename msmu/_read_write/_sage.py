import json
from pathlib import Path

import pandas as pd
import numpy as np

from ._base_reader import SearchResultReader, SearchResultSettings
from .._utils.fasta import parse_uniprot_accession
from . import label_info


class SageReader(SearchResultReader):
    """
    Reader for Sage output files.

    Parameters:
        identification_file: Path to the Sage output directory.
        quantification_file: Path to the quantification file (if applicable).
    """

    def __init__(
        self,
        identification_file: str | Path,
        drop_search_result: bool = False,
        identification_df: pd.DataFrame | None = None,
        quantification_file: str | Path | None = None,
        quantification_df: pd.DataFrame | None = None,
    ) -> None:
        super().__init__(_drop_search_result=drop_search_result)
        self.search_settings: SearchResultSettings = SearchResultSettings(
            search_engine="sage",
            quantification="sage",
            label=None,
            acquisition="dda",
            identification_file=identification_file,
            identification_df=identification_df,
            identification_level="psm",
            quantification_file=quantification_file if quantification_file is not None else None,
            quantification_df=quantification_df,
            quantification_level=None,
            ident_quant_merged=False,
            has_decoy=True,
        )
        self._feature_rename_dict: dict = {
            "peptide_len": "peptide_length",
            "spectrum_q": "q_value",
            "hyperscore": "score",
        }
        self.used_feature_cols.extend(
            [
                "expmass",
                "calcmass",
                "rt",
                "missed_cleavages",
                "semi_enzymatic",
                "contaminant",
                "PEP",
                "score",
                "q_value",
            ]
        )

    @staticmethod
    def _label_decoy(label: int) -> int:
        if label == -1:
            return 1
        else:
            return 0

    @staticmethod
    def _label_possible_contaminant(proteins: str) -> int:
        if "contam_" in proteins:
            return 1
        else:
            return 0

    @staticmethod
    def _extract_scan_number(scan_str: str) -> int:
        return int(scan_str.split("scan=")[1])

    def _read_config_file(self):
        with open(self.search_settings.config_path, "r") as f:
            config = json.load(f)
        return config

    def _make_needed_columns_for_identification(self, identification_df: pd.DataFrame) -> pd.DataFrame:
        # Build the feature frame on a FRESH DataFrame rather than mutating the input,
        # so the caller keeps the raw frame intact (to serve varm or be freed) without
        # a defensive copy. Column operations mirror the previous in-place version.
        feature_df = pd.DataFrame(index=identification_df.index)
        feature_df["proteins"] = parse_uniprot_accession(identification_df["proteins"])
        feature_df["peptide"] = identification_df["peptide"]
        feature_df["filename"] = self._map_unique(identification_df["filename"], self._strip_filename)
        feature_df["scan_num"] = identification_df["scannr"].apply(self._extract_scan_number)
        feature_df["stripped_peptide"] = identification_df["peptide"].apply(self._make_stripped_peptide)
        feature_df["charge"] = identification_df["charge"]
        feature_df["peptide_len"] = identification_df["peptide_len"]
        feature_df["expmass"] = identification_df["expmass"]
        feature_df["calcmass"] = identification_df["calcmass"]
        feature_df["rt"] = identification_df["rt"]
        feature_df["missed_cleavages"] = identification_df["missed_cleavages"]
        feature_df["semi_enzymatic"] = identification_df["semi_enzymatic"]
        feature_df["decoy"] = (identification_df["label"] == -1).astype(int)
        feature_df["contaminant"] = feature_df["proteins"].str.contains("contam_", regex=False).astype(int)
        feature_df["PEP"] = np.power(10, identification_df["posterior_error"])  # convert log10 PEP to PEP
        feature_df["hyperscore"] = identification_df["hyperscore"]
        feature_df["spectrum_q"] = identification_df["spectrum_q"]

        return feature_df


class TmtSageReader(SageReader):
    """
    Reader for TMT-labeled Sage output files.

    Parameters:
        search_dir: Path to the Sage output directory.
    """

    def __init__(
        self,
        identification_file: str | Path,
        drop_search_result: bool = False,
        identification_df: pd.DataFrame | None = None,
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
        self.search_settings.label = "tmt"
        self.search_settings.quantification_level = "psm"

    def _make_needed_columns_for_quantification(self, quantification_df: pd.DataFrame) -> pd.DataFrame:
        quantification_df["filename"] = self._map_unique(quantification_df["filename"], self._strip_filename)
        quantification_df["scan_num"] = quantification_df["scannr"].apply(self._extract_scan_number)
        quantification_df = self._make_unique_index(quantification_df)
        quantification_df = quantification_df.drop(["filename", "scannr", "scan_num", "ion_injection_time"], axis=1)

        return quantification_df

    def _make_rename_dict_for_obs(self, quantification_df: pd.DataFrame) -> dict:
        plex = len(quantification_df.columns)
        tmt_labels = getattr(label_info, f"Tmt{plex}").label
        sage_labels = [f"tmt_{x}" for x in range(1, plex + 1)]

        channel_dict = {sage_col: tmt for sage_col, tmt in zip(sage_labels, tmt_labels)}

        return channel_dict


class LfqSageReader(SageReader):
    """
    Reader for label-free Sage output files.

    Parameters:
        identification_file: Path to the Sage output directory.
        quantification_file: Path to the quantification file (if applicable).
    """

    def __init__(
        self,
        identification_file: str | Path,
        drop_search_result: bool = False,
        identification_df: pd.DataFrame | None = None,
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

        if quantification_file is not None:
            self.search_settings.quantification_level = "peptide"
        else:
            self.search_settings.quantification = None

    def _make_needed_columns_for_quantification(self, quantification_df: pd.DataFrame) -> pd.DataFrame:
        quantification_df = quantification_df.set_index("peptide", drop=True).rename_axis(index=None).copy()
        quantification_df = quantification_df.drop(["charge", "proteins", "q_value", "score", "spectral_angle"], axis=1)

        return quantification_df

    def _make_rename_dict_for_obs(self, quantification_df) -> dict:
        original_cols = quantification_df.columns.tolist()

        return {col: self._strip_filename(col) for col in original_cols}
