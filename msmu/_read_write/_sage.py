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
        quantification_file: str | Path | None,
    ) -> None:
        super().__init__()
        self.search_settings: SearchResultSettings = SearchResultSettings(
            search_engine="sage",
            quantification="sage",
            label=None,
            acquisition="dda",
            identification_file=identification_file,
            identification_level="psm",
            quantification_file=quantification_file if quantification_file is not None else None,
            quantification_level=None,
            ident_quant_merged=False,
            has_decoy=True,
        )
        self._feature_rename_dict: dict = {"peptide_len": "peptide_length", "spectrum_q": "q_value"}
        self.used_feature_cols.extend(
            [
                "missed_cleavages",
                "semi_enzymatic",
                "decoy",
                "contaminant",
                "PEP",
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
        identification_df["proteins"] = parse_uniprot_accession(identification_df["proteins"])
        identification_df["filename"] = identification_df["filename"].apply(self._strip_filename)
        identification_df["scan_num"] = identification_df["scannr"].apply(self._extract_scan_number)
        identification_df["stripped_peptide"] = identification_df["peptide"].apply(self._make_stripped_peptide)
        identification_df["decoy"] = identification_df["label"].apply(self._label_decoy)
        identification_df["contaminant"] = identification_df["proteins"].apply(self._label_possible_contaminant)
        identification_df["PEP"] = np.power(10, identification_df["posterior_error"])  # convert log10 PEP to PEP

        return identification_df.copy()


class TmtSageReader(SageReader):
    """
    Reader for TMT-labeled Sage output files.

    Parameters:
        search_dir: Path to the Sage output directory.
    """

    def __init__(
        self,
        identification_file: str | Path,
        quantification_file: str | Path | None,
    ) -> None:
        super().__init__(identification_file, quantification_file)
        self.search_settings.label = "tmt"
        self.search_settings.quantification_level = "psm"

    def _make_needed_columns_for_quantification(self, quantification_df: pd.DataFrame) -> pd.DataFrame:
        quantification_df["filename"] = quantification_df["filename"].apply(self._strip_filename)
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
        quantification_file: str | Path | None,
    ) -> None:
        super().__init__(identification_file, quantification_file)
        self.search_settings.label = "label_free"
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
        quantification_df = quantification_df.set_index("peptide", drop=True).rename_axis(index=None).copy()
        quantification_df = quantification_df.drop(["charge", "proteins", "q_value", "score", "spectral_angle"], axis=1)

        return quantification_df

    def _make_rename_dict_for_obs(self, quantification_df) -> dict:
        original_cols = quantification_df.columns.tolist()

        return {col: self._strip_filename(col) for col in original_cols}
