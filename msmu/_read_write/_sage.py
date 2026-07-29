import json
from pathlib import Path

import pandas as pd

from ._base_reader import SearchResultReader, SearchResultSettings, ReaderEngine
from . import _readers_pandas
from .._utils.fasta import parse_uniprot_accession_group
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
        # Deduplicate the per-PSM string transforms: protein groups, scan strings and peptides
        # each have far fewer distinct values than PSM rows, so evaluate once per distinct value
        # and map back (self._map_unique) instead of once per row. Identical result, much faster.
        if self._engine is ReaderEngine.PANDAS:
            return _readers_pandas.sage_identification(self, identification_df)
        return self._identification_columns_polars(identification_df)

    def _identification_columns_polars(self, identification_df) -> pd.DataFrame:
        """polars-native equivalent of the pandas feature build (same columns/values/dtypes).

        Runs the whole identification transform as polars expressions and converts to pandas once
        (the AnnData boundary). Verified equivalences (polars has no lookbehind / list.eval is slow):
          stripped peptide  re ``([A-Z]+)|(\\[\\+\\d+\\.\\d+\\])`` join == ``str.replace_all("[^A-Z]", "")``
          accession parse   per-group fn via ``replace_strict`` over UNIQUE groups (dedup)
          filename strip    ``Path(x).name.rsplit(".",1)[0]`` == ``split("/").list.last().replace(r"\\.[^.]*$", "")``
        """
        import polars as pl

        # Fail loud on a scannr with no "scan=<number>" token (e.g. a non-Thermo instrument), the
        # way the old pandas int(...) did -- otherwise the polars extract nulls the scan, the index
        # float-promotes, and every feature is silently dropped at the ident/quant intersection.
        invalid_scannr = identification_df.filter(
            pl.col("scannr").is_not_null() & pl.col("scannr").str.extract(r"scan=(\d+)", 1).is_null()
        )
        if invalid_scannr.height > 0:
            raise ValueError(
                "Sage scannr has no 'scan=<number>' token (non-Thermo scan format is not supported): "
                f"{invalid_scannr['scannr'].head(3).to_list()}"
            )

        uniq = identification_df.select("proteins").unique().to_series().to_list()
        accession_map = {value: parse_uniprot_accession_group(value) for value in uniq}
        feature_df = identification_df.select(
            pl.col("proteins").replace_strict(accession_map).alias("proteins"),
            pl.col("peptide"),
            pl.col("filename").str.split("/").list.last().str.replace(r"\.[^.]*$", "").alias("filename"),
            pl.col("scannr").str.extract(r"scan=(\d+)", 1).cast(pl.Int64).alias("scan_num"),
            pl.col("peptide").str.replace_all(r"[^A-Z]", "").alias("stripped_peptide"),
            pl.col("charge"),
            pl.col("peptide_len"),
            pl.col("expmass"),
            pl.col("calcmass"),
            pl.col("rt"),
            pl.col("missed_cleavages"),
            pl.col("semi_enzymatic"),
            (pl.col("label") == -1).fill_null(False).cast(pl.Int64).alias("decoy"),  # null label -> target (0)
            (pl.lit(10.0) ** pl.col("posterior_error")).alias("PEP"),  # convert log10 PEP to PEP
            pl.col("hyperscore"),
            pl.col("spectrum_q"),
        ).with_columns(
            pl.col("proteins").str.contains("contam_", literal=True).cast(pl.Int64).alias("contaminant"),
        )
        return feature_df.to_pandas()


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
        if self._engine is ReaderEngine.PANDAS:
            return _readers_pandas.tmt_sage_quant(self, quantification_df)
        return self._quantification_columns_polars(quantification_df)

    def _quantification_columns_polars(self, quantification_df) -> pd.DataFrame:
        """polars-native TMT quant: build the ``filename.scan_num`` index and keep the tmt columns.

        Same result as the pandas path (index == identification's unique index) so the downstream
        obs rename / replace(0, NaN) / build_mudata are unchanged.
        """
        import polars as pl

        tmt_cols = [col for col in quantification_df.columns if col.startswith("tmt_")]
        return (
            quantification_df.select(
                (
                    pl.col("filename").str.split("/").list.last().str.replace(r"\.[^.]*$", "")
                    + pl.lit(".")
                    # cast through Int64 to strip leading zeros (scan=001001 -> 1001), matching the
                    # identification side's int cast so the two indexes intersect.
                    + pl.col("scannr").str.extract(r"scan=(\d+)", 1).cast(pl.Int64).cast(pl.Utf8)
                ).alias("tmp_index"),
                *[pl.col(col) for col in tmt_cols],
            )
            .to_pandas()
            .set_index("tmp_index")
            .rename_axis(index=None)
        )

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
        # The LFQ peptide-quant table is small (peptides x samples); on the polars-native path
        # convert it to pandas here and reuse the pandas transform -- the identification frame is
        # where the polars win lives, not this table.
        if self._engine is ReaderEngine.POLARS:
            quantification_df = quantification_df.to_pandas()
        quantification_df = quantification_df.set_index("peptide", drop=True).rename_axis(index=None).copy()
        quantification_df = quantification_df.drop(["charge", "proteins", "q_value", "score", "spectral_angle"], axis=1)

        return quantification_df

    def _make_rename_dict_for_obs(self, quantification_df) -> dict:
        original_cols = quantification_df.columns.tolist()

        return {col: self._strip_filename(col) for col in original_cols}
