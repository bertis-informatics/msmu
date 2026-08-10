from pathlib import Path
import pandas as pd

from ._base_reader import (
    SearchResultReader,
    SearchResultSettings,
    SearchResultDataFrameConverter,
)
from . import label_info


class MaxQuantDataFrameConverter(SearchResultDataFrameConverter):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _read_file(file_path):
        import polars as pl

        # The base reader is polars-only (files -> native polars, a DataFrame input -> from_pandas).
        # fill_null(False) keeps a row whose Type is null: polars ~is_in(...) yields null for a null
        # Type and filter would drop it, where pandas ~isin keeps NaN rows.
        file_path, identification_df = SearchResultDataFrameConverter._read_file(file_path)
        identification_df = identification_df.filter(~pl.col("Type").is_in(["MULTI-SECPEP"]).fill_null(False))

        return file_path, identification_df


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
        identification_df: pd.DataFrame,
        drop_search_result: bool = False,
    ) -> None:
        super().__init__(_drop_search_result=drop_search_result)
        self.search_settings: SearchResultSettings = SearchResultSettings(
            search_engine="maxquant",
            quantification="maxquant",
            label=None,
            acquisition=None,
            identification_file=identification_file,
            identification_df=identification_df,
            identification_level="psm",
            quantification_file=None,
            quantification_df=None,
            quantification_level="psm",
            ident_quant_merged=True,
            has_decoy=True,
        )

        self.used_feature_cols.extend(
            [
                "rt",
                "missed_cleavages",
                "contaminant",
                "PEP",
            ]
        )

        self._feature_rename_dict: dict = {
            "Sequence": "stripped_peptide",
            "Modified sequence": "peptide",
            "Length": "peptide_length",
            "Missed cleavages": "missed_cleavages",
            "Charge": "charge",
            "Raw file": "filename",
            "MS/MS Scan Number": "scan_num",
            "Retention time": "rt",
        }

    def _make_needed_columns_for_identification(self, identification_df: pd.DataFrame) -> pd.DataFrame:
        """Build the feature frame on a FRESH DataFrame with polars expressions (read the raw frame
        read-only so it stays intact for varm or to be freed), converting to pandas once at the
        AnnData boundary. Quantification is taken from the raw frame by each subclass's
        _extract_quant_from_raw; columns are carried under their raw names and renamed by
        _normalise_identification_df via _feature_rename_dict.

        ``Reverse``/``Potential contaminant`` are "+" or null; null must map to 0 (``x == "+"`` is
        False for a null), hence ``fill_null(False)``. Decoy rows take ``Leading proteins`` instead
        of ``Proteins`` (a conditional assignment).
        """
        import polars as pl

        is_decoy = (pl.col("Reverse") == "+").fill_null(False)
        return identification_df.select(
            is_decoy.cast(pl.Int64).alias("decoy"),
            (pl.col("Potential contaminant") == "+").fill_null(False).cast(pl.Int64).alias("contaminant"),
            pl.when(is_decoy).then(pl.col("Leading proteins")).otherwise(pl.col("Proteins")).alias("proteins"),
            pl.col("Sequence"),  # -> stripped_peptide
            pl.col("Modified sequence"),  # -> peptide
            pl.col("Length"),  # -> peptide_length
            pl.col("Missed cleavages"),  # -> missed_cleavages
            pl.col("Charge"),  # -> charge
            pl.col("Raw file"),  # -> filename
            pl.col("MS/MS Scan Number"),  # -> scan_num
            pl.col("Retention time"),  # -> rt
            pl.col("PEP"),
        ).to_pandas()


class MaxTmtReader(MaxQuantReader):
    def __init__(
        self,
        identification_file: str | Path,
        identification_df: pd.DataFrame,
        drop_search_result: bool = False,
    ) -> None:
        super().__init__(identification_file, identification_df, drop_search_result=drop_search_result)
        self.search_settings.label = "tmt"
        self.search_settings.acquisition = "dda"

    def _extract_quant_from_raw(self, raw_identification_df: pd.DataFrame) -> pd.DataFrame:
        # Same "Reporter intensity corrected *" channels as the old split, but sourced from
        # the raw frame and re-indexed via _make_unique_index (filename.scan_num) so it
        # aligns with the fresh feature frame.
        import polars as pl

        reporter_cols = [c for c in raw_identification_df.columns if c.startswith("Reporter intensity corrected")]
        # Build the index through the SAME _make_unique_index the identification frame uses
        # (filename + "." + scan_num.astype(str), post-to_pandas) rather than a polars
        # cast(Utf8). A null scan coerces the identification frame's scan_num to float
        # ("123.0") via to_pandas, so a polars cast(Utf8) here ("123") would not match and
        # those features would be silently dropped at the ident/quant index intersection.
        quant = raw_identification_df.select(
            pl.col("Raw file").alias("filename"),
            pl.col("MS/MS Scan Number").alias("scan_num"),
            *[pl.col(col) for col in reporter_cols],
        ).to_pandas()
        quant = self._make_unique_index(quant)
        quant = quant.drop(columns=["filename", "scan_num"])

        return quant

    def _make_rename_dict_for_obs(self, quantification_df: pd.DataFrame) -> dict:
        plex = len(quantification_df.columns)
        tmt_labels = [label_info.to_sdrf_channel_label(reporter) for reporter in getattr(label_info, f"Tmt{plex}").label]
        mq_labels = [f"Reporter intensity corrected {x}" for x in range(1, plex + 1)]

        channel_dict = {mq_col: tmt for mq_col, tmt in zip(mq_labels, tmt_labels)}

        return channel_dict


class MaxLfqReader(MaxQuantReader):
    def __init__(
        self,
        identification_file: str | Path,
        identification_df: pd.DataFrame,
        _quantification: bool = True,
        drop_search_result: bool = False,
    ) -> None:
        super().__init__(
            identification_file=identification_file,
            identification_df=identification_df,
            drop_search_result=drop_search_result,
        )
        self.search_settings.label = "label_free"
        self.search_settings.quantification_level = "peptide" if _quantification else None
        self.search_settings.acquisition = "dda"

    def _extract_quant_from_raw(self, raw_identification_df: pd.DataFrame) -> pd.DataFrame:
        # Peptide-level quantification: sum PSM "Intensity" per (peptide, filename) and pivot
        # to peptide x filename. Sourced from the raw frame -- "Modified sequence"/"Raw file"
        # are the raw names of the feature frame's "peptide"/"filename", so the result matches
        # the old split (which pivoted the normalised frame).
        import polars as pl

        pivoted = (
            # Drop null group keys before the pivot: pandas pivot_table silently drops rows whose
            # index/column key is NaN, where polars group_by+pivot would keep a phantom "null" sample.
            raw_identification_df.filter(
                pl.col("Raw file").is_not_null() & pl.col("Modified sequence").is_not_null()
            )
            .group_by(["Modified sequence", "Raw file"])
            .agg(pl.col("Intensity").sum())
            .pivot(on="Raw file", index="Modified sequence", values="Intensity")
            .sort("Modified sequence")  # pandas pivot_table sorts the index
            .to_pandas()
            .set_index("Modified sequence")
            .rename_axis(index=None)
        )
        return pivoted.reindex(sorted(pivoted.columns), axis=1)  # match pandas column sort


class MaxDiaReader(MaxQuantReader):
    def __init__(
        self,
        identification_file: str | Path,
        identification_df: pd.DataFrame,
        drop_search_result: bool = False,
    ):
        super().__init__(
            identification_file,
            identification_df,
            drop_search_result=drop_search_result,
        )
        self.search_settings.label = "label_free"
        self.search_settings.acquisition = "dia"
