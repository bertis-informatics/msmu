from pathlib import Path
from typing import Literal
import pandas as pd

from ._base_reader import SearchResultReader, SearchResultSettings, ReaderEngine
from . import _readers_pandas


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
        if self._engine is ReaderEngine.PANDAS:
            return _readers_pandas.fragpipe_identification(self, identification_df)
        return self._identification_columns_polars(identification_df)

    def _identification_columns_polars(self, identification_df) -> pd.DataFrame:
        """polars-native equivalent of the pandas feature build (same columns/values).

        proteins = (``Protein`` + ``Mapped Proteins``) tokens, whitespace-stripped, the literal
        ``"nan"`` (pandas ``astype(str)`` of a missing value) dropped, joined with ``;`` -- so a
        null column must first become ``"nan"`` (``fill_null("nan")``) to match. scan_num goes
        through int (``00123`` -> ``123``) exactly like the pandas ``int(...)``.
        """
        import polars as pl

        # Fail loud on a Spectrum from which no scan number can be parsed (null, or not the expected
        # "file.scan.scan.charge" shape), the way the old pandas int(x.split(".")[1]) did -- otherwise
        # the polars split nulls filename/scan, the index float-promotes, and features are silently
        # dropped at the ident/quant intersection.
        invalid_spectrum = identification_df.filter(
            pl.col("Spectrum").str.split(".").list.get(1, null_on_oob=True).cast(pl.Int64, strict=False).is_null()
        )
        if invalid_spectrum.height > 0:
            raise ValueError(
                "FragPipe Spectrum has no parseable scan number (expected 'file.scan.scan.charge'): "
                f"{invalid_spectrum['Spectrum'].head(3).to_list()}"
            )

        combined = (
            pl.col("Protein").cast(pl.Utf8).fill_null("nan")
            + pl.lit(",")
            + pl.col("Mapped Proteins").cast(pl.Utf8).fill_null("nan")
        )
        proteins_expr = (
            combined.str.split(",")
            .list.eval(pl.element().str.strip_chars())
            .list.eval(pl.element().filter(pl.element() != "nan"))
            .list.join(";")
        )
        out = identification_df.select(
            pl.col("Spectrum").str.split(".").list.get(0).alias("filename"),
            pl.col("Spectrum").str.split(".").list.get(1).cast(pl.Int64).alias("scan_num"),
            proteins_expr.alias("proteins"),
            pl.col("Modified Peptide").fill_null(pl.col("Peptide")).alias("peptide"),
            (pl.col("Retention") / 60.0).alias("rt"),  # convert to minutes
            pl.col("Peptide"),  # -> stripped_peptide
            pl.col("Charge"),  # -> charge
            pl.col("Peptide Length"),  # -> peptide_length
            pl.col("Number of Missed Cleavages"),  # -> missed_cleavages
            pl.col("Calculated Peptide Mass"),  # -> calcmass
            pl.col("observed mass"),  # -> expmass
            pl.col("Hyperscore"),  # -> score
        ).with_columns(
            pl.col("proteins").str.contains("rev_", literal=True).cast(pl.Int64).alias("decoy"),
        )
        if out.select(pl.col("decoy").sum()).item() == 0:
            self.search_settings.has_decoy = False
        return out


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

        if self._engine is ReaderEngine.PANDAS:
            return _readers_pandas.tmt_fragpipe_quant(self, raw_identification_df, quant_cols)

        import polars as pl

        # index == pandas _make_unique_index output: filename + "." + str(int(scan)) -- the
        # int() strips leading zeros ("00123" -> "123"), so cast through Int64 before Utf8.
        return (
            raw_identification_df.select(
                (
                    pl.col("Spectrum").str.split(".").list.get(0)
                    + pl.lit(".")
                    + pl.col("Spectrum").str.split(".").list.get(1).cast(pl.Int64).cast(pl.Utf8)
                ).alias("tmp_index"),
                *[pl.col(col) for col in quant_cols],
            )
            .to_pandas()
            .set_index("tmp_index")
            .rename_axis(index=None)
        )


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
        # Small peptide x sample table; convert to pandas on the polars path and share one tail
        # (both engines produce the same Modified-Sequence-indexed intensity columns).
        if self._engine is ReaderEngine.POLARS:
            quantification_df = quantification_df.to_pandas()
        quantification_df = quantification_df.set_index("Modified Sequence", drop=True).rename_axis(index=None).copy()
        intensity_cols = [col for col in quantification_df.columns if col.endswith(" Intensity")]
        quantification_df = quantification_df[intensity_cols]

        return quantification_df

    def _make_rename_dict_for_obs(self, quantification_df):
        original_cols = quantification_df.columns.tolist()
        rename_dict = {col: col.removesuffix(" Intensity") for col in original_cols}

        return rename_dict
