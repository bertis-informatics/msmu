from pathlib import Path
import pandas as pd

from ._base_reader import (
    SearchResultReader,
    SearchResultSettings,
    SearchResultDataFrameConverter,
    _is_polars,
)
from . import label_info


class MaxQuantDataFrameConverter(SearchResultDataFrameConverter):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _read_file(file_path, as_polars: bool = False):
        file_path, identification_df = SearchResultDataFrameConverter._read_file(file_path, as_polars=as_polars)
        if _is_polars(identification_df):
            import polars as pl

            identification_df = identification_df.filter(~pl.col("Type").is_in(["MULTI-SECPEP"]))
        else:
            identification_df = identification_df.loc[~identification_df["Type"].isin(["MULTI-SECPEP"])]

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
        # Build the feature frame on a FRESH DataFrame (read the raw frame read-only so it
        # stays intact for varm or to be freed). Quantification is taken from the raw frame
        # by each subclass's _extract_quant_from_raw. Columns are carried under their raw
        # names and renamed by _normalise_identification_df via _feature_rename_dict.
        if _is_polars(identification_df):
            return self._identification_columns_polars(identification_df)
        feature_df = pd.DataFrame(index=identification_df.index)

        decoy = identification_df["Reverse"].apply(lambda x: 1 if x == "+" else 0)
        feature_df["decoy"] = decoy
        feature_df["contaminant"] = identification_df["Potential contaminant"].apply(lambda x: 1 if x == "+" else 0)

        proteins = identification_df["Proteins"].copy()
        proteins.loc[decoy == 1] = identification_df.loc[decoy == 1, "Leading proteins"]
        feature_df["proteins"] = proteins

        feature_df["Sequence"] = identification_df["Sequence"]  # -> stripped_peptide
        feature_df["Modified sequence"] = identification_df["Modified sequence"]  # -> peptide
        feature_df["Length"] = identification_df["Length"]  # -> peptide_length
        feature_df["Missed cleavages"] = identification_df["Missed cleavages"]  # -> missed_cleavages
        feature_df["Charge"] = identification_df["Charge"]  # -> charge
        feature_df["Raw file"] = identification_df["Raw file"]  # -> filename
        feature_df["MS/MS Scan Number"] = identification_df["MS/MS Scan Number"]  # -> scan_num
        feature_df["Retention time"] = identification_df["Retention time"]  # -> rt
        feature_df["PEP"] = identification_df["PEP"]

        return feature_df

    def _identification_columns_polars(self, identification_df) -> pd.DataFrame:
        """polars-native equivalent of the pandas feature build (same columns/values).

        ``Reverse``/``Potential contaminant`` are "+" or null; null must map to 0 (pandas
        ``x == "+"`` is False for NaN), hence ``fill_null(False)``. Decoy rows take
        ``Leading proteins`` instead of ``Proteins`` (the pandas conditional assignment).
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
        super().__init__(identification_file, identification_df)
        self.search_settings.label = "tmt"
        self.search_settings.acquisition = "dda"

    def _extract_quant_from_raw(self, raw_identification_df: pd.DataFrame) -> pd.DataFrame:
        # Same "Reporter intensity corrected *" channels as the old split, but sourced from
        # the raw frame and re-indexed via _make_unique_index (filename.scan_num) so it
        # aligns with the fresh feature frame.
        if _is_polars(raw_identification_df):
            import polars as pl

            reporter_cols = [c for c in raw_identification_df.columns if c.startswith("Reporter intensity corrected")]
            return (
                raw_identification_df.select(
                    (
                        pl.col("Raw file") + pl.lit(".") + pl.col("MS/MS Scan Number").cast(pl.Utf8)
                    ).alias("tmp_index"),
                    *[pl.col(col) for col in reporter_cols],
                )
                .to_pandas()
                .set_index("tmp_index")
                .rename_axis(index=None)
            )
        reporter_cols = [c for c in raw_identification_df.columns if c.startswith("Reporter intensity corrected")]
        quant = pd.DataFrame(index=raw_identification_df.index)
        quant["filename"] = raw_identification_df["Raw file"]
        quant["scan_num"] = raw_identification_df["MS/MS Scan Number"]
        for col in reporter_cols:
            quant[col] = raw_identification_df[col]
        quant = self._make_unique_index(quant)
        quant = quant.drop(columns=["filename", "scan_num"])

        return quant

    def _make_rename_dict_for_obs(self, quantification_df: pd.DataFrame) -> dict:
        plex = len(quantification_df.columns)
        tmt_labels = getattr(label_info, f"Tmt{plex}").label
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
        if _is_polars(raw_identification_df):
            import polars as pl

            pivoted = (
                raw_identification_df.group_by(["Modified sequence", "Raw file"])
                .agg(pl.col("Intensity").sum())
                .pivot(on="Raw file", index="Modified sequence", values="Intensity")
                .sort("Modified sequence")  # pandas pivot_table sorts the index
                .to_pandas()
                .set_index("Modified sequence")
                .rename_axis(index=None)
            )
            return pivoted.reindex(sorted(pivoted.columns), axis=1)  # match pandas column sort
        pep_quant_df = pd.DataFrame(
            {
                "filename": raw_identification_df["Raw file"].to_numpy(),
                "peptide": raw_identification_df["Modified sequence"].to_numpy(),
                "Intensity": raw_identification_df["Intensity"].to_numpy(),
            }
        )
        pep_quant_df = pep_quant_df.pivot_table(index="peptide", columns="filename", values="Intensity", aggfunc="sum")
        pep_quant_df = pep_quant_df.rename_axis(index=None, columns=None)

        return pep_quant_df


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
