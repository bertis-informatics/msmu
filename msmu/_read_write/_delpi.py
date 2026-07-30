from pathlib import Path
import pandas as pd

from ._base_reader import SearchResultReader, SearchResultSettings
from .._utils.fasta import parse_uniprot_accession_group


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
        drop_search_result: bool = False,
    ) -> None:
        super().__init__(_drop_search_result=drop_search_result)
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

    def _extract_quant_from_raw(self, raw_identification_df: pd.DataFrame) -> pd.DataFrame:
        # Block-diagonal precursor quant: filename = run_name, index = "run_name.pmsm_index". The
        # index is built with the SAME astype(str) formatting as the identification frame's
        # _make_unique_index (a null pmsm_index -> "nan" after the to_pandas float-promotion), so
        # the two indexes intersect. Pull just the three columns to pandas and share this one tail --
        # a polars cast(Utf8) would format a null index differently and drop features at the
        # ident/quant intersection.
        raw_identification_df = raw_identification_df.select(["run_name", "pmsm_index", "ms2_area"]).to_pandas()

        quant_source = pd.DataFrame(
            {
                "filename": raw_identification_df["run_name"].to_numpy(),
                "ms2_area": raw_identification_df["ms2_area"].to_numpy(),
            },
            index=(
                raw_identification_df["run_name"].astype(str) + "." + raw_identification_df["pmsm_index"].astype(str)
            ).to_numpy(),
        )
        quant_df = quant_source.reset_index()
        quant_df = quant_df.pivot(index="index", columns="filename", values="ms2_area")
        quant_df = quant_df.rename_axis(index=None, columns=None)

        return quant_df

    def _make_needed_columns_for_identification(self, identification_df):
        # Build the feature frame on a FRESH DataFrame (identification columns only),
        # reading the raw frame read-only so it stays intact for varm (or to be
        # freed). Quantification (ms2_area) is taken from the raw frame by
        # _extract_quant_from_raw. ("rt" is computed by the legacy path but dropped
        # at the used_feature_cols subset, so it is omitted here.)
        return self._identification_columns_polars(identification_df)

    def _identification_columns_polars(self, identification_df) -> pd.DataFrame:
        """polars-native equivalent of the pandas feature build (same columns/values).

        The pandas peptide cleanup ``.strip("<").strip(">").strip(".").strip("_")`` strips each
        character *in sequence* (not as a set), so it is replicated as four single-char
        ``strip_chars`` calls -- ``strip_chars("<>._")`` would over-strip (e.g. ``._<AB>_.`` ->
        ``AB`` instead of ``<AB>``). Accession parsing is deduplicated via ``replace_strict``.
        """
        import polars as pl

        uniq = identification_df.select("fasta_id").unique().to_series().to_list()
        accession_map = {value: parse_uniprot_accession_group(value) for value in uniq}

        def _strip_delim(col):
            return (
                pl.col(col).str.strip_chars("<").str.strip_chars(">").str.strip_chars(".").str.strip_chars("_")
            )

        return identification_df.select(
            pl.col("fasta_id").replace_strict(accession_map).alias("proteins"),
            _strip_delim("peptide").alias("peptide"),  # -> stripped_peptide
            _strip_delim("modified_sequence").alias("modified_sequence"),  # -> peptide
            pl.col("run_name"),  # -> filename
            pl.col("frame_num"),  # -> scan_num
            pl.col("precursor_charge"),  # -> charge
            pl.col("sequence_length"),  # -> peptide_length
            pl.col("posterior_error"),  # -> PEP
            pl.col("global_precursor_q_value"),  # -> q_value
            pl.col("score"),
            pl.col("is_decoy"),  # -> decoy
            pl.col("pmsm_index"),  # for _make_unique_index (dropped at subset)
        ).to_pandas()
