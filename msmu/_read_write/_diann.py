from pathlib import Path
import pandas as pd
import numpy as np
import scipy.sparse as sp

from ._base_reader import SearchResultReader, SearchResultSettings, _is_polars
from .._core._blockdiag import SparseQuantFrame
from .._utils.fasta import parse_uniprot_accession, parse_uniprot_accession_group


class DiannReader(SearchResultReader):
    """
    Reader for DIA-NN output files.

    Parameters:
        identification_file (str | Path): Path to the DIA-NN output directory.
    """

    def __init__(
        self,
        identification_file: str | Path,
        identification_df: pd.DataFrame,
        drop_search_result: bool = False,
        sparse: bool = False,
    ) -> None:
        super().__init__(_drop_search_result=drop_search_result)
        self._sparse = sparse
        self.search_settings: SearchResultSettings = SearchResultSettings(
            search_engine="diann",
            quantification="diann",
            label="label_free",
            acquisition="dia",
            identification_file=identification_file,
            identification_df=identification_df,
            identification_level="precursor",
            quantification_file=None,
            quantification_df=None,
            quantification_level="precursor",
            ident_quant_merged=True,
            has_decoy=False,
        )

        self.used_feature_cols.extend(
            [
                "rt",
                "decoy",
                "PEP",
                "q_value",
            ]
        )

        self.used_feature_cols.remove("scan_num")

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
            "Precursor.Charge": "charge",
            "Decoy": "decoy",
            f"{q_value_prefix}.Q.Value": "q_value",
            "RT": "rt",
            # "Precursor.Mass": "calcmass",
        }

        return rename_dict

    @staticmethod
    def _make_unique_index(input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()
        df["tmp_index"] = df["filename"] + "." + df["Precursor.Id"]
        df = df.set_index("tmp_index", drop=True).rename_axis(index=None)

        return df

    def _split_merged_identification_quantification(
        self, identification_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_identification_df = identification_df.copy()
        split_identification_df = split_identification_df.drop(columns=["Precursor.Quantity"])

        split_quant_df = identification_df[["filename", "Precursor.Quantity"]].reset_index()
        split_quant_df = split_quant_df.pivot(index="index", columns="filename", values="Precursor.Quantity")
        split_quant_df = split_quant_df.rename_axis(index=None, columns=None)
        split_quant_df = split_quant_df.replace(0, np.nan)

        return split_identification_df, split_quant_df

    def _extract_quant_from_raw(self, raw_identification_df: pd.DataFrame):
        # DIA-NN's precursor quant is block-diagonal: the feature id "Run.Precursor.Id" encodes
        # the run, so each precursor feature carries a value in exactly one run column and is NaN
        # in every other -- a (n_precursor_obs x n_run) matrix with ~one non-null per row. The
        # dense pivot below materialises all of it (0.5% filled); the sparse path builds only the
        # observed cells as a COO/CSR, avoiding the dense pivot entirely.
        if _is_polars(raw_identification_df):
            # polars-native path: pull only the three columns this needs, as pandas, then reuse the
            # exact same COO/pivot logic below (bit-identical to the pandas read path).
            raw_identification_df = raw_identification_df.select(
                ["Run", "Precursor.Id", "Precursor.Quantity"]
            ).to_pandas()
        features = (raw_identification_df["Run"] + "." + raw_identification_df["Precursor.Id"]).to_numpy()
        runs = raw_identification_df["Run"].to_numpy()
        values = raw_identification_df["Precursor.Quantity"].to_numpy(dtype=float)

        if not self._sparse:
            quant_source = pd.DataFrame(
                {"filename": runs, "Precursor.Quantity": values},
                index=features,
            )
            quant_df = quant_source.reset_index()
            quant_df = quant_df.pivot(index="index", columns="filename", values="Precursor.Quantity")
            quant_df = quant_df.rename_axis(index=None, columns=None)
            quant_df = quant_df.replace(0, np.nan)
            return quant_df

        # Sparse: each row is a distinct feature; place its value in its run column. 0/NaN are
        # treated as absent (matching the dense path's replace(0, np.nan)).
        sample_index, run_codes = np.unique(runs, return_inverse=True)
        observed = np.isfinite(values) & (values != 0)
        row = np.arange(len(features))[observed]
        col = run_codes[observed]
        matrix = sp.coo_matrix(
            (values[observed], (row, col)),
            shape=(len(features), len(sample_index)),
            dtype=np.float32,
        ).tocsr()
        return SparseQuantFrame(matrix=matrix, index=pd.Index(features), columns=pd.Index(sample_index))

    def _make_needed_columns_for_identification(self, identification_df: pd.DataFrame) -> pd.DataFrame:
        # Build the feature frame on a FRESH DataFrame (identification columns only),
        # reading the raw frame read-only so it stays intact for varm (or to be
        # freed). Quantification is taken from the raw frame by _extract_quant_from_raw.
        if _is_polars(identification_df):
            return self._identification_columns_polars(identification_df)
        self._set_mbr(identification_df)  # sets self._mbr (selects the q-value column to rename)
        self._set_decoy(identification_df)

        # object columns of the raw frame -> stringified into varm when kept
        for col in identification_df.columns:
            if identification_df[col].dtype == "object":
                self._cols_to_stringify.append(col)

        feature_df = pd.DataFrame(index=identification_df.index)
        feature_df["proteins"] = parse_uniprot_accession(identification_df["Protein.Ids"])
        feature_df["missed_cleavages"] = identification_df["Stripped.Sequence"].str.count(r"(?<=[KR])(?!P)")
        feature_df["peptide_length"] = identification_df["Stripped.Sequence"].str.len()
        feature_df["PEP"] = identification_df["PEP"]
        feature_df["Modified.Sequence"] = identification_df["Modified.Sequence"]  # -> peptide
        feature_df["Stripped.Sequence"] = identification_df["Stripped.Sequence"]  # -> stripped_peptide
        feature_df["Run"] = identification_df["Run"]  # -> filename
        feature_df["Precursor.Charge"] = identification_df["Precursor.Charge"]  # -> charge
        feature_df["RT"] = identification_df["RT"]  # -> rt
        feature_df["Precursor.Id"] = identification_df["Precursor.Id"]  # for _make_unique_index (dropped at subset)
        q_value_source = "Lib.Q.Value" if self._mbr else "Global.Q.Value"
        feature_df[q_value_source] = identification_df[q_value_source]  # -> q_value
        if self.search_settings.has_decoy:
            feature_df["Decoy"] = identification_df["Decoy"]  # -> decoy
        else:
            feature_df["decoy"] = 0

        return feature_df

    def _identification_columns_polars(self, identification_df) -> pd.DataFrame:
        """polars-native equivalent of the pandas feature build (same columns/values/dtypes).

        polars has no lookbehind, so the tryptic missed-cleavage count ``(?<=[KR])(?!P)`` is the
        equivalent ``count("[KR]") - count("[KR]P")``; accession parsing is deduplicated via
        ``replace_strict`` over unique protein-id strings. Converts to pandas once at the end.
        """
        import polars as pl

        # mbr / decoy flags (normally set by _set_mbr / _set_decoy) computed on the polars frame
        self._mbr = identification_df.select(pl.col("Lib.Q.Value").sum()).item() != 0
        self.search_settings.has_decoy = ("Decoy" in identification_df.columns) and bool(
            identification_df.select(pl.col("Decoy").cast(pl.Boolean).any()).item()
        )
        q_value_source = "Lib.Q.Value" if self._mbr else "Global.Q.Value"

        uniq = identification_df.select("Protein.Ids").unique().to_series().to_list()
        accession_map = {value: parse_uniprot_accession_group(value) for value in uniq}
        exprs = [
            pl.col("Protein.Ids").replace_strict(accession_map).alias("proteins"),
            (
                pl.col("Stripped.Sequence").str.count_matches("[KR]")
                - pl.col("Stripped.Sequence").str.count_matches("[KR]P")
            ).cast(pl.Int64).alias("missed_cleavages"),
            pl.col("Stripped.Sequence").str.len_chars().cast(pl.Int64).alias("peptide_length"),
            pl.col("PEP"),
            pl.col("Modified.Sequence"),  # -> peptide
            pl.col("Stripped.Sequence"),  # -> stripped_peptide
            pl.col("Run"),  # -> filename
            pl.col("Precursor.Charge"),  # -> charge
            pl.col("RT"),  # -> rt
            pl.col("Precursor.Id"),  # for _make_unique_index (dropped at subset)
            pl.col(q_value_source),  # -> q_value
        ]
        if self.search_settings.has_decoy:
            exprs.append(pl.col("Decoy"))  # -> decoy
        feature_df = identification_df.select(exprs).to_pandas()
        if not self.search_settings.has_decoy:
            feature_df["decoy"] = 0
        return feature_df

    def _set_mbr(self, identification_df: pd.DataFrame) -> None:
        if identification_df["Lib.Q.Value"].sum() == 0:
            self._mbr = False
        else:
            self._mbr = True

    def _set_decoy(self, identification_df: pd.DataFrame) -> None:
        self.search_settings.has_decoy = False
        if "Decoy" in (identification_df.columns):
            if identification_df["Decoy"].any():
                self.search_settings.has_decoy = True


class DiannProteinGroupReader(SearchResultReader):
    def __init__(self, search_dir: str | Path) -> None:
        super().__init__()
        raise NotImplementedError("DIA-NN protein group reader is not implemented yet.")
