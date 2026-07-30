from pathlib import Path
from typing import Literal
from dataclasses import dataclass
from typing import Callable

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

# from pyteomics import mzid

from ..logging_utils import get_logger
from .._core._blockdiag import SparseQuantFrame
from .._utils.peptide import (
    _calc_exp_mz,
    _count_missed_cleavages,
    _get_peptide_length,
    _make_stripped_peptide,
)


logger = get_logger(__name__)

# The readers are polars-only: csv/tsv/parquet are parsed with polars (multi-threaded), the
# identification frame stays polars through the reader transforms, and it converts to pandas once at
# the AnnData boundary. Every reader (Sage, DIA-NN, MaxQuant, FragPipe, DELPI) implements the polars
# transforms. The converter coerces a DataFrame passed directly (the ``read_*`` DataFrame-input API)
# to polars at the read boundary (see ``_read_file``), so the transforms always receive polars.


def set_polars_reader(enabled: bool = True) -> None:
    """Deprecated no-op: the readers are polars-only, so there is no pandas path to toggle.

    Kept for backwards compatibility (it is part of the public API). Calling it emits a
    ``DeprecationWarning`` and does nothing.
    """
    import warnings

    warnings.warn(
        "set_polars_reader() is deprecated and has no effect: the readers are polars-only.",
        DeprecationWarning,
        stacklevel=2,
    )


# pandas' default NA sentinels (pandas.io.parsers.STR_NA_VALUES), hard-coded to avoid a private
# import. polars only treats an empty field as null by default, so without this a "NA"/"#N/A"/"null"
# in a numeric column keeps the whole column as String (silently -> string values land in var, or a
# later numeric op crashes) instead of pandas' float-with-NaN. Passed to pl.read_csv so the two
# engines agree on which tokens are missing.
_PANDAS_NA_VALUES: list[str] = [
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
]


def _polars_read_native(file_path, suffix: str):
    """Read csv/tsv/parquet with polars and return a polars DataFrame (no pandas conversion).

    ``infer_schema_length=None`` scans every row for dtype inference (matching pandas' whole-file
    scan) so a column that only widens past the first 100 rows -- e.g. integer scans then a float,
    or a numeric column with a late non-numeric token -- does not hard-fail. ``null_values`` aligns
    the missing-token set with pandas (see ``_PANDAS_NA_VALUES``).
    """
    import polars as pl

    if suffix == ".csv":
        return pl.read_csv(file_path, infer_schema_length=None, null_values=_PANDAS_NA_VALUES)
    if suffix in (".tsv", ".tab", ".psm", ".txt"):
        return pl.read_csv(file_path, separator="\t", infer_schema_length=None, null_values=_PANDAS_NA_VALUES)
    if suffix == ".parquet":
        return pl.read_parquet(file_path)
    return None


@dataclass
class SearchResultSettings:
    """
    Dataclass to store search result settings.

    Attributes:
        search_engine: Name of the search engine used (e.g., "sage", "maxquant").
        quantification: Name of the quantification tool used (e.g., "sage", "maxquant", or None).
        label: Labeling method used (e.g., "tmt", "label_free").
        identification_file: identification file path.
        identification_level: Level of the identification data (e.g., "psm", "precursor", "peptide", "protein").
        quantification_file: identification file path (if applicable).
        quantification_level: Level of the quantification data (e.g., "psm", "precursor", "peptide", "protein", or None).
        ident_quant_merged: Indicates if identification and quantification are merged in a single file.
    """

    search_engine: str
    quantification: str | None
    label: Literal["tmt", "label_free"] | None
    acquisition: Literal["dda", "dia"] | None
    identification_file: list[str | Path | None]
    identification_df: pd.DataFrame
    identification_level: Literal["psm", "precursor", "peptide", "protein"]
    quantification_file: list[str | Path | None]
    quantification_df: pd.DataFrame | None
    quantification_level: Literal["psm", "precursor", "peptide", "protein"] | None
    ident_quant_merged: bool
    has_decoy: bool = True


@dataclass
class MuDataInput:
    """
    Dataclass to store inputs for creating a MuData object.

    Attributes:
        raw_identification_df: Raw identification DataFrame (varm['search_result']).
        norm_identification_df: Normalized identification DataFrame.
        norm_quant_df: Normalized quantification DataFrame.
        search_result: Original search result DataFrame.
    """

    raw_identification_df: pd.DataFrame
    norm_identification_df: pd.DataFrame
    norm_quant_df: pd.DataFrame | None
    decoy_df: pd.DataFrame | None


class SearchResultDataFrameConverter:
    """
    Base class for converting search result DataFrames into a format suitable for MuData.

    This class provides methods for normalizing identification and quantification DataFrames,
    as well as building the final MuData object. Inherited classes should implement specific
    logic for handling different search engine outputs.
    """

    def _convert_to_path(self, file_path: str | Path | pd.DataFrame) -> Path:
        if isinstance(file_path, str):
            if file_path.startswith(("http://", "https://", "ftp://")):
                return file_path
            else:
                return Path(file_path)
        elif isinstance(file_path, Path) or isinstance(file_path, pd.DataFrame):
            return file_path
        else:
            raise ValueError("file_path should be a string, Path object, or DataFrame.")

    def _convert_to_string(self, file_path: Path | None) -> str | None:
        if file_path is None:
            return None
        elif isinstance(file_path, str):
            return file_path
        elif isinstance(file_path, Path):
            return str(file_path)
        else:
            raise ValueError("file_path should be a Path object, or DataFrame.")

    @staticmethod
    def _read_file(file_path: str | Path):
        """Read a file into a polars DataFrame (the reader path is polars-only).

        csv/tsv/parquet are read natively with polars (see ``_polars_read_native``). A DataFrame
        passed directly (the ``read_*`` DataFrame-input API) is coerced to polars so the downstream
        reader transforms -- which are polars -- receive a polars frame.

        Returns a tuple of the file path (``None`` for a DataFrame input) and the polars frame.
        """
        import polars as pl

        if isinstance(file_path, pd.DataFrame):
            return None, pl.from_pandas(file_path)

        suffix = Path(file_path).suffix
        native_df = _polars_read_native(file_path, suffix)
        if native_df is None:
            raise ValueError(f"Unknown file type: {suffix}")
        return file_path, native_df

    def _read_files(self, file_paths: list[Path | pd.DataFrame], max_workers: int):
        """Read every input with polars and merge into a single polars DataFrame.

        polars is internally multi-threaded, so the files are read in-process (no ProcessPool) and
        concatenated with polars -- the frame never touches pandas until the reader's AnnData
        boundary. ``max_workers`` is accepted for API compatibility but unused (polars parallelises
        internally).

        ``diagonal_relaxed`` unions columns (missing -> null) AND relaxes dtypes, matching pandas'
        concat: multi-file inputs with an optional column (e.g. DIA-NN with/without Decoy) merge
        instead of raising the schema-mismatch error ``vertical_relaxed`` would.
        """
        import polars as pl

        results = [self.__class__._read_file(fp) for fp in file_paths]
        frames = [result[1] for result in results]
        merged_df = pl.concat(frames, how="diagonal_relaxed") if len(frames) > 1 else frames[0]
        merged_file_path = [result[0] for result in results if result[0] is not None]
        return merged_file_path, merged_df

    def convert(self, file_paths: list[str | Path | pd.DataFrame], max_workers: int = 4):
        """Read a list of file paths (or DataFrames) into a single merged polars DataFrame.

        Parameters:
            file_paths: file paths or DataFrames to read and merge.
            max_workers: accepted for API compatibility; unused (polars parallelises internally).

        Returns:
            A tuple of the merged file path(s) and the merged polars DataFrame.
        """
        file_paths_ = [self._convert_to_path(fp) for fp in file_paths]
        merged_file_path, merged_df = self._read_files(file_paths=file_paths_, max_workers=max_workers)
        merged_file_path = [self._convert_to_string(fp) for fp in merged_file_path]

        logger.debug("Files imported and merged into DataFrame with shape %s.", merged_df.shape)

        return merged_file_path, merged_df


class SearchResultReader:
    """
    Base class for reading and processing search engine results.

    Attributes:
        search_settings: Settings for the search results.
        used_feature_cols: List of columns to be used in the feature DataFrame.
        base_level: Base level of the data (e.g., "psm" or "precursor").
        _feature_rename_dict: Dictionary for renaming feature columns.

    Methods:
        read() -> md.MuData:
            Reads and processes the search results into a MuData object.
    """

    def __init__(self, _drop_search_result: bool = False) -> None:
        md.set_options(pull_on_update=False)
        self.search_settings: SearchResultSettings
        self._drop_search_result = _drop_search_result

        self._calc_exp_mz: Callable = _calc_exp_mz
        self._count_missed_cleavages: Callable = _count_missed_cleavages
        self._make_stripped_peptide: Callable = _make_stripped_peptide
        self._get_peptide_length: Callable = _get_peptide_length

        self.used_feature_cols: list[str] = [
            "proteins",
            "peptide",
            "stripped_peptide",
            "filename",
            "scan_num",
            "charge",
            "peptide_length",
        ]

        self._cols_to_stringify: list[str] = []  # placeholder, will be defined in inherited class

    @staticmethod
    def _make_unique_index(input_df: pd.DataFrame) -> pd.DataFrame:
        # Callers pass a freshly-built frame they own, so mutate in place. The old
        # defensive `input_df.copy()` materialized a third full copy of the multi-GB
        # identification frame at the read peak (measured: it owns ~13 GB of the
        # normalise spike). set_index returns a new frame that shares column blocks,
        # so the only added allocation here is the single tmp_index column.
        input_df["tmp_index"] = input_df["filename"] + "." + input_df["scan_num"].astype(str)
        return input_df.set_index("tmp_index", drop=True).rename_axis(index=None)

    @staticmethod
    def _strip_filename(filename: str) -> str:
        return Path(filename).name.rsplit(".", 1)[0]

    def _stringify_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert mixed type of pd.Series to sting to store as h5mu.
        """
        if len(self._cols_to_stringify) == 0:
            return df

        df = df.copy()
        for col in self._cols_to_stringify:
            if col in df.columns:
                df[col] = df[col].astype(str)

        return df

    def _validate_search_outputs(self) -> None:
        output_list: list[Path | None] = [
            self.search_settings.identification_file,
            self.search_settings.quantification_file,
        ]
        for file_path in output_list:
            if file_path is None:
                continue
            if not file_path.exists():
                raise FileNotFoundError(f"{file_path} does not exist!")

    def _read_config_file(self):
        raise NotImplementedError("_read_config_file method needs to be implemented in inherited class.")

    def _import_search_results(self) -> dict:
        output_dict: dict = dict()

        output_dict["identification"] = self.search_settings.identification_df
        output_dict["quantification"] = self.search_settings.quantification_df

        return output_dict

    def _extract_quant_from_raw(self, raw_identification_df: pd.DataFrame) -> pd.DataFrame:
        """Build the quantification frame directly from the raw merged frame.

        Used by merged readers that build the feature frame fresh: the feature
        frame carries identification columns only, so quantification is extracted
        from the raw frame (indexed to match the feature frame) instead of being
        split back out of the normalised frame.
        """
        raise NotImplementedError(
            "_extract_quant_from_raw must be implemented by merged readers that build the feature frame fresh."
        )

    def _make_needed_columns_for_identification(self, identification_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
            "_make_needed_columns_for_identification method needs to be implemented in inherited class."
        )

    def _normalise_identification_df(self, identification_df: pd.DataFrame) -> pd.DataFrame:
        # All readers build the feature frame on a fresh DataFrame, reading the raw frame
        # read-only, so it is passed directly without a defensive copy -- the raw frame
        # stays intact to serve varm or be freed.
        norm_identification_df = self._make_needed_columns_for_identification(
            identification_df
        )  # this will be method overriden in inherited class
        norm_identification_df = norm_identification_df.rename(columns=self._feature_rename_dict)
        norm_identification_df = self._make_unique_index(norm_identification_df)

        return norm_identification_df

    def _make_needed_columns_for_quantification(self, quantification_df: pd.DataFrame) -> pd.DataFrame:
        # flow through function, can be overriden in inherited class
        return quantification_df

    def _make_rename_dict_for_obs(self, quantification_df: pd.DataFrame) -> dict:
        # flow through function, can be overriden in inherited class
        return dict()

    def _normalise_quantification_df(self, quantification_df: pd.DataFrame) -> pd.DataFrame:
        # A sparse block-diagonal quant carrier already stores observed cells only (0/NaN are
        # absent), so the column rename / replace(0, NaN) below do not apply -- pass it through.
        if isinstance(quantification_df, SparseQuantFrame):
            # Correct only when this reader does not customise the quantification hooks (DIA-NN, the
            # sole sparse producer today, overrides neither). Guard so a reader that DOES override
            # them cannot later emit a sparse carrier and have its rename/select silently skipped.
            overrides_quant_hooks = (
                type(self)._make_needed_columns_for_quantification
                is not SearchResultReader._make_needed_columns_for_quantification
                or type(self)._make_rename_dict_for_obs is not SearchResultReader._make_rename_dict_for_obs
            )
            if overrides_quant_hooks:
                raise NotImplementedError(
                    f"{type(self).__name__} customises a quantification hook but emitted a "
                    "SparseQuantFrame; the sparse path does not apply those hooks yet."
                )
            return quantification_df
        # The quantification frame is never stored (varm holds the identification raw,
        # not this), so it can be mutated directly -- no defensive copy. Readers whose
        # _make_needed_columns_for_quantification needs its own copy still make one.
        norm_quant_df = self._make_needed_columns_for_quantification(
            quantification_df
        )  # this will be method overriden in inherited classs
        quant_rename_dict = self._make_rename_dict_for_obs(
            norm_quant_df
        )  # this will be method overriden in inherited class
        norm_quant_df = norm_quant_df.rename(columns=quant_rename_dict)
        norm_quant_df = norm_quant_df.replace(0, np.nan)

        return norm_quant_df

    def _make_mudata_input(self) -> MuDataInput:
        """
        Creates a MuDataInput object containing raw (.varm)and normalized psm (.var) and quantification (.X) DataFrames.

        Returns:
            MuDataInput: A MuDataInput object with raw and normalized data.
        """
        raw_dict: dict = self._import_search_results()
        # Avoid an eager full copy; normalization already copies internally.
        raw_identification_df: pd.DataFrame = raw_dict["identification"]

        norm_identification_df: pd.DataFrame = self._normalise_identification_df(raw_identification_df)
        if self.search_settings.ident_quant_merged:
            # Feature frame is built fresh (identification columns only); extract
            # quantification straight from the raw frame.
            quantification_df = self._extract_quant_from_raw(raw_identification_df)
        else:
            quantification_df = raw_dict["quantification"] if self.search_settings.quantification is not None else None

        if self.search_settings.has_decoy and "decoy" not in self.used_feature_cols:
            self.used_feature_cols.append("decoy")

        used_feature_cols = list(dict.fromkeys(self.used_feature_cols))
        norm_identification_df = norm_identification_df.loc[:, used_feature_cols]

        target_mask = np.ones(len(norm_identification_df), dtype=bool)

        if self.search_settings.has_decoy:
            if "decoy" not in norm_identification_df.columns:
                logger.error("Decoy column is expected but not found in the identification DataFrame.")
                raise ValueError("Decoy column is expected but not found in the identification DataFrame.")
            else:
                target_df, decoy_df = self._separate_decoy_df(norm_identification_df)
                target_mask = norm_identification_df["decoy"].eq(0).to_numpy()
                logger.debug("Decoy entries separated: %s", decoy_df.shape)
        else:
            target_df = norm_identification_df.copy()
            decoy_df = None

        if self._drop_search_result:
            # Keep only index alignment when raw search_result is not stored.
            raw_identification_df = pd.DataFrame(index=target_df.index)
        else:
            # The raw frame is stored in varm, which must be pandas. It is always a polars frame
            # here (the transforms consumed it as polars) -- convert once at this seam.
            raw_identification_df = raw_identification_df.to_pandas()
            raw_identification_df = raw_identification_df.copy()
            raw_identification_df.index = norm_identification_df.index
            # Keep raw and normalized rows in strict positional sync.
            raw_identification_df = raw_identification_df.iloc[target_mask, :].copy()

        norm_quant_df = self._normalise_quantification_df(quantification_df) if quantification_df is not None else None

        mudata_input: MuDataInput = MuDataInput(
            raw_identification_df=raw_identification_df,  # varm["search_result"]
            norm_identification_df=target_df,  # var
            norm_quant_df=norm_quant_df,  # X
            decoy_df=decoy_df,  # decoy entries
        )

        return mudata_input

    def _separate_decoy_df(self, norm_identification_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if "decoy" not in norm_identification_df.columns:
            raise ValueError("Decoy column not found in identification DataFrame.")

        decoy_df = norm_identification_df[norm_identification_df["decoy"] == 1].copy()
        target_df = norm_identification_df[norm_identification_df["decoy"] == 0].copy()

        return target_df, decoy_df

    def _update_default_adata_uns(self, adata: ad.AnnData) -> ad.AnnData:
        adata.uns.update(
            {
                "level": self.search_settings.identification_level,
                "search_engine": self.search_settings.search_engine,
                "quantification": self.search_settings.quantification,
                "label": self.search_settings.label,
                "acquisition": self.search_settings.acquisition,
                "identification_file": str(self.search_settings.identification_file),
                "quantification_file": (
                    str(self.search_settings.quantification_file)
                    if self.search_settings.quantification_file is not None
                    else None
                ),
            }
        )
        return adata

    def _build_mudata(self, mudata_input: MuDataInput) -> md.MuData:
        adata_dict = {}

        # Stringify only when raw search_result is materialized in varm.
        if not self._drop_search_result:
            for col in mudata_input.raw_identification_df.columns:
                if mudata_input.raw_identification_df[col].dtype == "object" and col not in self._cols_to_stringify:
                    self._cols_to_stringify.append(col)
            mudata_input.raw_identification_df = self._stringify_cols(mudata_input.raw_identification_df)

        # both feature and quantification are available in the same level
        if self.search_settings.quantification_level == self.search_settings.identification_level:
            if isinstance(mudata_input.norm_quant_df, SparseQuantFrame):
                sparse_quant = mudata_input.norm_quant_df
                common_index = mudata_input.norm_identification_df.index.intersection(sparse_quant.index)
                mod_adata = ad.AnnData(
                    X=sparse_quant.anndata_x(common_index),  # (samples x features) sparse, no dense pivot
                    obs=pd.DataFrame(index=sparse_quant.columns),
                    var=mudata_input.norm_identification_df.loc[common_index, :],
                )
            else:
                common_index = mudata_input.norm_identification_df.index.intersection(
                    mudata_input.norm_quant_df.index
                )
                # float32 to match msmu's .X convention (the other three .X builders below, the
                # sparse branch above, and read_diann's sparse output are all float32); without this
                # the dense same-level path was the lone float64 producer.
                mod_adata = ad.AnnData(mudata_input.norm_quant_df.loc[common_index, :].T.astype(np.float32))
                mod_adata.var = mudata_input.norm_identification_df.loc[common_index, :]
            if not self._drop_search_result:
                mod_adata.varm["search_result"] = mudata_input.raw_identification_df.loc[common_index, :]

            mod_adata = self._update_default_adata_uns(mod_adata)
            if mudata_input.decoy_df is not None:
                mod_adata.uns["decoy"] = mudata_input.decoy_df

            if self.search_settings.quantification_level in ["psm", "precursor"]:
                adata_dict["psm"] = mod_adata
            else:
                adata_dict[self.search_settings.quantification_level] = mod_adata

        # only feature is available
        elif self.search_settings.quantification_level is None:
            dummy_quantification_df = pd.DataFrame(
                index=mudata_input.norm_identification_df.index,
                columns=mudata_input.norm_identification_df["filename"].unique().tolist(),
            )
            mod_adata = ad.AnnData(dummy_quantification_df.T.astype(np.float32))
            mod_adata.var = mudata_input.norm_identification_df
            if not self._drop_search_result:
                mod_adata.varm["search_result"] = mudata_input.raw_identification_df
            mod_adata = self._update_default_adata_uns(mod_adata)
            if mudata_input.decoy_df is not None:
                mod_adata.uns["decoy"] = mudata_input.decoy_df

            adata_dict["psm"] = mod_adata

        # feature and quantification are available in different levels
        # (e.g., feature: psm, quantification: peptide)
        else:
            dummy_quantification_df = pd.DataFrame(
                index=mudata_input.norm_identification_df.index,
                columns=mudata_input.norm_quant_df.columns,
            )
            feat_adata = ad.AnnData(dummy_quantification_df.T.astype(np.float32))
            feat_adata.var = mudata_input.norm_identification_df
            if not self._drop_search_result:
                feat_adata.varm["search_result"] = mudata_input.raw_identification_df
            feat_adata = self._update_default_adata_uns(feat_adata)
            feat_adata.uns["decoy"] = mudata_input.decoy_df

            if self.search_settings.identification_level in ["psm", "precursor"]:
                adata_dict["psm"] = feat_adata
            else:
                adata_dict[self.search_settings.identification_level] = feat_adata

            quant_adata = ad.AnnData(mudata_input.norm_quant_df.T.astype(np.float32))
            quant_adata.uns.update(
                {
                    "level": self.search_settings.quantification_level,
                }
            )
            if self.search_settings.quantification_level in ["psm", "precursor"]:
                adata_dict["psm"] = quant_adata
            else:
                adata_dict[self.search_settings.quantification_level] = quant_adata

        mdata: md.MuData = md.MuData(adata_dict)

        return mdata

    def read(self) -> md.MuData:
        """
        Reads and processes the search results into a MuData object.

        Returns:
            A MuData object containing the processed search results.
        """
        # self._validate_search_outputs()

        mudata_input: MuDataInput = self._make_mudata_input()
        mdata: md.MuData = self._build_mudata(mudata_input=mudata_input)

        return mdata
