from pathlib import Path
from typing import Literal
from dataclasses import dataclass
from typing import Callable

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

# from pyteomics import mzid
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm

from ..logging_utils import get_logger
from .._utils.peptide import (
    _calc_exp_mz,
    _count_missed_cleavages,
    _get_peptide_length,
    _make_stripped_peptide,
)


logger = get_logger(__name__)


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
    def _read_file(file_path: str | Path) -> tuple[str | Path | None, pd.DataFrame]:
        """
        Reads a file and returns its path and content as a DataFrame.

        Parameters:
            file_path: The path to the file to be read.

        Returns:
            A tuple containing the file path and the content as a DataFrame.
        """

        if isinstance(file_path, pd.DataFrame):
            return None, file_path

        tmp_file_path = Path(file_path)

        suffix = tmp_file_path.suffix
        if suffix in [".csv"]:
            df = pd.read_csv(file_path)
        elif suffix in [".tsv", ".tab", ".psm", ".txt"]:
            df = pd.read_csv(file_path, sep="\t")
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        elif suffix in [".parquet"]:
            df = pd.read_parquet(file_path)
        elif suffix in [".json"]:
            df = pd.read_json(file_path)
        # elif suffix in [".mzid"]:
        #     df = mzid.DataFrame(file_path)
        else:
            raise ValueError(f"Unknown file type: {suffix}")

        return file_path, df

    def _read_files(self, file_paths: list[Path | pd.DataFrame], max_workers: int) -> tuple[Path | None, pd.DataFrame]:
        """
        Reads a file and returns its path and content as a DataFrame.

        Parameters:
            file_path: The path to the file to be read.

        Returns:
            A tuple containing the file path and the content as a DataFrame.
        """

        results = []
        if len(file_paths) == 1:
            # Fast path: avoid ProcessPool startup/pickling overhead for single input.
            results.append(self.__class__._read_file(file_paths[0]))
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_file = {executor.submit(self.__class__._read_file, file): file for file in file_paths}
                for future in tqdm(
                    as_completed(future_file),
                    total=len(file_paths),
                    desc="Reading files",
                ):
                    file = future_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.exception("Error processing %s.", file)
                        raise e

        merged_df = pd.concat([result[1] for result in results], ignore_index=True)
        if len(results) >= 1:
            merged_file_path = [result[0] for result in results if result[0] is not None]

        elif len(results) == 0:
            merged_file_path = None

        return merged_file_path, merged_df

    def convert(
        self, file_paths: list[str | Path | pd.DataFrame], max_workers: int = 4
    ) -> tuple[str | Path | None, pd.DataFrame]:
        """
        Converts a list of file paths or DataFrames into a single DataFrame.

        Parameters:
            file_paths: A list of file paths or DataFrames to be converted.
            max_workers: The maximum number of worker processes to use for reading files.
        Returns:
            A tuple containing the merged file path (if applicable) and the merged DataFrame.
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

    def __init__(self, _drop_search_result: bool) -> None:
        md.set_options(pull_on_update=False)
        self.search_settings: SearchResultSettings
        self._drop_search_result = _drop_search_result
        # Readers that build the feature frame on a fresh DataFrame (reading the
        # raw frame read-only, never mutating it) flip this so the raw frame is
        # not defensively copied before normalisation -- it stays intact to serve
        # varm or be freed. Legacy in-place readers leave it False.
        self._builds_feature_frame_fresh = False

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

    @staticmethod
    def _map_unique(series: pd.Series, scalar_func) -> pd.Series:
        """Apply a scalar function over distinct values only, then map back.

        Equivalent to ``series.apply(scalar_func)`` but evaluates ``scalar_func``
        once per unique value -- a large win for low-cardinality columns such as
        ``filename`` (a handful of raw files repeated across millions of PSMs).
        """
        return series.map({value: scalar_func(value) for value in series.unique()})

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

    def _split_merged_identification_quantification(
        self, identification_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError(
            "_split_merged_identification_quantification method needs to be implemented in inherited class."
        )

    def _make_needed_columns_for_identification(self, identification_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
            "_make_needed_columns_for_identification method needs to be implemented in inherited class."
        )

    def _normalise_identification_df(self, identification_df: pd.DataFrame) -> pd.DataFrame:
        # Non-destructive readers build a fresh frame, so the raw frame can be read
        # directly; legacy in-place readers still get a defensive copy.
        source_df = identification_df if self._builds_feature_frame_fresh else identification_df.copy()
        norm_identification_df = self._make_needed_columns_for_identification(
            source_df
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
            identification_df, quantification_df = self._split_merged_identification_quantification(
                norm_identification_df
            )
            logger.debug(
                "Identification and quantification data split: %s, %s",
                identification_df.shape,
                quantification_df.shape,
            )
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
            common_index = mudata_input.norm_identification_df.index.intersection(mudata_input.norm_quant_df.index)
            mod_adata = ad.AnnData(mudata_input.norm_quant_df.loc[common_index, :].T)
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
