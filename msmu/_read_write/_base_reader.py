from pathlib import Path
from typing import Literal
from dataclasses import dataclass
import logging
from typing import Callable
import warnings

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from .._utils.peptide import (
    _calc_exp_mz,
    _count_missed_cleavages,
    _get_peptide_length,
    _make_stripped_peptide,
)


logger = logging.getLogger(__name__)


def _read_file(path: str | Path | pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Reads a file into a pandas DataFrame based on the file extension.

    Parameters:
        path: The file path to read.
        **kwargs: Additional keyword arguments to pass to the pandas read function.

    Returns:
        A pandas DataFrame containing the data from the file.
    """

    if isinstance(path, pd.DataFrame):
        return path

    if isinstance(path, Path):
        path = str(path)

    suffix = path.rsplit(".", maxsplit=1)[-1]

    if suffix in ["csv"]:
        return pd.read_csv(path, **kwargs)
    elif suffix in ["tsv", "tab"]:
        return pd.read_csv(path, sep="\t", **kwargs)
    elif suffix in ["xlsx", "xls"]:
        return pd.read_excel(path, **kwargs)
    elif suffix in ["parquet"]:
        return pd.read_parquet(path, **kwargs)
    elif suffix in ["json"]:
        return pd.read_json(path, **kwargs)
    else:
        raise ValueError(f"Unknown file type: {suffix}")


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
    identification_file: str | Path
    identification_level: Literal["psm", "precursor", "peptide", "protein"]
    quantification_file: str | None
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

    def __init__(self):
        md.set_options(pull_on_update=False)
        self.search_settings: SearchResultSettings

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
        df = input_df.copy()
        df["tmp_index"] = df["filename"] + "." + df["scan_num"].astype(str)
        df = df.set_index("tmp_index", drop=True).rename_axis(index=None)

        return df

    @staticmethod
    def _strip_filename(filename: str) -> str:
        return Path(filename).name.rsplit(".", 1)[0]

    def _stringify_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        # Convert specified columns to string type to avoid potential issues with mixed types
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

    def _read_identification_file(self) -> pd.DataFrame:
        identification_df = _read_file(self.search_settings.identification_file)
        identification_df = self._stringify_cols(identification_df)

        return identification_df

    def _read_config_file(self):
        raise NotImplementedError("_read_config_file method needs to be implemented in inherited class.")

    def _import_search_results(self) -> dict:
        output_dict: dict = dict()

        if self.search_settings.identification_file is not None:
            identification_df = self._read_identification_file()
            logger.info(f"Identification file loaded: {identification_df.shape}")

            if self.search_settings.quantification_file is not None:
                quantification_df = _read_file(self.search_settings.quantification_file)
                logger.info(f"Quantification file loaded: {quantification_df.shape}")
            else:
                quantification_df = None
        else:
            raise ValueError("Identification file path is not provided.")

        output_dict["identification"] = identification_df
        output_dict["quantification"] = quantification_df

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
        norm_identification_df = self._make_needed_columns_for_identification(
            identification_df.copy()
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
        norm_quant_df = self._make_needed_columns_for_quantification(
            quantification_df.copy()
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
        raw_identification_df: pd.DataFrame = raw_dict["identification"].copy()

        norm_identification_df: pd.DataFrame = self._normalise_identification_df(raw_identification_df)
        if self.search_settings.ident_quant_merged:
            identification_df, quantification_df = self._split_merged_identification_quantification(
                norm_identification_df
            )
            logger.info(
                f"Identification and quantification data split: {identification_df.shape}, {quantification_df.shape}"
            )
        else:
            identification_df = norm_identification_df.copy()
            quantification_df = (
                raw_dict["quantification"].copy() if self.search_settings.quantification is not None else None
            )

        norm_identification_df = norm_identification_df.loc[:, self.used_feature_cols]
        if self.search_settings.has_decoy:
            if "decoy" not in norm_identification_df.columns:
                logger.error("Decoy column is expected but not found in the identification DataFrame.")
                raise
            else:
                target_df, decoy_df = self._separate_decoy_df(norm_identification_df)
                logger.info(f"Decoy entries separated: {decoy_df.shape}")
        else:
            target_df = norm_identification_df.copy()
            decoy_df = None

        raw_identification_df.index = norm_identification_df.index
        raw_identification_df = raw_identification_df.loc[target_df.index,]

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
        target_df = target_df.drop(columns=["decoy"])

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
        # both feature and quantification are available in the same level
        if self.search_settings.quantification_level == self.search_settings.identification_level:
            common_index = mudata_input.norm_identification_df.index.intersection(mudata_input.norm_quant_df.index)
            mod_adata = ad.AnnData(mudata_input.norm_quant_df.loc[common_index, :].T)
            mod_adata.var = mudata_input.norm_identification_df.loc[common_index, :]
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
            mod_adata.varm["search_result"] = mudata_input.raw_identification_df
            mod_adata = self._update_default_adata_uns(mod_adata)
            if mudata_input.decoy_df is not None:
                mod_adata.uns["decoy"] = mudata_input.decoy_df

            adata_dict["psm"] = mod_adata

        # feature and quantification are available in different levels
        # (e.g., feature: psm, quantification: peptide)
        else:
            dummy_quantification_df = pd.DataFrame(
                index=mudata_input.norm_identification_df.index, columns=mudata_input.norm_quant_df.columns
            )
            feat_adata = ad.AnnData(dummy_quantification_df.T.astype(np.float32))
            feat_adata.var = mudata_input.norm_identification_df
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
