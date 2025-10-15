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

from ._reader_utils import ProteinIdParser
from .._utils.peptide import (
    _calc_exp_mz,
    _count_missed_cleavages,
    _get_peptide_length,
    _make_stripped_peptide,
)


logger = logging.getLogger(__name__)


def _get_separator(file_path: Path) -> str:
    if file_path.suffix in {".tsv", ".txt"}:
        return "\t"
    if file_path.suffix == ".csv":
        return ","
    warnings.warn(
        f"File format of {file_path} is not supported. Defaulting to tab separator.",
        stacklevel=2,
    )
    return "\t"


def _assign_protein_id_info(mdata: md.MuData) -> md.MuData:
    protein_id_info: ProteinIdParser = ProteinIdParser()
    protein_id_info.parse(proteins=mdata["feature"].var["proteins"], source="uniprot")

    protein_processed_mdata: md.MuData = mdata.copy()
    protein_processed_mdata["feature"].var["proteins"] = protein_id_info.accessions
    protein_processed_mdata.uns["protein_info"] = protein_id_info.protein_info

    return protein_processed_mdata


@dataclass
class SearchResultSettings:
    """
    Dataclass to store search result settings.
    Attributes:
        search_engine (str): Name of the search engine used (e.g., "sage", "maxquant").
        quantification (str | None): Name of the quantification tool used (e.g., "sage", "maxquant", or None).
        label (str): Labeling method used (e.g., "tmt", "label_free").
        output_dir (Path): Directory where the search results are stored.
        feature_file (str): Name of the feature file.
        feature_level (str): Level of the feature data (e.g., "psm", "precursor", "peptide", "protein").
        quantification_file (str | None): Name of the quantification file (if applicable).
        quantification_level (str | None): Level of the quantification data (e.g., "psm", "precursor", "peptide", "protein", or None).
        config_file (str | None): Name of the configuration file (if applicable).
        feat_quant_merged (bool): Indicates if feature and quantification are merged in a single file.
    """
    search_engine:str
    quantification:str | None
    label:Literal["tmt", "label_free"] | None
    acquisition:Literal["dda", "dia"] | None
    output_dir:Path
    feature_file:str
    feature_level:Literal["psm", "precursor", "peptide", "protein"]
    quantification_file:str | None
    quantification_level:Literal["psm", "precursor", "peptide", "protein"] | None
    config_file:str | None
    feat_quant_merged:bool

    @property
    def feature_path(self) -> Path | None:
        """
        Returns the full path to the feature file if it exists, otherwise returns None.
        """
        if self.feature_file is None:
            return None
        return self.output_dir / self.feature_file

    @property
    def quantification_path(self) -> Path | None:
        """
        Returns the full path to the quantification file if it exists, otherwise returns None.
        """
        if self.quantification_file is None:
            return None
        return self.output_dir / self.quantification_file

    @property
    def config_path(self) -> Path | None:
        """
        Returns the full path to the configuration file if it exists, otherwise returns None.
        """
        if self.config_file == None:
            return None
        return self.output_dir / self.config_file


@dataclass
class MuDataInput:
    """
    Dataclass to store inputs for creating a MuData object.
    Attributes:
        raw_feature_df (pd.DataFrame): Raw feature DataFrame (varm['search_result']).
        norm_feature_df (pd.DataFrame): Normalized feature DataFrame.
        norm_quant_df (pd.DataFrame): Normalized quantification DataFrame.
        search_result (pd.DataFrame): Original search result DataFrame.
        search_config (dict): Configuration settings from the search engine.
    """
    raw_feature_df: pd.DataFrame
    norm_feature_df: pd.DataFrame
    norm_quant_df: pd.DataFrame | None
    search_config: dict


class SearchResultReader:
    """
    Base class for reading and processing search engine results.
    Attributes:
    search_settings (SearchResultSettings): Settings for the search results.
    used_feature_cols (list[str]): List of columns to be used in the feature DataFrame.
    base_level (Literal["psm", "precursor"] | None): Base level of the data (e.g., "psm" or "precursor").
    _feature_rename_dict (dict): Dictionary for renaming feature columns.
    Methods:
        read() -> md.MuData:
            Reads and processes the search results into a MuData object.
    """
    def __init__(self):
        md.set_options(pull_on_update=False)
        self.search_settings: SearchResultSettings

        self._get_separator: Callable = _get_separator
        self._calc_exp_mz: Callable = _calc_exp_mz
        self._count_missed_cleavages: Callable = _count_missed_cleavages
        self._make_stripped_peptide: Callable = _make_stripped_peptide
        self._get_peptide_length: Callable = _get_peptide_length
        self._assign_protein_id_info: Callable = _assign_protein_id_info

        self.used_feature_cols: list[str] = [
            # "protein_group",
            "proteins",
            "peptide",
            "stripped_peptide",
            "filename",
            "scan_num",
            "charge",
            "peptide_length",
        ]

        self._cols_to_stringify: list[str] = [] # placeholder, will be defined in inherited class
        # self._feature_rename_dict: dict = {} # placeholder, will be defined in inherited class
        # self._quantification_rename_dict: dict = {} # placeholder, will be defined in inherited class

    @staticmethod
    def _make_unique_index(input_df:pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()
        df['tmp_index'] = df["filename"] + "." + df["scan_num"].astype(str)
        df = df.set_index("tmp_index", drop=True).rename_axis(index=None)

        return df

    @staticmethod
    def _strip_filename(filename: str) -> str:
        return Path(filename).name.rsplit(".", 1)[0]
    
    def _stringify_cols(self, df:pd.DataFrame) -> pd.DataFrame:
        # Convert specified columns to string type to avoid potential issues with mixed types
        if len(self._cols_to_stringify) == 0:
            return df

        df = df.copy()
        for col in self._cols_to_stringify:
            if col in df.columns:
                df[col] = df[col].astype(str)

        return df

    def _validate_search_outputs(self) -> None:
        output_list:list[Path | None] = [
            self.search_settings.feature_path,
            self.search_settings.quantification_path,
        ]
        for file_path in output_list:
            if file_path is None:
                continue
            if not file_path.exists():
                raise FileNotFoundError(f"{file_path} does not exist!")

    def _read_feature_file(self) -> pd.DataFrame:
        tmp_sep = self._get_separator(self.search_settings.feature_path)
        feature_df = pd.read_csv(self.search_settings.feature_path, sep=tmp_sep)
        feature_df = self._stringify_cols(feature_df)

        return feature_df
    
    def _read_config_file(self):
        raise NotImplementedError("_read_config_file method needs to be implemented in inherited class.")

    def _import_search_results(self) -> dict:
        output_dict:dict = dict()

        if self.search_settings.feature_path is not None:
            feature_df = self._read_feature_file()
            logger.info(f"Feature file loaded: {feature_df.shape}")
            
            if self.search_settings.quantification_path is not None:
                tmp_sep = self._get_separator(self.search_settings.quantification_path)
                quantification_df = pd.read_csv(self.search_settings.quantification_path, sep=tmp_sep)
                logger.info(f"Quantification file loaded: {quantification_df.shape}")
            else:
                quantification_df = None

            if self.search_settings.config_path is not None:
                config = self._read_config_file()
                logger.info(f"Config file loaded.")
        else:
            raise ValueError("Feature file path is not provided.")
        
        output_dict["feature"] = feature_df
        output_dict["quantification"] = quantification_df
        if self.search_settings.config_path is not None:
            output_dict["config"] = config

        return output_dict

    def _split_merged_feature_quantification(self, feature_df:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError("_split_merged_feature_quantification method needs to be implemented in inherited class.")

    def _make_needed_columns_for_feature(self, feature_df:pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("_make_needed_columns_for_feature method needs to be implemented in inherited class.")

    def _normalise_feature_df(self, feature_df:pd.DataFrame) -> pd.DataFrame:
        norm_feature_df = self._make_needed_columns_for_feature(feature_df.copy()) # this will be method overriden in inherited class
        norm_feature_df = norm_feature_df.rename(columns=self._feature_rename_dict)
        norm_feature_df = self._make_unique_index(norm_feature_df)
        
        return norm_feature_df
    
    def _make_needed_columns_for_quantification(self, quantification_df:pd.DataFrame) -> pd.DataFrame:
        # flow through function, can be overriden in inherited class
        return quantification_df

    def _make_rename_dict_for_obs(self, quantification_df:pd.DataFrame) -> dict:
        # flow through function, can be overriden in inherited class
        return dict()

    def _normalise_quantification_df(self, quantification_df: pd.DataFrame) -> pd.DataFrame:
        norm_quant_df = self._make_needed_columns_for_quantification(quantification_df.copy()) # this will be method overriden in inherited classs
        quant_rename_dict = self._make_rename_dict_for_obs(norm_quant_df) # this will be method overriden in inherited class
        norm_quant_df = norm_quant_df.rename(columns=quant_rename_dict)
        norm_quant_df = norm_quant_df.replace(0, np.nan)
        
        return norm_quant_df

    def _make_mudata_input(self) -> MuDataInput:
        """
        Creates a MuDataInput object containing raw (.varm)and normalized feature (.var) and quantification (.X) DataFrames.
        Returns:
            MuDataInput: A MuDataInput object with raw and normalized data.
        """
        raw_dict:dict = self._import_search_results()
        raw_feature_df:pd.DataFrame = raw_dict["feature"].copy()

        norm_feat_df:pd.DataFrame = self._normalise_feature_df(raw_feature_df)
        if self.search_settings.feat_quant_merged:
            feature_df, quantification_df = self._split_merged_feature_quantification(norm_feat_df)
            logger.info(f"Feature and quantification data split: {feature_df.shape}, {quantification_df.shape}")
        else:
            feature_df = norm_feat_df.copy()
            quantification_df = raw_dict["quantification"].copy() if self.search_settings.quantification is not None else None

        norm_feat_df = norm_feat_df.loc[:, self.used_feature_cols]

        raw_feature_df.index = norm_feat_df.index

        norm_quant_df = self._normalise_quantification_df(quantification_df) if quantification_df is not None else None

        mudata_input:MuDataInput = MuDataInput(
            raw_feature_df=raw_feature_df, # varm["search_result"]
            norm_feature_df=norm_feat_df, # var
            norm_quant_df=norm_quant_df, # X
            search_config=raw_dict.get("config", dict()) if "config" in raw_dict else dict(),
        )

        return mudata_input

    def _update_default_adata_uns(self, adata:ad.AnnData, config: dict | None) -> ad.AnnData:
        adata.uns.update(
            {
                "level": self.search_settings.feature_level,
                "search_engine": self.search_settings.search_engine,
                "quantification": self.search_settings.quantification,
                "label": self.search_settings.label,
                "search_output_dir": str(self.search_settings.output_dir),
                "search_config": config,
            }
        )
        return adata

    def _build_mudata(self, mudata_input:MuDataInput) -> md.MuData: 
        adata_dict = {}
        # both feature and quantification are available in the same level
        if self.search_settings.quantification_level == self.search_settings.feature_level:
            common_index = mudata_input.norm_feature_df.index.intersection(mudata_input.norm_quant_df.index)
            mod_adata = ad.AnnData(mudata_input.norm_quant_df.loc[common_index, :].T)
            mod_adata.var = mudata_input.norm_feature_df.loc[common_index, :]
            mod_adata.varm["search_result"] = mudata_input.raw_feature_df.loc[common_index, :]
            mod_adata = self._update_default_adata_uns(mod_adata, mudata_input.search_config)

            if self.search_settings.quantification_level in ["psm", "precursor"]:
                adata_dict["feature"] = mod_adata
            else:
                adata_dict[self.search_settings.quantification_level] = mod_adata

        # only feature is available
        elif self.search_settings.quantification_level is None:
            dummy_quantification_df = pd.DataFrame(
                index=mudata_input.norm_feature_df.index, 
                columns=mudata_input.norm_feature_df["filename"].unique().tolist()
                )
            mod_adata = ad.AnnData(dummy_quantification_df.T.astype(np.float32))
            mod_adata.var = mudata_input.norm_feature_df
            mod_adata.varm["search_result"] = mudata_input.raw_feature_df
            mod_adata = self._update_default_adata_uns(mod_adata, mudata_input.search_config)

            adata_dict["feature"] = mod_adata
        
        # feature and quantification are available in different levels 
        # (e.g., feature: psm, quantification: peptide)
        else:
            dummy_quantification_df = pd.DataFrame(
                index=mudata_input.norm_feature_df.index, 
                columns=mudata_input.norm_quant_df.columns
                )
            feat_adata = ad.AnnData(dummy_quantification_df.T.astype(np.float32))
            feat_adata.var = mudata_input.norm_feature_df
            feat_adata.varm["search_result"] = mudata_input.raw_feature_df
            feat_adata = self._update_default_adata_uns(feat_adata, mudata_input.search_config)

            if self.search_settings.feature_level in ["psm", "precursor"]:
                adata_dict["feature"] = feat_adata
            else:
                adata_dict[self.search_settings.feature_level] = feat_adata

            quant_adata = ad.AnnData(mudata_input.norm_quant_df.T.astype(np.float32))
            quant_adata.uns.update(
                {
                    "level": self.search_settings.quantification_level,
                }
            )
            if self.search_settings.quantification_level in ["psm", "precursor"]:
                adata_dict["feature"] = quant_adata
            else:
                adata_dict[self.search_settings.quantification_level] = quant_adata

        mdata: md.MuData = md.MuData(adata_dict)

        return mdata

    def read(self) -> md.MuData:
        """
        Reads and processes the search results into a MuData object.
        Returns:
            md.MuData: A MuData object containing the processed search results.
        """
        self._validate_search_outputs()

        mudata_input:MuDataInput = self._make_mudata_input()
        mdata:md.MuData = self._build_mudata(mudata_input=mudata_input)

        mdata = self._assign_protein_id_info(mdata=mdata)

        return mdata