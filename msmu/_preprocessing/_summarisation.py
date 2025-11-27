import re
import logging

import numpy as np
import pandas as pd

from ._filter import _mask_boolean_filter

# for type checking only
import anndata as ad
from typing import Literal


logger = logging.getLogger(__name__)


class FeatureRanker:
    """Ranking methods for selecting top features based on quantification data."""

    @staticmethod
    def total_intensity(identification_df, quantification_df, col_to_groupby):
        """
        Rank features based on total intensity across all samples.

        Args:
            identification_df (pd.DataFrame): DataFrame containing feature identifications.
            quantification_df (pd.DataFrame): DataFrame containing feature quantifications.
            col_to_groupby (str): Column name to group by for ranking.

        Returns:
            pd.DataFrame: DataFrame with added 'rank_score' and 'rank' columns.
        """
        sum_intensity = quantification_df.sum(axis=1)
        identification_df.loc[:, "rank_score"] = sum_intensity
        identification_df.loc[:, "rank"] = identification_df.groupby(col_to_groupby)["rank_score"].rank(ascending=False)

        return identification_df

    @staticmethod
    def max_intensity(identification_df, quantification_df, col_to_groupby):
        """
        Rank features based on maximum intensity across all samples.

        Args:
            identification_df (pd.DataFrame): DataFrame containing feature identifications.
            quantification_df (pd.DataFrame): DataFrame containing feature quantifications.
            col_to_groupby (str): Column name to group by for ranking.

        Returns:
            pd.DataFrame: DataFrame with added 'rank_score' and 'rank' columns.
        """
        max_intensity = quantification_df.max(axis=1)
        identification_df.loc[:, "rank_score"] = max_intensity
        identification_df.loc[:, "rank"] = identification_df.groupby(col_to_groupby)["rank_score"].rank(ascending=False)

        return identification_df

    @staticmethod
    def median_intensity(identification_df, quantification_df, col_to_groupby):
        """
        Rank features based on median intensity across all samples.

        Args:
            identification_df (pd.DataFrame): DataFrame containing feature identifications.
            quantification_df (pd.DataFrame): DataFrame containing feature quantifications.
            col_to_groupby (str): Column name to group by for ranking.

        Returns:
            pd.DataFrame: DataFrame with added 'rank_score' and 'rank' columns.
        """
        median_intensity = quantification_df.median(axis=1)
        identification_df.loc[:, "rank_score"] = median_intensity
        identification_df.loc[:, "rank"] = identification_df.groupby(col_to_groupby)["rank_score"].rank(ascending=False)

        return identification_df


class Scorer:
    """Scoring methods for aggregating PSM scores to peptide/protein scores."""

    EPS = 1e-10

    def __init__(self, pep: float | np.ndarray | list[float]):
        self._raw_pep = np.asarray(pep, dtype=float)
        self._picked_pep: float | None = None

    @classmethod
    def best_pep(cls, values):
        """Factory for best PEP aggregation."""
        scorer = cls(values)
        scorer._picked_pep = scorer._best_pep()
        return scorer

    # @classmethod
    # def combined(cls, values):
    #     """Factory for combined PEP aggregation."""
    #     scorer = cls(values)
    #     scorer._picked_pep = scorer._combined_pep()
    #     return scorer

    def _best_pep(self) -> float:
        """Return the minimum PEP (best evidence)."""
        arr = np.asarray(self._raw_pep, dtype=float)
        if arr.size == 0:
            return np.nan
        return np.nanmin(arr)

    # def _combined_pep(self) -> float:
    #     """Combine PEPs as 1 - product(1 - PEP_i)."""
    #     arr = np.asarray(self._raw_pep, dtype=float)
    #     arr = np.clip(arr, 0.0, 1.0)
    #     valid = arr[~np.isnan(arr)]
    #     if valid.size == 0:
    #         return np.nan
    #     log_term = np.sum(np.log1p(-valid))
    #     return 1.0 - np.exp(log_term)

    @property
    def picked_pep(self) -> float:
        """The aggregated PEP value."""
        return self._picked_pep

    @property
    def picked_score(self) -> float:
        """The −log10 transformed score."""
        if self._picked_pep is None or np.isnan(self._picked_pep):
            return np.nan
        return -np.log10(self._picked_pep + self.EPS)

    @classmethod
    def func(cls, method: str):
        """Return a pure function that returns numeric PEPs (for pandas .agg)."""
        if method == "best_pep":
            return lambda x: cls.best_pep(x).picked_pep
        elif method == "combined":
            return lambda x: cls.combined(x).picked_pep
        else:
            raise ValueError(f"Scoring method '{method}' not recognized.")


class Aggregator:
    """
    Base class for aggregating identification and quantification data.
    """

    def __init__(
        self,
        identification_df: pd.DataFrame,
        quantification_df: pd.DataFrame,
        decoy_df: pd.DataFrame | None,
        agg_method: Literal["median", "mean", "sum"],
        score_method: Literal["best_pep", "fisher", "stouffer", "combined"],
    ) -> None:
        self._id_df: pd.DataFrame = identification_df.copy()
        self._quant_df: pd.DataFrame = quantification_df.copy()
        self._decoy_id_df: pd.DataFrame = decoy_df.copy() if decoy_df is not None else pd.DataFrame()
        self._agg_method: Literal["median", "mean", "sum"] = agg_method
        self._score_method: Literal["best_pep", "fisher", "stouffer", "combined"] = score_method

        self._id_agg_dict: dict = dict()  # placeholder
        self._col_to_groupby: str = ""  # placeholder
        self._decoy_agg_dict: dict = dict()  # placeholder

    @classmethod
    def peptide(
        cls,
        identification_df,
        quantification_df,
        decoy_df,
        agg_method,
        score_method,
        protein_col,
        peptide_col,
    ):
        """
        Create a peptide-level aggregator.
        """
        aggregator = cls(
            identification_df,
            quantification_df,
            decoy_df,
            agg_method,
            score_method,
        )
        aggregator._col_to_groupby = peptide_col
        aggregator._protein_col = protein_col
        aggregator._id_agg_dict = {
            aggregator._col_to_groupby: (aggregator._col_to_groupby, "first"),
            aggregator._protein_col: (aggregator._protein_col, "first"),
            "stripped_peptide": ("stripped_peptide", "first"),
            "count_psm": ("peptide", "count"),
            "PEP": ("PEP", Scorer.func(score_method)),
        }

        aggregator._decoy_agg_dict = {
            aggregator._protein_col: (aggregator._protein_col, "first"),
            "stripped_peptide": ("stripped_peptide", "first"),
            "PEP": ("PEP", Scorer.func(score_method)),
        }

        return aggregator

    @classmethod
    def protein(
        cls,
        identification_df,
        quantification_df,
        decoy_df,
        agg_method,
        score_method,
        protein_col,
    ):
        """
        Create a protein-level aggregator.
        """
        aggregator = cls(identification_df, quantification_df, decoy_df, agg_method, score_method)
        aggregator._col_to_groupby = protein_col
        aggregator._id_agg_dict = {
            # "total_psm": "sum",
            "count_psm": ("count_psm", "sum"),
            "count_stripped_peptide": ("stripped_peptide", "nunique"),
            "PEP": ("PEP", Scorer.func(score_method)),
        }

        aggregator._decoy_agg_dict = {"PEP": ("PEP", Scorer.func(score_method))}

        return aggregator

    @classmethod
    def ptm_site(
        cls,
        identification_df,
        quantification_df,
        agg_method,
    ):
        """
        Create a PTM site-level aggregator.
        """
        aggregator = cls(identification_df, quantification_df, None, agg_method, None)
        aggregator._col_to_groupby = "protein_site"
        aggregator._id_agg_dict = {
            "count_psm": ("count_psm", "sum"),
            "peptide": ("peptide", lambda x: ";".join(sorted(x.unique()))),
            "count_peptide": ("peptide", "nunique"),
            "count_stripped_peptide": ("stripped_peptide", "nunique"),
            "modified_protein": ("modified_protein", "first"),
            "protein_group": ("protein_group", "first"),
        }

        return aggregator

    def aggregate_identification(self) -> pd.DataFrame:
        agg_id_df: pd.DataFrame = self._id_df.copy()
        col_to_groupby = self._col_to_groupby

        agg_id_df = agg_id_df.groupby(col_to_groupby, observed=False).agg(**self._id_agg_dict)

        agg_id_df = agg_id_df.rename_axis(index=None)

        return agg_id_df

    def aggregate_quantification(self) -> pd.DataFrame:
        agg_quant_df: pd.DataFrame = self._quant_df.copy()
        agg_quant_df[self._col_to_groupby] = self._id_df[self._col_to_groupby]
        agg_quant_df = agg_quant_df.groupby(self._col_to_groupby, observed=False).agg(self._agg_method)

        agg_quant_df = agg_quant_df.rename_axis(index=None)

        return agg_quant_df

    def aggregate_decoy(self) -> pd.DataFrame:
        agg_decoy_df: pd.DataFrame = self._decoy_id_df.copy()
        agg_decoy_df = agg_decoy_df.groupby(self._col_to_groupby, observed=False).agg(**self._decoy_agg_dict)

        agg_decoy_df = agg_decoy_df.rename_axis(index=None)

        return agg_decoy_df


class SummarisationPrep:
    """
    Preparation steps for summarisation.

    Attributes:
        mdata (MuData): MuData object containing feature-level data.
        filter_dict (dict): Dictionary specifying filtering criteria.
        rank_dict (dict): Dictionary specifying ranking criteria.
    """

    def __init__(self, adata: ad.AnnData, col_to_groupby: str, has_decoy: bool) -> None:
        self.adata: ad.AnnData = adata.copy()
        self._col_to_groupby = col_to_groupby

        self._filter_dict: dict = {}  # {"column_name": (keep, value)} | {"purity": ("gt", 0.7)}
        self._rank_tuple: tuple = ()  # ("method", num_top) | ("max_intensity", 3)
        self._has_decoy: bool = has_decoy

    @property
    def filter_dict(self) -> dict:
        return self._filter_dict

    @filter_dict.setter
    def filter_dict(self, new_filter_dict: dict) -> None:
        logger.info(f"Applying filter criteria: {new_filter_dict}")
        self._filter_dict = new_filter_dict

    @property
    def rank_tuple(self) -> tuple:
        return self._rank_tuple

    @rank_tuple.setter
    def rank_tuple(self, new_rank_tuple: tuple) -> None:
        logger.info(f"Ranking features by '{new_rank_tuple[0]}' to select top {new_rank_tuple[1]} features.")
        self._rank_tuple = new_rank_tuple

    def prepare_data_to_summarise(self) -> pd.DataFrame:
        identification_df: pd.DataFrame = self.adata.var.copy()
        quantification_df: pd.DataFrame = self.adata.to_df().transpose().copy()
        if self._has_decoy:
            decoy_df: pd.DataFrame = self.adata.uns["decoy"].copy()

        return identification_df, quantification_df, decoy_df if self._has_decoy else None

    def _make_filter_mask(self, id_df: pd.DataFrame):
        filter_indices = pd.Series(False, index=id_df.index)

        for column, (keep, value) in self._filter_dict.items():
            column_mask = _mask_boolean_filter(series_to_mask=id_df[column], keep=keep, value=value)
            filter_indices = filter_indices | column_mask

        return filter_indices

    def _make_rank_mask(self) -> pd.Series:
        rank_method, top_n = self.rank_tuple

        ranked_id_df = FeatureRanker().__getattribute__(rank_method)(
            identification_df=self.adata.var,
            quantification_df=self.adata.to_df().transpose(),
            col_to_groupby=self._col_to_groupby,
        )

        rank_mask = _mask_boolean_filter(series_to_mask=ranked_id_df["rank"], keep="le", value=top_n)

        return rank_mask

    def _mask_quantification(self, quant_df: pd.DataFrame, mask_indices: pd.Series) -> pd.DataFrame:
        mask_with_nan_quant = quant_df.copy()
        mask_with_nan_quant.loc[~mask_indices, :] = np.nan

        return mask_with_nan_quant

    def prep(self):
        identification_df, quantification_df, decoy_df = self.prepare_data_to_summarise()

        # make filter mask
        if self._filter_dict:
            filter_mask = self._make_filter_mask(identification_df)
            quantification_df = self._mask_quantification(quantification_df, filter_mask)

        # make rank mask
        if self.rank_tuple:
            rank_mask = self._make_rank_mask()
            quantification_df = self._mask_quantification(quantification_df, rank_mask)

        return identification_df, quantification_df, decoy_df if self._has_decoy else None


class PtmSummarisationPrep(SummarisationPrep):
    """
    Preparation steps for PTM site summarisation.
        1. Filter data with only modified peptides with modi_identifier
        2. Get modified sites from peptide
        3. Label peptide site
        4. Explode data to single protein for labeling protein site
        5. Label protein site to each single protein
        6. Wrap up single protein to single protein group
        7. Group by modified peptide and its peptide site
        8. Merge data with peptide value indexed by peptide
    """

    def __init__(self, adata: ad.AnnData, modi_identifier: str, fasta: pd.DataFrame) -> None:
        self._modi_identifier = modi_identifier
        self._fasta_dict: dict = fasta["Sequence"].to_dict()
        self._col_to_groupby = "ptm_site"

        super().__init__(adata, self._col_to_groupby, has_decoy=False)

    def prep(self):
        identification_df, quantification_df, _ = self.prepare_data_to_summarise()
        identification_df["peptide"] = identification_df.index
        modi_df = self._extract_modi_peptide_df(data=identification_df)

        labelled_ptm_df = self.label_ptm_site(
            data=modi_df,
        )

        quantification_df = pd.merge(
            labelled_ptm_df[["peptide", "protein_site"]],
            quantification_df,
            how="left",
            left_on="peptide",
            right_index=True,
        ).drop(columns="peptide")

        # make rank mask
        if self.rank_tuple:
            rank_mask = self._make_rank_mask()
            quantification_df = self._mask_quantification(quantification_df, rank_mask)

        return labelled_ptm_df, quantification_df

    def _extract_modi_peptide_df(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        extracted_df: pd.DataFrame = data.copy()
        extracted_df = extracted_df.loc[extracted_df["peptide"].str.contains(re.escape(self._modi_identifier))].copy()
        logger.info(f"Extracted modified peptides: {len(extracted_df)} / {len(data)}")

        return extracted_df

    def label_ptm_site(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Label PTM site to each single protein and get data arranged by peptide - peptide site

        Parameters:
            data (pd.DataFrame): Peptide data from msmu mudata['peptide']

        Returns:
            ptm_data (pd.DataFrame): PTM data arranged by peptide - peptide site
        """
        ptm_info: pd.DataFrame = data.copy()

        ptm_info["peptide_site"] = ptm_info["peptide"].apply(lambda x: self._get_mod_sites(x, self._modi_identifier))

        # label peptide site
        ptm_info["peptide_site"] = ptm_info["peptide_site"].apply(lambda x: self._label_peptide_site(x))

        # explode data to single protein for label protein site
        ptm_info = self._explode_mod_site(ptm_info)
        ptm_info = self._explode_protein_groups(ptm_info)
        ptm_info = self._explode_protein_group(ptm_info)

        # label protein site to each single protein
        ptm_info["protein_site"] = ptm_info.apply(
            lambda x: self._label_protein_site(
                protein=x._prots,
                peptide=x.stripped_peptide,
                pep_site=x.peptide_site,
                fasta_dict=self._fasta_dict,
            ),
            axis=1,
        )
        ptm_info = ptm_info.loc[ptm_info["protein_site"].str.len() > 0].copy()
        ptm_info["modified_protein"] = ptm_info["protein_site"].apply(lambda x: x.split("|")[0])

        # wrap up single protein to single protein group
        ptm_info = self._implode_protein_group(ptm_info)

        # group by modified peptide and its peptide site
        ptm_info = self._implode_peptide_peptide_site(ptm_info)

        return ptm_info

    def _get_mod_sites(self, pep: str, modi_identifier) -> list:
        mod_sites: list = pep.split(modi_identifier)
        mod_sites: list = mod_sites[:-1]

        return mod_sites

    def _label_peptide_site(self, mod_sites: list) -> list:
        sites = list()
        site_pos: int = 0
        for mod in mod_sites:
            mod = "".join(filter(str.isalpha, mod))
            site_pos = site_pos + len(mod)
            site = f"{mod[-1]}{site_pos}"
            sites.append(site)

        return sites

    def _label_protein_site(self, protein: str, peptide: str, pep_site: str, fasta_dict: dict) -> str:
        aa: str = pep_site[0]
        pos: int = int(pep_site[1:])
        prot_site: str = ""

        res: list = list()
        prot_split = self._get_uniprot(protein)

        if prot_split in fasta_dict.keys():
            refseq: str = fasta_dict[prot_split]
            for match in re.finditer(peptide, refseq):
                matched = f"{prot_split}|{aa}{pos + match.span()[0]}"
                res.append(matched)
            prot_site = "/".join(res)

        return prot_site

    def _explode_mod_site(self, pep_labed_data: pd.DataFrame) -> pd.DataFrame:
        pep_labed_data = pep_labed_data.explode("peptide_site", ignore_index=True)

        return pep_labed_data

    def _explode_protein_groups(self, pep_labed_data: pd.DataFrame) -> pd.DataFrame:
        pep_labed_data["_prot_gr"] = pep_labed_data["protein_group"]
        pep_labed_data["_prot_gr"] = pep_labed_data["_prot_gr"].str.split(";")
        exploded_data = pep_labed_data.explode("_prot_gr", ignore_index=True)

        return exploded_data

    def _explode_protein_group(self, data) -> pd.DataFrame:
        data["_prots"] = data["_prot_gr"]
        data["_prots"] = data["_prots"].str.split(",")
        exploded_data = data.explode("_prots", ignore_index=True)

        return exploded_data

    def _implode_protein_group(self, data) -> pd.DataFrame:
        data = (
            data.groupby(["peptide", "peptide_site", "_prot_gr"], as_index=False)
            .agg(
                {
                    "protein_site": ",".join,
                    "protein_group": "first",
                    "modified_protein": ",".join,
                    "stripped_peptide": "first",
                    "count_psm": "sum",
                    # "repr_protein": "first",
                }
            )
            .copy()
        )

        return data

    def _implode_peptide_peptide_site(self, data) -> pd.DataFrame:
        data = data.groupby(["peptide", "peptide_site"], as_index=False).agg(
            {
                "protein_site": ";".join,
                "protein_group": "first",
                "modified_protein": ";".join,
                "stripped_peptide": "first",
                "count_psm": "sum",
                # "repr_protein": "first",
            }
        )

        return data

    def _get_uniprot(self, protein: str) -> str:
        return protein
