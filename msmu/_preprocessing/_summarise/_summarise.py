import warnings
from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from ..._read_write._readers import add_modality
from ..._utils.utils import get_modality_dict
from ._summariser import PeptideSummariser, ProteinSummariser, PtmSummariser

warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")


def to_peptide(
    mdata: md.MuData,
    rank_method: str | None = None,
    from_: str = "psm",
    sum_method: str = "median",
    peptide_col: str = "peptide",
    protein_col: str = "protein_group",
    top_n: int | None = None,
) -> md.MuData:
    """
    Summarise peptide level data from PSM level data.
    """
    modality_dict: dict[str, ad.AnnData] = get_modality_dict(mdata=mdata, level=from_)

    adata_list: list[ad.AnnData] = list()
    for mod_adata in modality_dict.values():
        summ: PeptideSummariser = PeptideSummariser(
            adata=mod_adata,
            peptide_col=peptide_col,
            protein_col=protein_col,
            from_=from_,
        )
        data: pd.DataFrame = summ.get_data()

        if top_n is not None:
            data: pd.DataFrame = summ.rank_psm(data=data, rank_method=rank_method)
            data: pd.DataFrame = summ.filter_by_rank(data=data, top_n=top_n)

        summaried_data: pd.DataFrame = summ.summarise_data(
            data=data, sum_method=sum_method
        )
        peptide_adata: pd.DataFrame = summ.data2adata(data=summaried_data)
        adata_list.append(peptide_adata)

    merged_peptide_adata: ad.AnnData = ad.concat(adatas=adata_list, join="outer")
    if np.all(np.isnan(merged_peptide_adata.X.astype(np.float64))):  # for lfq
        peptide_arr: pd.DataFrame = mdata["peptide"].to_df().T
        intersect_idx = peptide_arr.index.intersection(merged_peptide_adata.var_names)
        peptide_arr: pd.DataFrame = peptide_arr.loc[intersect_idx].T
        merged_peptide_adata: ad.AnnData = merged_peptide_adata[:, intersect_idx]
        merged_peptide_adata.X = peptide_arr.T

    merged_var: pd.DataFrame = _merged_var_df(
        adata_list=adata_list, protein_col=protein_col
    )

    merged_peptide_adata.var = merged_var.loc[merged_peptide_adata.var_names]
    merged_peptide_adata.uns["level"] = "peptide"

    mdata: md.MuData = add_modality(
        mdata=mdata,
        adata=merged_peptide_adata,
        mod_name="peptide",
        parent_mods=list(modality_dict.keys()),
    )
    mdata.push_obs()
    mdata.update_var()

    return mdata


def _merged_var_df(adata_list: list[ad.AnnData], protein_col: str) -> pd.DataFrame:
    merged_var: list[str] = [x.var for x in adata_list]
    merged_var: pd.DataFrame = pd.concat(merged_var, join="outer")
    merged_agg_dict: dict = {
        protein_col: "first",
        "stripped_peptide": "first",
        "modifications": "first",
        "total_psm": "sum",
        "num_used_psm": "sum",
    }
    if "peptide_type" in merged_var.columns:
        merged_agg_dict["peptide_type"] = "first"
    if "repr_protein" in merged_var.columns:
        merged_agg_dict["repr_protein"] = "first"

    merged_var: pd.DataFrame = merged_var.groupby(merged_var.index, observed=False).agg(
        merged_agg_dict
    )
    merged_var: pd.DataFrame = merged_var[~merged_var.index.duplicated(keep="first")]

    return merged_var


def to_protein(
    mdata,
    protein_col="protein_group",
    min_n_peptides=1,
    sum_method="median",
    from_="peptide",
) -> md.MuData:
    """
    Summarise protein level data from peptide level data.
    """
    modality_dict: dict[str, ad.AnnData] = get_modality_dict(mdata=mdata, level=from_)
    adata: ad.AnnData = modality_dict[from_].copy()

    summ: ProteinSummariser = ProteinSummariser(
        adata=adata, protein_col=protein_col, from_=from_
    )
    data: pd.DataFrame = summ.get_data()
    unique_filtered_data: pd.DataFrame = summ.filter_unique_peptides(data=data)
    summarised_data: pd.DataFrame = summ.summarise_data(
        data=unique_filtered_data, sum_method=sum_method
    )
    summarised_data: pd.DataFrame = summ.filter_n_min_peptides(
        data=summarised_data, min_n_peptides=min_n_peptides
    )

    protein_adata: ad.AnnData = summ.data2adata(data=summarised_data)

    mdata: md.MuData = add_modality(
        mdata=mdata, adata=protein_adata, mod_name=summ._to, parent_mods=[from_]
    )
    mdata.push_obs()
    mdata.update_var()

    return mdata


def to_ptm_site(
    mdata: md.MuData,
    protein_col: str,
    fasta_file: str | Path,
    modification_name: str,
    modification_mass: float | None = None,
    sum_method: str = "median",
) -> md.MuData:
    """
    Summarise PTM site level data from peptide level data.
    """

    if modification_mass is None:
        if modification_name in PtmPreset.__dict__:
            preset = getattr(PtmPreset, modification_name)()
            modification_mass = preset.mass
        else:
            raise ValueError(
                f"Unknown modification name: {modification_name}. Please provide modification mass."
            )
    else:
        if modification_name in PtmPreset.__dict__:
            warnings.warn(
                f"Modification mass is provided for {modification_name}. "
                f"Provided mass will be used instead of default mass."
            )
            preset = PtmPreset(name=modification_name, mass=modification_mass)

        elif modification_name not in PtmPreset.__dict__:  # user defined modification
            preset = PtmPreset(name=modification_name, mass=modification_mass)
        else:
            raise ValueError(
                f"Unknown modification name: {modification_name}. Please provide modification mass."
            )

    peptide_adata = mdata.mod["peptide"].copy()

    summ: PtmSummariser = PtmSummariser(adata=peptide_adata, protein_col=protein_col)
    data: pd.DataFrame = summ.get_data()
    ptm_label_df = summ.label_ptm_site(
        data=data, modification_mass=preset.mass, fasta_file=fasta_file
    )
    ptm_df = summ.summarise_data(data=ptm_label_df, sum_method=sum_method)

    ptm_adata: ad.AnnData = summ.data2adata(data=ptm_df)

    mdata = add_modality(
        mdata=mdata,
        adata=ptm_adata,
        mod_name=f"{modification_name}_site",
        parent_mods=["peptide"],
    )
    mdata.push_obs()
    mdata.update_var()

    return mdata


@dataclass
class PtmPreset:
    name: str | None = field(default=None)
    mass: float | None = field(default=None)

    @classmethod
    def phospho(cls):
        return cls(name="phospho", mass=79.96633)

    @classmethod
    def acetyl(cls):
        return cls(name="acetyl", mass=42.010565)
