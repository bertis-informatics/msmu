import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

from ..._read_write._readers import add_modality
from ..._utils.utils import get_modality_dict
from ._summariser import PeptideSummariser, ProteinSummariser, PtmSummariser

warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")

def to_peptide(
    mdata:md.MuData,
    rank_method:str | None=None,
    from_: str = "psm",
    sum_method:str="median",
    peptide_col:str="peptide",
    protein_col:str="protein_group",
    top_n:int|None=None,
) -> md.MuData:
    modality_dict:dict[str, ad.AnnData] = get_modality_dict(mdata=mdata, level=from_)

    adata_list: list[ad.AnnData] = list()
    for mod_adata in modality_dict.values():
        summ:PeptideSummariser = PeptideSummariser(adata=mod_adata, peptide_col=peptide_col, protein_col=protein_col, from_=from_)
        data:pd.DataFrame = summ.get_data()

        if top_n is not None:
            data:pd.DataFrame = summ.rank_psm(data=data, method=rank_method)
            data:pd.DataFrame = summ.filter_by_rank(data=data, top_n=top_n)

        summaried_data:pd.DataFrame = summ.summarise_data(data=data, sum_method=sum_method)
        peptide_adata:pd.DataFrame = summ.data2adata(data=summaried_data)
        adata_list.append(peptide_adata)

    merged_peptide_adata:ad.AnnData = ad.concat(adatas=adata_list, join="outer")
    if np.all(np.isnan(merged_peptide_adata.X.astype(np.float64))):  # for lfq
        peptide_arr:pd.DataFrame = mdata["peptide"].to_df().T
        intersect_idx = peptide_arr.index.intersection(merged_peptide_adata.var_names)
        peptide_arr:pd.DataFrame = peptide_arr.loc[intersect_idx].T
        merged_peptide_adata:ad.AnnData = merged_peptide_adata[:, intersect_idx]
        merged_peptide_adata.X = peptide_arr.T

    merged_var:pd.DataFrame = _merged_var_df(adata_list=adata_list, protein_col=protein_col)

    merged_peptide_adata.var = merged_var.loc[merged_peptide_adata.var_names]
    merged_peptide_adata.uns["level"] = "peptide"

    print(merged_peptide_adata)

    mdata:md.MuData = add_modality(mdata=mdata, adata=merged_peptide_adata, mod_name="peptide", parent_mods=modality_dict.keys())
    mdata.push_obs()
    mdata.update_var()

    return mdata


def _merged_var_df(adata_list: list[ad.AnnData], protein_col: str) -> pd.DataFrame:
    merged_var:list[str] = [x.var for x in adata_list]
    merged_var:pd.DataFrame = pd.concat(merged_var, join="outer")
    merged_agg_dict:dict = {
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

    merged_var:pd.DataFrame = merged_var.groupby(merged_var.index, observed=False).agg(merged_agg_dict)
    merged_var:pd.DataFrame = merged_var[~merged_var.index.duplicated(keep="first")]

    return merged_var


def to_protein(mdata, protein_col="protein_group", min_n_peptides=1, sum_method="median", from_="peptide") -> md.MuData:
    modality_dict:dict[str, ad.AnnData] = get_modality_dict(mdata=mdata, level=from_)
    adata:ad.AnnData = modality_dict[from_].copy()

    summ:ProteinSummariser = ProteinSummariser(adata=adata, protein_col=protein_col, from_=from_)
    data:pd.DataFrame = summ.get_data()
    unique_filtered_data:pd.DataFrame = summ.filter_unique_peptides(data=data)
    summarised_data:pd.DataFrame = summ.summarise_data(data=unique_filtered_data, sum_method=sum_method)
    summarised_data:pd.DataFrame = summ.filter_n_min_peptides(data=summarised_data, min_n_peptides=min_n_peptides)

    protein_adata:ad.AnnData = summ.data2adata(data=summarised_data)

    mdata:md.MuData = add_modality(mdata=mdata, adata=protein_adata, mod_name=summ._to, parent_mods=[from_])
    mdata.push_obs()
    mdata.update_var()

    return mdata


def to_ptm_site(
    mdata: md.MuData,
    fasta_file: str | Path,
    modification_name: str,
    modification_mass: float | None = None,
    sum_method: str = "median",
):
    peptide_adata = mdata.mod["peptide"].copy()
    peptide_data = peptide_adata.var.copy()
    peptide_arr = peptide_adata.X.T.copy()

    if modification_mass is None:
        if modification_name == "phospho":
            modification_mass = 79.96633
        else:
            raise ValueError(f"Unknown modification name: {modification_name}. Please provide modification mass.")

    summariser = PtmSummariser(mdata=mdata)
    ptm_label_df = summariser.label_ptm_site(peptide_data, modification_mass, fasta_file)
    ptm_df = summariser.summarise(ptm_data=ptm_label_df, arr=peptide_arr, sum_method=sum_method)

    ptm_adata = ad.AnnData(X=peptide_arr, var=ptm_df)

    mdata = add_modality(
        mdata=mdata,
        adata=ptm_adata,
        mod_name=f"{modification_name}_site",
    )

    raise mdata

# def _summarise_ptm_site(mdata, sum_method):
#     # merge quant data
#     phospho_df = pd.merge(phospho_pep_df, normed_pep_df, on="Peptide", how="left", validate="many_to_one")

#     all_cols = normed_pep_df.columns
#     sample_cols = [col for col in all_cols if col not in SELECTED_COL_CHANNEL]

#     phospho_df = phospho_df[["Phosphosite", "Phosphoprotein", *sample_cols]]
#     phospho_df.loc[:, "Phosphosite"] = phospho_df.Phosphosite.str.split(",")
#     phospho_df.loc[:, "Phosphoprotein"] = phospho_df.Phosphoprotein.str.split(",")

#     phospho_df = phospho_df.explode(["Phosphosite", "Phosphoprotein"], ignore_index=True)
#     phospho_df = phospho_df.groupby(["Phosphosite", "Phosphoprotein"], as_index=False).agg(summarize_values)

#     return phospho_df
