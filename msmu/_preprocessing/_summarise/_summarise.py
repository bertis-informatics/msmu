import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from ..._read_write._readers import add_modality
from .._normalise._normalise import get_modality_dict


def to_peptide(
    mdata,
    rank_method: str = "max_intensity",
    sum_method: str = "median",
    protein_col: str = "protein_group",
    top_n: None | int = None,
):
    psm_dict: dict = get_modality_dict(mdata, level="psm")

    adata_list: list = list()
    for psm_name, psm_adata in psm_dict.items():
        psm_df: pd.DataFrame = psm_adata.to_df().transpose()
        samples: pd.Series = psm_df.columns

        psm_df["peptide"] = psm_adata.var["peptide"].astype(str)
        psm_df["modifications"] = psm_adata.var["modifications"]
        psm_df["stripped_peptide"] = psm_adata.var["stripped_peptide"]
        psm_df[protein_col] = psm_adata.var[protein_col]
        if "proteins_remapped" in psm_adata.var.columns:
            psm_df["proteins_remapped"] = psm_adata.var["proteins_remapped"]
        if "peptide_type" in psm_adata.var.columns:
            psm_df["peptide_type"] = psm_adata.var["peptide_type"]

        peptide_count: pd.DataFrame = (
            psm_df["peptide"].value_counts().to_frame("total_psm")
        )
        psm_df: pd.DataFrame = psm_df.merge(
            peptide_count, left_on="peptide", right_index=True
        )

        if top_n is not None:
            ranked_psm_df: pd.DataFrame = _rank_psm(
                psm_df=psm_df, sample_col=samples, method=rank_method
            )
            top_psm_df: pd.DataFrame = ranked_psm_df[ranked_psm_df["rank"] <= top_n]
        else:
            top_psm_df = psm_df.copy()

        # summarise info
        agg_dict: dict[str, str] = {
            protein_col: "first",
            "stripped_peptide": "first",
            "modifications": "first",
            "total_psm": "first",
            "peptide": "count",
        }
        if "proteins_remapped" in psm_df.columns:
            agg_dict["proteins_remapped"] = "first"
        if "peptide_type" in psm_df.columns:
            agg_dict["peptide_type"] = "first"

        peptide_var = top_psm_df.groupby("peptide", observed=False).agg(agg_dict)

        peptide_var = peptide_var.rename(columns={"peptide": "num_used_psm"})
        peptide_var = peptide_var.rename_axis(index=None)

        # summarise values
        assert sum_method in [
            "median",
            "mean",
        ], f"Unknown sum method: {sum_method}. Please choose from: median"
        
        if np.all(np.isnan(mdata[psm_name].X.astype(np.float64))):  # for lfq
            peptide_arr = mdata["peptide"].to_df().T[samples]
            intersect_idx = peptide_arr.index.intersection(top_psm_df["peptide"])
            peptide_var = peptide_var.loc[intersect_idx]
            peptide_arr = peptide_arr.loc[intersect_idx].T
        else:
            peptide_arr = top_psm_df.groupby("peptide", observed=False)[samples].agg(sum_method).T

        # create adata with summarise info (var) and values (X)
        peptide_adata = ad.AnnData(X=peptide_arr, var=peptide_var)
        adata_list.append(peptide_adata)

    merged_adata = ad.concat(adata_list, join="outer")
    merged_var = [x.var for x in adata_list]
    merged_var = pd.concat(merged_var, join="outer")
    merged_agg_dict = {
        protein_col: "first",
        "stripped_peptide": "first",
        "modifications": "first",
        "total_psm": "sum",
        "num_used_psm": "sum",
    }
    if "proteins_remapped" in merged_var.columns:
        merged_agg_dict["proteins_remapped"] = "first"
    if "peptide_type" in merged_var.columns:
        merged_agg_dict["peptide_type"] = "first"

    merged_var = merged_var.groupby(merged_var.index, observed=False).agg(
        merged_agg_dict
    )
    merged_var = merged_var[~merged_var.index.duplicated(keep="first")]

    merged_adata.var = merged_var.loc[merged_adata.var_names]
    merged_adata.uns["level"] = "peptide"

    mdata = add_modality(
        mdata=mdata,
        adata=merged_adata,
        mod_name="peptide",
        parent_mods=list(psm_dict.keys()),
    )

    mdata.mod["peptide"] = merged_adata

    return mdata


def _rank_psm(psm_df: pd.DataFrame, method: str, sample_col: list[str]) -> pd.DataFrame:
    if method == "max_intensity":
        psm_df.loc[:, "max_intensity"] = psm_df[sample_col].max(axis=1)
        psm_df["rank"] = psm_df.groupby("peptide")["max_intensity"].rank(
            ascending=False
        )
    else:
        raise ValueError(
            f"Unknown rank method: {method}. Please choose from: max_intensity"
        )

    return psm_df


def to_protein(
    mdata: md.MuData,
    use_unique: bool = True,
    min_n_peptides: int = 1,
    sum_method: str = "median",
    level: str = "peptide",
    protein_col: str = "protein_group",
) -> md.MuData:
    level_adata: ad.AnnData = mdata.mod[level]
    level_df: pd.DataFrame = level_adata.to_df().transpose()
    samples: pd.Series = level_df.columns

    level_df[protein_col] = level_adata.var[protein_col]
    level_df["stripped_peptide"] = level_adata.var["stripped_peptide"]
    if "proteins_remapped" in level_adata.var.columns:
        level_df["proteins_remapped"] = level_adata.var["proteins_remapped"]
    if "peptide_type" in level_adata.var.columns:
        level_df["peptide_type"] = level_adata.var["peptide_type"]

    # make level_df(parent) and agg method dict for protein summarisation
    if level == "peptide":
        level_df["peptide"] = level_adata.var.index
        level_df["total_psm"] = level_adata.var["total_psm"]
        level_df["num_used_psm"] = level_adata.var["num_used_psm"]

        agg_dict: dict[str, str] = {
            "proteins_remapped": "first",
            "total_psm": "sum",
            "num_used_psm": "sum",
            "stripped_peptide": "nunique",
        }
        if "proteins_remapped" in level_df.columns:
            agg_dict["proteins_remapped"] = "first"

        rename_dict: dict[str, str] = {"stripped_peptide": "num_peptides"}

    elif level == "psm":
        level_df["peptide"] = level_adata.var["peptide"]
        level_df["total_psm"] = level_adata.var["total_psm"]
        level_df["num_used_psm"] = level_adata.var["num_used_psm"]

        agg_dict: dict[str, str] = {
            "total_psm": "sum",
            "num_used_psm": "sum",
            "stripped_peptide": "nunique",
        }
        rename_dict: dict[str, str] = {"stripped_peptide": "num_peptides"}
    else:
        raise ValueError(f"Unknown level: {level}. Please choose from: peptide, psm")

    # filter unique peptides
    if use_unique:
        if "peptide_type" in level_df.columns:
            level_df: pd.DataFrame = level_df[level_df["peptide_type"] == "unique"]
        else:
            level_df: pd.DataFrame = level_df[
                len(level_df["num_used_psm"].str.split(";")) == 1
            ]

    # summarise protein level
    protein_var: pd.DataFrame = level_df.groupby(protein_col, observed=False).agg(
        agg_dict
    )

    # rename columns
    protein_var = protein_var.rename(columns=rename_dict)
    # filter proteins with min_n_peptides
    protein_var = protein_var[protein_var["num_peptides"] >= min_n_peptides]
    # rename index
    protein_var = protein_var.rename_axis(index=None)

    # summarise values
    protein_arr = level_df.groupby(protein_col, observed=False)[samples].agg(sum_method)
    protein_arr = protein_arr.loc[protein_var.index]
    protein_arr = protein_arr.transpose()

    # create adata with summarise info (var) and values (X)
    protein_adata: ad.AnnData = ad.AnnData(X=protein_arr, var=protein_var)
    protein_adata.uns["level"] = "protein"
    parent_mods: list[str] = list(get_modality_dict(mdata=mdata, level=level).keys())

    # add protein modality to mdata
    mdata = add_modality(
        mdata=mdata,
        adata=protein_adata,
        mod_name="protein",
        parent_mods=parent_mods,
    )

    return mdata


def to_ptm_site(mdata, modifications, fasta_file, sum_method):
    # fasta_dict = read_fasta(fasta_file)


    raise NotImplementedError
