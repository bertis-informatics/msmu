import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from .._normalise._normalise import get_modality_dict


def to_peptide(
    mdata,
    rank_method: str = "max_intensity",
    sum_method: str = "median",
    top_n: None | int = None,
):
    psm_dict: dict = get_modality_dict(mdata, level="psm")

    adata_list: list = list()
    for psm_name, psm_adata in psm_dict.items():
        psm_df: pd.DataFrame = psm_adata.to_df().transpose()
        samples: pd.Series = psm_df.columns

        psm_df["peptide"] = psm_adata.var["peptide"]
        psm_df["modifications"] = psm_adata.var["modifications"]
        psm_df["stripped_peptide"] = psm_adata.var["stripped_peptide"]
        psm_df["proteins"] = psm_adata.var["proteins"]

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
        peptide_var = top_psm_df.groupby("peptide").agg(
            {
                "proteins": "first",
                "stripped_peptide": "first",
                "modifications": "first",
                "total_psm": "first",
                "peptide": "count",
            }
        )
        peptide_var = peptide_var.rename(columns={"peptide": "num_used_psm"})
        peptide_var = peptide_var.rename_axis(index=None)

        peptide_var = _label_unique_peptide(peptide_var)

        # summarise values
        assert sum_method in [
            "median"
        ], f"Unknown sum method: {sum_method}. Please choose from: median"
        peptide_arr = top_psm_df.groupby("peptide")[samples].agg(sum_method).T

        if np.all(np.isnan(peptide_arr)):  # for lfq
            peptide_arr = mdata["peptide"].to_df().T[samples]
            peptide_arr = peptide_arr.loc[peptide_var.index]

        # create adata with summarise info (var) and values (X)
        peptide_adata = ad.AnnData(X=peptide_arr, var=peptide_var)
        adata_list.append(peptide_adata)

    merged_adata = ad.concat(adata_list, join="outer")
    merged_var = [x.var for x in adata_list]
    merged_var = pd.concat(merged_var, join="outer")
    merged_var = merged_var.groupby(merged_var.index).agg(
        {
            "proteins": "first",
            "stripped_peptide": "first",
            "modifications": "first",
            "total_psm": "sum",
            "num_used_psm": "sum",
            "unique": "first",
        }
    )
    merged_var = merged_var[~merged_var.index.duplicated(keep="first")]

    merged_adata.var = merged_var.loc[merged_adata.var_names]
    merged_adata.uns["level"] = "peptide"

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


def _label_unique_peptide(peptide_df: pd.DataFrame) -> ad.AnnData:
    peptide_df["num_proteins"] = peptide_df["proteins"].str.split(";").apply(len)
    peptide_df["unique"] = peptide_df["num_proteins"] == 1

    return peptide_df


def to_protein(
    mdata: md.MuData,
    use_unique: bool = True,
    min_n_peptides: int = 1,
    sum_method: str = "median",
) -> md.MuData:
    peptide_adata: ad.AnnData = mdata.mod["peptide"]
    peptide_df: pd.DataFrame = peptide_adata.to_df().transpose()
    samples: pd.Series = peptide_df.columns

    peptide_df["proteins"] = peptide_adata.var["proteins"]
    peptide_df["peptide"] = peptide_adata.var.index
    peptide_df["stripped_peptide"] = peptide_adata.var["stripped_peptide"]
    peptide_df["total_psm"] = peptide_adata.var["total_psm"]
    peptide_df["num_used_psm"] = peptide_adata.var["num_used_psm"]
    peptide_df["unique"] = peptide_adata.var["unique"]

    if use_unique:
        peptide_df: pd.DataFrame = peptide_df[peptide_df["unique"]]

    protein_var: pd.DataFrame = peptide_df.groupby("proteins").agg(
        {
            "peptide": "count",
            "total_psm": "sum",
            "num_used_psm": "sum",
        }
    )

    protein_var = protein_var.rename(columns={"peptide": "num_peptides"})
    protein_var = protein_var[protein_var["num_peptides"] >= min_n_peptides]
    protein_var = protein_var.rename_axis(index=None)

    protein_arr: pd.DataFrame = peptide_df.loc[
        peptide_df["proteins"].isin(protein_var.index)
    ]
    protein_arr = protein_arr.groupby("proteins")[samples].agg(sum_method).T

    protein_adata: ad.AnnData = ad.AnnData(X=protein_arr, var=protein_var)
    protein_adata.uns["level"] = "protein"

    mdata.mod["protein"] = protein_adata

    return mdata


def to_ptm_site():
    raise NotImplementedError
