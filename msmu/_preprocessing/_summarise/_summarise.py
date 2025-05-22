import warnings
from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from ..._read_write._readers import add_modality
from ..._utils.utils import get_modality_dict, uns_logger
from ._summariser import PeptideSummariser, ProteinSummariser, PtmSummariser

warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
warnings.filterwarnings(action="ignore", message="Mean of empty slice")


@uns_logger
def to_peptide(
    mdata: md.MuData,
    rank_method: str | None = None,
    from_: str = "psm",
    sum_method: str = "median",
    protein_col: str = "protein_group",
    top_n: int | None = None,
    keep_mbr_only: bool = False,
) -> md.MuData:
    """
    Summarise peptide (or precursor; for lfq) level data from PSM level data.

    Args:
        mdata (md.MuData): MuData object containing PSM level data.
        rank_method (str | None): Method to rank PSMs. If None, no ranking is applied.
        from_ (str): Level to summarise from. Default is "psm".
        sum_method (str): Method to summarise quantification of PSMs. Default is "median".
        protein_col (str): Column name for protein groups. Default is "protein_group" which is from msmu protein inference.
        top_n (int | None): Number of top PSMs to keep after ranking. If None, all PSMs are kept.
        keep_mbr_only (bool): For LFQ, MBR can be used for quantifiction. If True, keep precursor quantity without MSMS evidence. Default is False.

    Returns:
        md.MuData: MuData object containing peptide level data.
    """
    modality_dict: dict[str, ad.AnnData] = get_modality_dict(mdata=mdata, level=from_)
    label_list: list = list(set([x.uns["label"] for x in modality_dict.values()]))
    if len(label_list) > 1:
        raise ValueError("Multiple labels found in the input data. Please provide a single label.")
    else:
        label = label_list[0]

    # assign peptide_col for groupby (summarisation) as "peptide" for tmt and "precursor" for lfq
    # in lfq, column "precursor" is made from reader class with "peptide" + "." + charge
    peptide_col = "precursor" if label == "lfq" else "peptide"

    adata_list: list[ad.AnnData] = list()
    for mod_adata in modality_dict.values():
        summ: PeptideSummariser = PeptideSummariser(
            adata=mod_adata,
            peptide_col=peptide_col,
            protein_col=protein_col,
            from_=from_,
        )
        data: pd.DataFrame = summ.get_data()

        # get top n PSMs for each peptide
        if top_n is not None:
            data: pd.DataFrame = summ.rank_(data=data, rank_method=rank_method)
            data: pd.DataFrame = summ.filter_by_rank(data=data, top_n=top_n)

        summarised_data: pd.DataFrame = summ.summarise_data(data=data, sum_method=sum_method)
        peptide_adata: pd.DataFrame = summ.data2adata(data=summarised_data)
        adata_list.append(peptide_adata)

    merged_peptide_adata: ad.AnnData = ad.concat(adatas=adata_list, join="outer")

    # TODO: make module for lfq summarisation
    if label == "lfq":
        peptide_arr: pd.DataFrame = mdata["peptide"].to_df().T.copy()
        intersect_idx = peptide_arr.index.intersection(merged_peptide_adata.var_names)
        peptide_arr: pd.DataFrame = peptide_arr.loc[intersect_idx].T
        merged_peptide_adata: ad.AnnData = merged_peptide_adata[:, intersect_idx]
        merged_peptide_adata.X = peptide_arr.astype(float)

        # make msms_evidence dataframe
        msms_evidence_list: list[pd.DataFrame] = list()
        for mod_adata in modality_dict.values():
            tmp_evidence: pd.DataFrame = mod_adata.var[["precursor", "filename"]]
            msms_evidence_list.append(tmp_evidence)

        msms_evidence: pd.DataFrame = pd.concat(msms_evidence_list, axis=0)
        msms_evidence["evidence"] = 1
        msms_evidence = msms_evidence.groupby(["precursor", "filename"], as_index=False, observed=False).agg("sum")
        msms_evidence = msms_evidence.pivot(index="precursor", columns="filename", values="evidence")
        msms_evidence = msms_evidence.rename_axis(index=None, columns=None)

        rename_dict: dict = {filename: sample for filename, sample in zip(mdata.obs["tag"], mdata.obs_names)}
        msms_evidence = msms_evidence.rename(columns=rename_dict)
        msms_evidence = msms_evidence.notna().astype(int).replace({0: np.nan})

        msms_evidence = msms_evidence.loc[merged_peptide_adata.var_names, merged_peptide_adata.obs_names]

        # assign msms_evidence to a layer of adata (merged_peptide_adata)
        merged_peptide_adata.layers["msms_evidence"] = msms_evidence.values.T

        # filter out quantity without MSMS evidence if keep_mbr_only is False
        if keep_mbr_only == False:
            merged_peptide_adata.X = np.multiply(merged_peptide_adata.X, msms_evidence.values.T).astype(float)

    # make merged_var dataframe
    merged_var: pd.DataFrame = _merged_var_df(adata_list=adata_list, protein_col=protein_col)

    merged_peptide_adata.var = merged_var.loc[merged_peptide_adata.var_names]
    merged_peptide_adata.uns["level"] = "peptide"

    # add peptide_adata to mdata
    pep_summ_mdata: md.MuData = add_modality(
        mdata=mdata,
        adata=merged_peptide_adata,
        mod_name="peptide",
        parent_mods=list(modality_dict.keys()),
    )
    pep_summ_mdata.push_obs()
    pep_summ_mdata.update_var()

    return pep_summ_mdata


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

    merged_var: pd.DataFrame = merged_var.groupby(merged_var.index, observed=False).agg(merged_agg_dict)
    merged_var: pd.DataFrame = merged_var[~merged_var.index.duplicated(keep="first")]

    return merged_var


@uns_logger
def to_protein(
    mdata,
    top_n: int | None = None,
    rank_method: str = "",
    protein_col="protein_group",
    min_n_peptides=1,
    sum_method="median",
    from_="peptide",
    # keep_mbr_only: bool = False,  # TODO: add keep_mbr_only to protein level
) -> md.MuData:
    """
    Summarise protein level data from peptide level data.

    Args:
        mdata (md.MuData): MuData object containing peptide level data.
        top_n (int | None): Number of top peptides to keep after ranking. If None, all peptides are kept.
        rank_method (str): Method to rank peptides. Default is "" (no ranking).
        protein_col (str): Column name for protein groups. Default is "protein_group" which is from msmu protein inference.
        min_n_peptides (int): Minimum number of peptides required to keep a protein. Default is 1.
        sum_method (str): Method to summarise quantification of peptides. Default is "median".
        from_ (str): Level to summarise from. Default is "peptide".

    Returns:
        md.MuData: MuData object containing protein level data.
    """
    modality_dict: dict[str, ad.AnnData] = get_modality_dict(mdata=mdata, level=from_)
    adata: ad.AnnData = modality_dict[from_].copy()

    summ: ProteinSummariser = ProteinSummariser(adata=adata, protein_col=protein_col, from_=from_)
    data: pd.DataFrame = summ.get_data()
    unique_filtered_data: pd.DataFrame = summ.filter_unique_peptides(data=data)

    # get top n peptides for each protein
    if top_n is not None:
        unique_filtered_data: pd.DataFrame = summ.rank_(data=unique_filtered_data, rank_method=rank_method)
        unique_filtered_data: pd.DataFrame = summ.filter_by_rank(data=unique_filtered_data, top_n=top_n)

    # summarise data
    summarised_data: pd.DataFrame = summ.summarise_data(data=unique_filtered_data, sum_method=sum_method)

    # filter out proteins with less than min_n_peptides
    summarised_data: pd.DataFrame = summ.filter_n_min_peptides(data=summarised_data, min_n_peptides=min_n_peptides)

    protein_adata: ad.AnnData = summ.data2adata(data=summarised_data)

    protein_summed_mdata: md.MuData = add_modality(
        mdata=mdata, adata=protein_adata, mod_name=summ._to, parent_mods=[from_]
    )
    protein_summed_mdata.push_obs()
    protein_summed_mdata.update_var()

    return protein_summed_mdata


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

    Args:
        mdata (md.MuData): MuData object containing peptide level data.
        protein_col (str): Column name for protein groups. Default is "protein_group" which is from msmu protein inference.
        fasta_file (str | Path): Path to the FASTA file for protein sequences used for protein site labeling.
        modification_name (str): Name of the PTM modification (e.g. phospho, acetyl).
        modification_mass (float | None): Mass of the PTM modification. If None, default mass will be used.
        sum_method (str): Method to summarise quantification of peptides. Default is "median".

    Returns:
        md.MuData: MuData object containing PTM site level data. Modality name will be "{modification_name}_site".
    """

    if modification_mass is None:
        if modification_name in PtmPreset.__dict__:
            preset = getattr(PtmPreset, modification_name)()
            modification_mass = preset.mass
        else:
            raise ValueError(f"Unknown modification name: {modification_name}. Please provide modification mass.")
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
            raise ValueError(f"Unknown modification name: {modification_name}. Please provide modification mass.")

    peptide_adata = mdata.mod["peptide"].copy()

    summ: PtmSummariser = PtmSummariser(adata=peptide_adata, protein_col=protein_col)
    data: pd.DataFrame = summ.get_data()
    ptm_label_df = summ.label_ptm_site(data=data, modification_mass=preset.mass, fasta_file=fasta_file)
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
