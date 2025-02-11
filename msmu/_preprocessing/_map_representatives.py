import mudata as md
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Tuple


def map_representatives(
    mdata: md.MuData,
    modality: str,
    peptide_colname: str = "stripped_peptide",
    protein_colname: str = "proteins_filtered",
    remove_subsumable: bool = True,
) -> md.MuData:
    """
    Map protein information to peptides.

    Args:
        mdata (MuData): MuData object
        modality (str): modality
        peptide_colname (str): column name for peptide information
        protein_colname (str): column name for protein information
        remove_subsumable (bool): whether to remove subsumable proteins

    Returns:
        mdata (MuData): MuData object
    """
    peptide_map, protein_map, protein_info = get_protein_mapping(
        peptides=mdata[modality].var[peptide_colname],
        proteins=mdata[modality].var[protein_colname],
        remove_subsumable=remove_subsumable,
    )

    mdata[modality].uns["peptide_map"] = peptide_map
    mdata[modality].uns["protein_map"] = protein_map
    mdata[modality].uns["protein_info"] = protein_info

    mdata[modality].var["proteins_remapped"] = (
        mdata[modality].var[peptide_colname].map(peptide_map.set_index("peptide").to_dict()["repr_protein"])
    )
    mdata[modality].var["peptide_type"] = [
        "unique" if len(x.split(";")) == 1 else "shared" for x in mdata[modality].var["proteins_remapped"]
    ]

    return mdata


def get_protein_mapping(
    peptides: pd.Series,
    proteins: pd.Series,
    remove_subsumable: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Infer protein grouping information.

    Args:
        peptides (pd.Series): peptide information
        proteins (pd.Series): protein information
        remove_subsumable (bool): whether to remove subsumable proteins

    Returns:
        peptide_map (pd.DataFrame): peptide mapping information
        protein_map (pd.DataFrame): protein mapping information
        protein_info (pd.DataFrame): protein information
    """
    # Initial load
    map_df = _get_map_df(peptides, proteins)
    peptide_df, protein_df = _get_df(map_df)
    initial_protein_df = protein_df.copy()

    # Calculate inclusion, remove subsets and duplicates
    indist_mat, subset_mat = _calculate_inclusion(map_df, peptide_df, protein_df)
    map_df, subset_map = _get_subset_proteins(map_df, protein_df, subset_mat)
    map_df, indist_map = _get_indist_proteins(map_df, protein_df, indist_mat, subset_map)

    # Update protein
    _, protein_df = _get_df(map_df)

    # Map protein groups
    _, _, subsumable_prots = _group_protein(protein_df)

    # Map protein group members
    map_df["protein_group"] = map_df["protein"].map(indist_map["memb"])

    # Get final output
    peptide_map, protein_map, protein_info = _get_final_output(
        map_df=map_df,
        initial_protein_df=initial_protein_df,
        subset_repr_map=subset_map["repr"],
        indist_repr_map=indist_map["repr"],
        subsumable_prots=subsumable_prots,
        remove_subsumable=remove_subsumable,
    )

    return peptide_map, protein_map, protein_info


def _get_map_df(
    peptides: pd.Series,
    proteins: pd.Series,
) -> pd.DataFrame:
    """
    Returns peptide-to-protein mapping information.

    Args:
        peptides (pd.Series): peptide information
        proteins (pd.Series): protein information

    Returns:
        map_df (pd.DataFrame): mapping information
    """
    # Split proteins and explode the DataFrame
    map_df = pd.DataFrame({"protein": proteins, "peptide": peptides})
    map_df["protein"] = map_df["protein"].str.split(";")
    map_df = map_df.explode("protein").drop_duplicates().reset_index(drop=True)

    # Add canonical column
    map_df["protein_canonical"] = map_df["protein"].str.split("|").str[1].str.split("-").str[0]

    return map_df.sort_values("protein").reset_index(drop=True)


def _get_peptide_df(map_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns peptide information.

    Args:
        map_df (pd.DataFrame): mapping information

    Returns:
        peptide_df (pd.DataFrame): peptide information
    """
    # Group by peptide and count the number of proteins
    peptide_df = map_df[["protein", "peptide"]]
    peptide_df = peptide_df.groupby("peptide", as_index=False, observed=False).count()
    peptide_df["is_unique"] = peptide_df["protein"] == 1

    return peptide_df


def _get_protein_df(map_df: pd.DataFrame, peptide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns protein information.

    Args:
        map_df (pd.DataFrame): mapping information
        peptide_df (pd.DataFrame): peptide information

    Returns:
        protein_df (pd.DataFrame): protein information
    """
    PEP_COLS = ["peptide", "is_unique"]
    GROUP_COLS = ["protein"]

    data = map_df.merge(peptide_df[PEP_COLS], on="peptide", how="left")

    # Count shared & unique peptides for each protein
    protein_df = (
        data.groupby(GROUP_COLS, observed=False)
        .agg(shared_peptides=("is_unique", lambda x: (~x).sum()), unique_peptides=("is_unique", "sum"))
        .reset_index()
    )

    return protein_df


def _get_df(map_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns peptide and protein information.

    Args:
        map_df (pd.DataFrame): mapping information

    Returns:
        peptide_df (pd.DataFrame): peptide information
        protein_df (pd.DataFrame): protein
    """
    peptide_df = _get_peptide_df(map_df=map_df)
    protein_df = _get_protein_df(map_df=map_df, peptide_df=peptide_df)

    return peptide_df, protein_df


def _calculate_inclusion(
    map_df: pd.DataFrame,
    peptide_df: pd.DataFrame,
    protein_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate protein group inclusion relationship.

    Args:
        map_df (pd.DataFrame): mapping information
        peptide_df (pd.DataFrame): peptide information
        protein_df (pd.DataFrame): protein information

    Returns:
        indist_mat (np.ndarray): identical protein matrix
        subset_mat (np.ndarray): subset protein matrix
    """
    peptide_index = {pep: idx for idx, pep in enumerate(peptide_df["peptide"])}
    protein_index = {prot: idx for idx, prot in enumerate(protein_df["protein"])}
    coordinates = np.array(
        [
            map_df["protein"].map(protein_index).values,
            map_df["peptide"].map(peptide_index).values,
        ],
        dtype=int,
    )

    # Get inclusion matrix
    matrix = sp.lil_matrix((len(protein_df), len(peptide_df)), dtype=float)
    matrix[coordinates[0], coordinates[1]] = 1
    matrix = matrix.tocsr()

    # Get protein matrix
    prot_mat = matrix.dot(matrix.T).toarray()

    # Get indistinguishable and subset matrix
    inclusion_mat = prot_mat == prot_mat.max(axis=0)
    indist_mat = inclusion_mat & (inclusion_mat.T == inclusion_mat)
    subset_mat = inclusion_mat & ~indist_mat

    return indist_mat, subset_mat


def _get_subset_proteins(
    map_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    subset_mat: np.ndarray,
) -> Tuple[pd.DataFrame, dict]:
    """
    Get subset group information.

    Args:
        map_df (pd.DataFrame): mapping information
        protein_df (pd.DataFrame): protein information
        subset_mat (np.ndarray): subset matrix

    Returns:
        map_df (pd.DataFrame): filtered mapping information
        subset_map (dict): subset map[repr, memb, index]
    """
    subset_repr_map, subset_memb_map, subset_indexset = _get_subset_map(subset_mat, protein_df)
    subset_proteins = protein_df.loc[subset_indexset, "protein"].tolist()
    map_df = map_df[~map_df["protein"].isin(subset_proteins)].copy()
    print(f"Merged subsets:", len(subset_proteins), flush=True)

    return map_df, {"repr": subset_repr_map, "memb": subset_memb_map, "index": subset_indexset}


def _get_subset_map(
    subset_mat: np.ndarray,
    protein_df: pd.DataFrame,
) -> Tuple[dict, dict, list]:
    """
    Returns subset group information.

    Args:
        subset_mat (np.ndarray): subset matrix
        protein_df (pd.DataFrame): protein information

    Returns:
        subset_repr_map (dict): subset representative map
        subset_memb_map (dict): subset member map
        subset_indexset (list): subset
    """
    parent_index, offspring_index = np.where(subset_mat)

    subset_repr_map = {}
    subset_memb_map = {}

    # get subset members
    for parent_idx, offspring_idx in zip(parent_index, offspring_index):
        offspring = protein_df.loc[offspring_idx, "protein"]
        parent = protein_df.loc[parent_idx, "protein"]

        if subset_memb_map.get(parent):
            subset_memb_map[parent] += ";" + offspring
        else:
            subset_memb_map[parent] = offspring

    # get subset representative
    for repr, memb in subset_memb_map.items():
        for m in memb.split(";"):
            if subset_memb_map.get(m):
                if len(subset_memb_map[m].split(";")) < len(memb.split(";")):
                    subset_repr_map[m] = repr
                elif len(subset_memb_map[m].split(";")) == len(memb.split(";")):
                    raise AssertionError(f"The length cannot be the same. {repr} vs {subset_repr_map[m]}")
            else:
                subset_repr_map[m] = repr

    # remove members that are also representatives
    subset_memb_map = {k: subset_memb_map[k] for k in pd.Series(subset_repr_map).unique()}

    return subset_repr_map, subset_memb_map, list(set(offspring_index))


def _get_indist_proteins(
    map_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    indist_mat: np.ndarray,
    subset_map: dict,
) -> Tuple[pd.DataFrame, dict, int]:
    """
    Get indistinguishable group information.

    Args:
        map_df (pd.DataFrame): mapping information
        protein_df (pd.DataFrame): protein information
        indist_mat (np.ndarray): indistinguishable matrix
        subset_map (dict): subset map[repr, memb, index]

    Returns:
        map_df (pd.DataFrame): filtered mapping information
        indist_map (dict): indistinguishable map[repr, memb, index]
    """
    indist_repr_map, indist_memb_map, indist_indexset = _get_indist_map(indist_mat, subset_map["index"], protein_df)
    map_df = map_df.copy()
    map_df["protein"] = map_df["protein"].map(indist_repr_map).fillna(map_df["protein"])
    map_df = map_df[["protein", "peptide"]].drop_duplicates().reset_index(drop=True)
    print(f"Merged indistinguishables:", len(indist_repr_map) - len(indist_memb_map), flush=True)

    return map_df, {"repr": indist_repr_map, "memb": indist_memb_map, "index": indist_indexset}


def _get_indist_map(
    indist_mat: np.ndarray,
    subset_indexset: list,
    protein_df: pd.DataFrame,
) -> Tuple[dict, dict, list, int]:
    """
    Returns indistinguishable group information.

    Args:
        indist_mat (np.ndarray): indistinguishable matrix
        subset_indexset (list): subset index set
        protein_df (pd.DataFrame): protein information

    Returns:
        indist_repr_map (dict): indistinguishable representative map
        indist_memb_map (dict): indistinguishable member map
        indist_indexset (list): indistinguishable index set
        unique_count (int): number of unique groups
    """
    nonsubset_indexset = [idx for idx in range(len(indist_mat)) if idx not in subset_indexset]
    indist_mat = np.delete(indist_mat, subset_indexset, axis=0)
    indist_mat = np.delete(indist_mat, subset_indexset, axis=1)

    # mappings
    indist_repr_map = {}
    indist_memb_map = {}

    unique_groups = np.unique(indist_mat, axis=0)
    identical_groups = unique_groups[unique_groups.sum(axis=1) > 1]
    indist_indexset = []

    for group in identical_groups:
        memb_index = np.take(nonsubset_indexset, np.where(group)[0])
        memb_entry = protein_df.loc[memb_index, "protein"].to_list()

        rep_protein = _select_canon_prot(memb_entry)

        for memb in memb_entry:
            indist_repr_map[memb] = rep_protein

        indist_memb_map[rep_protein] = ";".join(memb_entry)
        indist_indexset += list(memb_index)

    return indist_repr_map, indist_memb_map, indist_indexset


def _select_canon_prot(protein_list: list[str]) -> str:
    """
    Select canonical protein from protein list based on priority.
    canonical > swissprot > trembl > contam

    Args:
        protein_list (list[str]): list of proteins (uniprot entry)

    Returns:
        protein_group (str): canonical protein group
    """
    swissprot_canon_ls = [prot for prot in protein_list if prot.startswith("sp") and "-" not in prot]
    if swissprot_canon_ls:
        return ",".join(swissprot_canon_ls)

    swissprot_ls = [prot for prot in protein_list if prot.startswith("sp")]
    if swissprot_ls:
        return ",".join(swissprot_ls)

    trembl_ls = [prot for prot in protein_list if prot.startswith("tr")]
    if trembl_ls:
        return ",".join(trembl_ls)

    contam_ls = [prot for prot in protein_list if prot.startswith("contam")]
    if contam_ls:
        return ",".join(contam_ls)

    return ""


def _group_protein(protein_df: pd.DataFrame) -> Tuple[list, list, list, list]:
    """
    Group proteins into distinct, distinguishable, identical, and others.

    Args:
        protein_df (pd.DataFrame): DataFrame containing protein information.

    Returns:
        distinct_prots (list): list of distinct proteins
        distinguishable_prots (list): list of distinguishable proteins
        subsumable_prots (list): list of other proteins
    """
    distinct_prots = []
    distinguishable_prots = []
    subsumable_prots = []

    for _, row in protein_df.iterrows():
        if row["shared_peptides"] == 0:
            distinct_prots.append(row["protein"])
        elif row["unique_peptides"] > 0:
            distinguishable_prots.append(row["protein"])
        else:
            subsumable_prots.append(row["protein"])

    print(f"\nMapped proteins", flush=True)
    print(f"- distinct:", len(distinct_prots), flush=True)
    print(f"- distinguishable:", len(distinguishable_prots), flush=True)
    print(f"- subsumable:", len(subsumable_prots), flush=True)

    return distinct_prots, distinguishable_prots, subsumable_prots


def _get_final_output(
    map_df: pd.DataFrame,
    initial_protein_df: pd.DataFrame,
    subset_repr_map: dict,
    indist_repr_map: dict,
    subsumable_prots: list,
    remove_subsumable: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns final output dataframes.

    Args:
        map_df (pd.DataFrame): mapping information
        initial_protein_df (pd.DataFrame): protein information
        subset_repr_map (dict): subset member map
        indist_repr_map (dict): identical representative map
        subsumable_prots (list): list of other proteins
        remove_subsumable (bool): whether to remove subsumable proteins

    Returns:
        peptide_map (pd.DataFrame): peptide mapping information
        protein_map (pd.DataFrame): protein mapping information
        protein_info (pd.DataFrame): protein information
    """
    # Get peptide mapping
    peptide_map = (
        map_df[["peptide", "protein"]]
        .drop_duplicates()
        .groupby("peptide", as_index=False, observed=False)["protein"]
        .agg(lambda x: ";".join(x))
        .rename(columns={"protein": "repr_protein"})
    )

    # Get protein mapping
    subset_prot = initial_protein_df["protein"].map(subset_repr_map)
    indist_prot = initial_protein_df["protein"].map(indist_repr_map)
    repr_prot = indist_prot.fillna(subset_prot).fillna(initial_protein_df["protein"])

    # Get protein type
    type_colnames = [
        "subsetted",
        "indistinguishable",
        "subsumable",
    ]
    protein_map = pd.DataFrame(
        {
            "uniprot_entry": initial_protein_df["protein"],
            "repr_protein": repr_prot,
            "subsetted": subset_prot.notna(),
            "indistinguishable": indist_prot.notna(),
        }
    ).drop_duplicates()
    protein_map["subsumable"] = protein_map["repr_protein"].isin(subsumable_prots)

    protein_map["type"] = (
        protein_map[type_colnames]
        .apply(lambda x: [type_colnames[i] if value else None for i, value in enumerate(x)], axis=1)
        .apply(lambda x: "|".join(filter(None, x)))
    )
    protein_map = protein_map.drop(columns=type_colnames)

    # Get protein information
    protein_info = map_df[["protein", "protein_group"]].drop_duplicates().rename(columns={"protein": "repr_protein"})
    protein_info["protein_group"] = protein_info["protein_group"].fillna(protein_info["repr_protein"])

    # Remove subsumable proteins
    if remove_subsumable:
        peptide_map["repr_protein"] = peptide_map["repr_protein"].apply(
            lambda x: ";".join([prot for prot in x.split(";") if prot not in subsumable_prots])
        )
        protein_info = protein_info[~protein_info["repr_protein"].isin(subsumable_prots)]

    return peptide_map, protein_map, protein_info
