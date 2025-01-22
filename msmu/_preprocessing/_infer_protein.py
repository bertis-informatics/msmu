import mudata as md
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Tuple


def map_protein(
    mdata: md.MuData,
    modality: str,
    peptide_colname: str = "stripped_peptide",
    protein_colname: str = "proteins_wo_decoy",
) -> md.MuData:
    """
    Map protein information to peptides.

    Args:
        mdata (MuData): MuData object
        modality (str): modality

    Returns:
        mdata (MuData): MuData object
    """
    peptide_map, protein_map, protein_info = get_protein_mapping(
        peptides=mdata[modality].var[peptide_colname],
        proteins=mdata[modality].var[protein_colname],
    )

    mdata[modality].uns["peptide_map"] = peptide_map
    mdata[modality].uns["protein_map"] = protein_map
    mdata[modality].uns["protein_info"] = protein_info

    return mdata


def get_protein_mapping(
    peptides: pd.Series,
    proteins: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Infer protein grouping information.

    Args:
        peptides (pd.Series): peptide information
        proteins (pd.Series): protein information

    Returns:
        peptide_map (pd.DataFrame): peptide mapping information
        protein_map (pd.DataFrame): protein mapping information
        protein_info (pd.DataFrame): protein information
    """
    # Initial load
    map_df = _get_map_df(peptides, proteins)
    peptide_df, protein_df = _get_df(map_df, canonical=True)
    initial_protein_df = protein_df.copy()

    # Remove subsumables
    map_df, subsum_map = _remove_subsumables(map_df, protein_df)
    peptide_df, protein_df = _get_df(map_df, canonical=True)

    # Calculate inclusion, remove subsets and duplicates
    identical_mat, subset_mat = _calculate_inclusion(map_df, peptide_df, protein_df)
    map_df, subset_map = _remove_subsets(map_df, protein_df, subset_mat)
    map_df, duplicate_map, unique_cnt = _remove_duplicates(map_df, protein_df, identical_mat, subset_map)

    # Update peptide & protein
    peptide_df, protein_df = _get_df(map_df, canonical=False)
    assert unique_cnt == len(protein_df), f"Filtered protein count does not match."

    # Map protein groups
    protein_df["protein_identical"] = protein_df["protein"].map(duplicate_map["memb"])
    distinct_prots, distinguishable_prots, identical_prots, others_prots = _group_protein(protein_df)

    # Map protein group members
    protein_group_map = {**subsum_map["memb"], **duplicate_map["memb"]}
    map_df["protein_group"] = map_df["protein"].map(protein_group_map)

    # Get final output
    peptide_map, protein_map, protein_info = _get_final_output(
        map_df=map_df,
        protein_df=initial_protein_df,
        subsum_repr_map=subsum_map["repr"],
        subset_memb_map=subset_map["memb"],
        identical_repr_map=duplicate_map["repr"],
        identical_prots=identical_prots,
        others_prots=others_prots,
    )

    assert len(peptide_df) == len(peptide_map), "Peptide mapping count does not match."
    assert len(protein_info) == len(protein_df), "Protein information count does not match."
    assert len(peptide_map) == len(peptide_df), "Peptide mapping count does not match."

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


def _get_peptide_df(
    map_df: pd.DataFrame,
    canonical: bool,
) -> pd.DataFrame:
    """
    Returns peptide information.

    Args:
        map_df (pd.DataFrame): mapping information
        canonical (bool): whether to include canonical uniqueness

    Returns:
        peptide_df (pd.DataFrame): peptide information
    """
    # Group by peptide and count the number of proteins
    peptide_df = map_df[["protein", "peptide"]]
    peptide_df = peptide_df.groupby("peptide", as_index=False, observed=False).count()
    peptide_df["is_unique"] = peptide_df["protein"] == 1

    if canonical:
        peptide_canonical_df = map_df[["protein_canonical", "peptide"]].drop_duplicates()
        peptide_canonical_df = peptide_canonical_df.groupby("peptide", as_index=False, observed=False).count()
        peptide_df["is_unique_canonical"] = peptide_canonical_df["protein_canonical"] == 1

    return peptide_df


def _get_protein_df(
    map_df: pd.DataFrame,
    peptide_df: pd.DataFrame,
    canonical: bool,
) -> pd.DataFrame:
    """
    Returns protein information.

    Args:
        map_df (pd.DataFrame): mapping information
        peptide_df (pd.DataFrame): peptide information
        canonical (bool): whether to include canonical uniqueness

    Returns:
        protein_df (pd.DataFrame): protein information
    """
    if canonical:
        PEP_COLS = ["peptide", "is_unique", "is_unique_canonical"]
        GROUP_COLS = ["protein", "protein_canonical"]
    else:
        PEP_COLS = ["peptide", "is_unique"]
        GROUP_COLS = ["protein"]

    data = map_df.merge(peptide_df[PEP_COLS], on="peptide", how="left")

    # Count shared & unique peptides for each protein
    protein_df = (
        data.groupby(GROUP_COLS, observed=False)
        .agg(shared_peptides=("is_unique", lambda x: (~x).sum()), unique_peptides=("is_unique", "sum"))
        .reset_index()
    )

    # Count shared & unique peptides for each canonical protein
    if canonical:
        protein_canonical_df = (
            data.groupby("protein", observed=False)
            .agg(
                shared_peptides_canonical=("is_unique_canonical", lambda x: (~x).sum()),
                unique_peptides_canonical=("is_unique_canonical", "sum"),
            )
            .reset_index()
        )
        protein_df = protein_df.merge(protein_canonical_df, on="protein", how="left", validate="one_to_one")

    return protein_df


def _get_df(
    map_df: pd.DataFrame,
    canonical: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns peptide and protein information.

    Args:
        map_df (pd.DataFrame): mapping information
        canonical (bool): whether to include canonical uniqueness

    Returns:
        peptide_df (pd.DataFrame): peptide information
        protein_df (pd.DataFrame): protein
    """
    peptide_df = _get_peptide_df(map_df=map_df, canonical=canonical)
    protein_df = _get_protein_df(map_df=map_df, peptide_df=peptide_df, canonical=canonical)

    return peptide_df, protein_df


def _remove_subsumables(
    map_df: pd.DataFrame,
    protein_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """
    Remove subsumable groups.

    Args:
        map_df (pd.DataFrame): mapping information
        protein_df (pd.DataFrame): protein information

    Returns:
        map_df (pd.DataFrame): filtered mapping information
        subsum_map (dict): subsumable map[representative, member]

    """
    subsum_repr_map, subsum_memb_map = _get_subsumable_canonicals(protein_df)

    map_df = map_df.copy()
    map_df["protein"] = map_df["protein"].map(subsum_repr_map).fillna(map_df["protein"])
    map_df = map_df.drop_duplicates().reset_index(drop=True)
    print(f"Removed subsumables:", len(subsum_repr_map) - len(subsum_memb_map), flush=True)

    return map_df, {"repr": subsum_repr_map, "memb": subsum_memb_map}


def _get_subsumable_canonicals(protein_df: pd.DataFrame) -> Tuple[dict, dict]:
    """
    Returns subsumable group information.

    Args:
        protein_df (pd.DataFrame): protein information

    Returns:
        subsum_repr_map (dict): subsumable representative protein map
        subsum_memb_map (dict): subsumable member proteins map
    """
    subsumable_index = (protein_df["unique_peptides"] == 0) & (protein_df["unique_peptides_canonical"] != 0)

    subsum_repr_map = {}
    subsum_memb_map = {}
    subsumable_df = protein_df[subsumable_index]

    for protein_canonical, data in subsumable_df.groupby("protein_canonical", observed=False):
        protein_list = data["protein"].tolist()
        canonicals = [prot for prot in protein_list if "-" not in prot]
        repr_prots = canonicals[0] if len(canonicals) == 1 else ";".join(protein_list)

        # Push to member map and representative map
        subsum_memb_map[repr_prots] = ";".join(protein_list)
        subsum_repr_map.update({prot: repr_prots for prot in protein_list})

    return subsum_repr_map, subsum_memb_map


def _calculate_inclusion(
    map_df: pd.DataFrame,
    peptide_df: pd.DataFrame,
    protein_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate protein group inclusion relationship.

    Args:
        new_map_df (pd.DataFrame): mapping information
        peptide_df (pd.DataFrame): peptide information
        protein_df (pd.DataFrame): protein information

    Returns:
        identical_mat (np.ndarray): identical protein matrix
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

    # peptide-protein mapping matrix
    matrix = sp.lil_matrix((len(protein_df), len(peptide_df)), dtype=float)
    matrix[coordinates[0], coordinates[1]] = 1
    matrix = matrix.tocsr()

    # get inclusion matrix
    prot_mat = matrix.dot(matrix.T).toarray()

    # inclusion solution
    inclusion_mat = prot_mat == prot_mat.max(axis=0)
    identical_mat = inclusion_mat & (inclusion_mat.T == inclusion_mat)
    subset_mat = inclusion_mat & ~identical_mat

    return identical_mat, subset_mat


def _remove_subsets(
    map_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    subset_mat: np.ndarray,
) -> Tuple[pd.DataFrame, dict]:
    """
    Remove subset groups.

    Args:
        map_df (pd.DataFrame): mapping information
        protein_df (pd.DataFrame): protein information
        subset_mat (np.ndarray): subset matrix

    Returns:
        map_df (pd.DataFrame): filtered mapping information
        subset_map (dict): subset map[representative, member, index]
    """
    map_df = map_df.copy()
    subset_repr_map, subset_memb_map, subset_indexset = _get_subset_map(subset_mat, protein_df)
    subset_proteins = protein_df.loc[subset_indexset, "protein"].tolist()
    map_df = map_df[~map_df["protein"].isin(subset_proteins)].copy()
    print(f"Removed subsets:", len(subset_proteins), flush=True)

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
    # for index, value in enumerate(parent_index):
    for parent_idx, offspring_idx in zip(parent_index, offspring_index):
        offspring = protein_df.loc[offspring_idx, "protein"]
        parent = protein_df.loc[parent_idx, "protein"]

        subset = subset_repr_map.get(parent)
        if subset is None:
            subset = ""
        else:
            subset += ";"

        superset = subset_memb_map.get(offspring)
        if superset is None:
            superset = ""
        else:
            superset += ";"
        subset_repr_map[parent] = subset + offspring
        subset_memb_map[offspring] = superset + parent

    return subset_repr_map, subset_memb_map, list(set(offspring_index))


def _remove_duplicates(
    map_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    identical_mat: np.ndarray,
    subset_map: dict,
) -> Tuple[pd.DataFrame, dict, int]:
    """
    Remove duplicate groups.

    Args:
        map_df (pd.DataFrame): mapping information
        protein_df (pd.DataFrame): protein information
        subset_map (dict): subset map
        identical_mat (np.ndarray): identical matrix

    Returns:
        map_df (pd.DataFrame): filtered mapping information
        duplicate_map (dict): duplicate map[representative, member]
        unique_count (int): number of unique groups
    """
    identical_repr_map, identical_memb_map, identical_indexset, unique_count = _get_identical_map(
        identical_mat, subset_map["index"], protein_df
    )
    map_df = map_df.copy()
    map_df["protein"] = map_df["protein"].map(identical_repr_map).fillna(map_df["protein"])
    map_df = map_df[["protein", "peptide"]].drop_duplicates().reset_index(drop=True)
    print(f"Removed duplicates:", len(identical_repr_map) - len(identical_memb_map), flush=True)

    return map_df, {"repr": identical_repr_map, "memb": identical_memb_map, "index": identical_indexset}, unique_count


def _get_identical_map(
    identical_mat: np.ndarray,
    subset_indexset: list,
    protein_df: pd.DataFrame,
) -> Tuple[dict, dict, list, int]:
    """
    Returns identical group information.

    Args:
        identical_mat (np.ndarray): identical matrix
        subset_indexset (list): subset index set
        protein_df (pd.DataFrame): protein information

    Returns:
        identical_repr_map (dict): identical representative map
        identical_memb_map (dict): identical member map
        identical_indexset (list): identical index set
        unique_count (int): number of unique groups
    """
    nonsubset_indexset = [idx for idx in range(len(identical_mat)) if idx not in subset_indexset]
    identical_mat = np.delete(identical_mat, subset_indexset, axis=0)
    identical_mat = np.delete(identical_mat, subset_indexset, axis=1)

    # mappings
    identical_repr_map = {}
    identical_memb_map = {}

    unique_groups = np.unique(identical_mat, axis=0)
    identical_groups = unique_groups[unique_groups.sum(axis=1) > 1]
    identical_indexset = []

    for group in identical_groups:
        memb_index = np.take(nonsubset_indexset, np.where(group)[0])
        memb_entry = protein_df.loc[memb_index, "protein"].to_list()

        rep_protein = select_canon_prot(memb_entry)

        for memb in memb_entry:
            identical_repr_map[memb] = rep_protein

        identical_indexset += list(memb_index)
        identical_memb_map[rep_protein] = ";".join(memb_entry)

    return identical_repr_map, identical_memb_map, identical_indexset, len(unique_groups)


def select_canon_prot(prot_ls: list) -> str:
    """
    Select canonical protein from protein list based on priority.
    canonical > swissprot > trembl > contam

    Args:
        prot_ls (list): list of proteins (uniprot entry)

    Returns:
        protein_group (str): canonical protein group
    """
    swissprot_canon_ls = [prot for prot in prot_ls if prot.startswith("sp") and "-" not in prot]
    if swissprot_canon_ls:
        return ";".join(swissprot_canon_ls)

    swissprot_ls = [prot for prot in prot_ls if prot.startswith("sp")]
    if swissprot_ls:
        return ";".join(swissprot_ls)

    trembl_ls = [prot for prot in prot_ls if prot.startswith("tr")]
    if trembl_ls:
        return ";".join(trembl_ls)

    contam_ls = [prot for prot in prot_ls if prot.startswith("contam")]
    if contam_ls:
        return ";".join(contam_ls)

    return ""


def _group_protein(protein_df: pd.DataFrame) -> Tuple[list, list, list, list]:
    """
    Group proteins into distinct, distinguishable, identical, and others.

    Args:
        protein_df (pd.DataFrame): DataFrame containing protein information.

    Returns:
        distinct_prots (list): list of distinct proteins
        distinguishable_prots (list): list of distinguishable proteins
        identical_prots (list): list of identical proteins
        others_prots (list): list of other proteins
    """
    distinct_prots = []
    distinguishable_prots = []
    identical_prots = []
    others_prots = []

    for _, row in protein_df.iterrows():
        if row["shared_peptides"] == 0:
            distinct_prots.append(row["protein"])
        elif row["unique_peptides"] > 0:
            distinguishable_prots.append(row["protein"])
        elif pd.notna(row["protein_identical"]):
            identical_prots.append(row["protein"])
        else:
            others_prots.append(row["protein"])

    print(f"Mapped proteins", flush=True)
    print(f"- distinct:", len(distinct_prots), flush=True)
    print(f"- distinguishable:", len(distinguishable_prots), flush=True)
    print(f"- identical:", len(identical_prots), flush=True)
    print(f"- others:", len(others_prots), flush=True)

    return distinct_prots, distinguishable_prots, identical_prots, others_prots


def _get_final_output(
    map_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    subsum_repr_map: dict,
    subset_memb_map: dict,
    identical_repr_map: dict,
    identical_prots: list,
    others_prots: list,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns final output dataframes.

    Args:
        map_df (pd.DataFrame): mapping information
        protein_df (pd.DataFrame): protein information
        subsum_repr_map (dict): subsumable representative map
        subset_memb_map (dict): subset member map
        identical_repr_map (dict): identical representative map
        identical_prots (list): list of identical proteins
        others_prots (list): list of other proteins

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
        .agg(lambda x: ",".join(x))
    )

    # Get protein mapping
    grouped_prot = protein_df["protein"].map(subsum_repr_map).fillna(protein_df["protein"])  # Canonical
    subset_prot = grouped_prot.map(subset_memb_map)  # Expand to subset members
    identical_prot = grouped_prot.map(identical_repr_map)  # Expand to identical members
    final_prot = identical_prot.fillna(subset_prot).fillna(grouped_prot)

    protein_map = pd.DataFrame(
        {
            "uniProt_entry": protein_df["protein"],
            "protein": final_prot,
            "is_subset": subset_prot.notna(),
            "is_identical": identical_prot.notna(),
        }
    ).drop_duplicates()

    # Get protein information
    protein_info = map_df[["protein", "protein_group"]].drop_duplicates()
    protein_info["is_identical"] = protein_info["protein"].isin(identical_prots)
    protein_info["is_subsumable"] = protein_info["protein"].isin(others_prots)

    return peptide_map, protein_map, protein_info


def map_protein_info(rep_prot: str, fasta_dict: dict) -> str:
    """
    Map protein information from the FASTA data.

    Args:
        rep_prot (str): Representative protein string.
        fasta_dict (dict): Dictionary containing protein information.

    Returns:
        mapped_protein (str): Mapped protein information.
    """
    result = set()
    for prot in rep_prot.split(";"):
        prot_id = prot.split("|")[1]
        prot_info = fasta_dict.get(prot_id)
        if prot_info:
            result.add(prot_info)

    return ";".join(result)
