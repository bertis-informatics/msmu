from typing import Tuple, TypedDict, Union
from collections import deque
import warnings

import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
import mudata as md

from .._utils import get_modality_dict, uns_logger


class Mapping(TypedDict):
    repr: dict[str, str]
    memb: dict[str, str]


def map_representatives(
    mdata: md.MuData,
    modality: Union[str, None] = None,
    level: Union[str, None] = "psm",
    peptide_colname: str = "stripped_peptide",
    protein_colname: str = "proteins",
) -> md.MuData:
    """
    DEPRECATED: Use `infer_protein` instead.

    Map protein information to peptides.

    Args:
        mdata (MuData): MuData object
        modality (str): modality
        peptide_colname (str): column name for peptide information
        protein_colname (str): column name for protein information

    Returns:
        mdata (MuData): MuData object with updated protein mappings
    """

    warnings.warn(
        "map_representatives is deprecated. Use infer_protein instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return infer_protein(
        mdata=mdata,
        modality=modality,
        level=level,
        peptide_colname=peptide_colname,
        protein_colname=protein_colname,
    )


@uns_logger
def infer_protein(
    mdata: md.MuData,
    modality: Union[str, None] = None,
    level: Union[str, None] = "psm",
    peptide_colname: str = "stripped_peptide",
    protein_colname: str = "proteins",
) -> md.MuData:
    """
    Map protein information to peptides.

    Args:
        mdata (MuData): MuData object
        modality (str): modality
        peptide_colname (str): column name for peptide information
        protein_colname (str): column name for protein information

    Returns:
        mdata (MuData): MuData object with updated protein mappings
    """

    mod_dict = get_modality_dict(mdata, level=level, modality=modality)
    peptides = [peptide for mod in mod_dict.values() for peptide in mod.var[peptide_colname]]
    proteins = [protein for mod in mod_dict.values() for protein in mod.var[protein_colname]]

    # Get protein mapping information
    peptide_map, protein_map = get_protein_mapping(peptides, proteins)

    # Store mapping information in MuData object
    mdata.uns["peptide_map"] = peptide_map
    mdata.uns["protein_map"] = protein_map

    # Make protein information mapping dict from mdata.uns['protein_info']
    protein_info = mdata.uns["protein_info"].copy()
    protein_info["concated_accession"] = protein_info["source"] + "_" + protein_info["accession"]
    protein_info = protein_info.set_index("accession")
    protein_info = protein_info[["concated_accession"]]

    protein_info_dict = protein_info.to_dict(orient="dict")["concated_accession"]

    # Remap proteins and classify peptides
    for mod_name, _ in mod_dict.items():
        mdata[mod_name].var[protein_colname] = (
            mdata[mod_name].var[peptide_colname].map(peptide_map.set_index("peptide").to_dict()["protein_group"])
        )
        mdata[mod_name].var["peptide_type"] = [
            "unique" if len(x.split(";")) == 1 else "shared" for x in mdata[mod_name].var["proteins"]
        ]
        mdata[mod_name].var = mdata[mod_name].var.rename(columns={protein_colname: "protein_group"})

        mdata[mod_name].var["repr_protein"] = (
            mdata[mod_name].var["protein_group"].apply(lambda x: select_representative(x, protein_info_dict))
        )

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
    _, initial_protein_df = _get_df(map_df)
    print("Initial proteins:", len(initial_protein_df), flush=True)

    # Find indistinguishable proteins
    map_df, indist_map = _find_indistinguisable(map_df)

    # Find subsettable proteins
    map_df, subset_map = _find_subsettable(map_df)

    # Find subsumable proteins
    map_df, subsum_map, removed_proteins = _find_subsumable(map_df)
    removed_proteins = [p for p2 in removed_proteins for p in p2.split(",")]
    initial_protein_df = initial_protein_df[~initial_protein_df["protein"].isin(removed_proteins)].reset_index(
        drop=True
    )

    # Get final output
    peptide_map, protein_map = _get_final_output(
        map_df=map_df,
        initial_protein_df=initial_protein_df,
        indist_repr_map=indist_map["repr"],
        subset_repr_map=subset_map["repr"],
        subsum_repr_map=subsum_map["repr"],
    )

    return peptide_map, protein_map


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
        protein_df (pd.DataFrame): protein information
    """
    peptide_df = _get_peptide_df(map_df=map_df)
    protein_df = _get_protein_df(map_df=map_df, peptide_df=peptide_df)

    return peptide_df, protein_df


def _get_matrix(
    map_df: pd.DataFrame,
    peptide_df: pd.DataFrame,
    protein_df: pd.DataFrame,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Calculate protein group inclusion relationship.

    Args:
        map_df (pd.DataFrame): mapping information
        peptide_df (pd.DataFrame): peptide information
        protein_df (pd.DataFrame): protein information

    Returns:
        peptide_mat (sp.csr_matrix): peptide-protein matrix
        protein_mat (np.ndarray): protein-protein matrix
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

    # Get peptide-protein matrix
    peptide_mat = sp.lil_matrix((len(protein_df), len(peptide_df)), dtype=float)
    peptide_mat[coordinates[0], coordinates[1]] = 1
    peptide_mat = peptide_mat.tocsr()

    # Get protein-protein matrix
    protein_mat = peptide_mat.dot(peptide_mat.T).toarray()

    return peptide_mat, protein_mat


def _find_indistinguisable(
    map_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Mapping]:
    """
    Get indistinguishable group information.

    Args:
        map_df (pd.DataFrame): mapping information

    Returns:
        map_df (pd.DataFrame): filtered mapping information
        indist_map (Mapping): indistinguishable mapping information
    """
    # Prepare dataframe and matrix
    peptide_df, protein_df = _get_df(map_df)
    _, protein_mat = _get_matrix(map_df, peptide_df, protein_df)

    # Get indistinguishable matrix
    inclusion_mat = protein_mat == protein_mat.max(axis=0)
    indist_mat = inclusion_mat & (inclusion_mat.T == inclusion_mat)

    # Get indistinguishable mappings
    indist_map = _get_indist_map(indist_mat, protein_df)

    # Update protein
    map_df["protein"] = map_df["protein"].map(indist_map["repr"]).fillna(map_df["protein"])
    map_df = map_df.drop_duplicates().reset_index(drop=True)
    peptide_df, protein_df = _get_df(map_df)

    removed_indist = len(indist_map["repr"]) - len(indist_map["memb"])
    print(f"- Removed indistinguishable: {removed_indist}", flush=True)

    return map_df, indist_map


def _get_indist_map(
    indist_mat: np.ndarray,
    protein_df: pd.DataFrame,
) -> Mapping:
    """
    Returns indistinguishable group information.

    Args:
        indist_mat (np.ndarray): indistinguishable matrix
        protein_df (pd.DataFrame): protein information

    Returns:
        indist_map (Mapping): indistinguishable mapping information
    """
    # Initialize mappings
    indist_repr_map = {}
    indist_memb_map = {}

    # Get Groups
    graph = _build_graph(indist_mat)
    groups = _find_groups(graph)

    # Get mappings
    for group in groups:
        memb_prot = protein_df.iloc[group]["protein"].values
        repr_prot = ",".join(memb_prot)

        for memb in memb_prot:
            indist_repr_map[memb] = repr_prot

        indist_memb_map[repr_prot] = ";".join(memb_prot)

    return Mapping(repr=indist_repr_map, memb=indist_memb_map)


def _build_graph(indist_mat: np.ndarray) -> dict[int, list[int]]:
    """
    Build graph from indistinguishable matrix.

    Args:
        indist_mat (np.ndarray): indistinguishable matrix

    Returns:
        graph (dict[int, list[int]]): graph representation
    """
    x_idx, y_idx = np.where(indist_mat)
    graph: dict[int, list[int]] = {}
    for x, y in zip(x_idx, y_idx):
        if x >= y:
            continue
        graph.setdefault(x, []).append(y)
        graph.setdefault(y, []).append(x)
    return graph


def _find_groups(graph: dict[int, list[int]]) -> list[list[int]]:
    """
    Find groups in the graph.

    Args:
        graph (dict[int, list[int]]): graph representation

    Returns:
        groups (list[list[int]]): list of groups
    """
    visited = set()
    groups = []

    for node in graph:
        if node not in visited:
            stack = deque([node])
            group = []

            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                group.append(n)
                stack.extend(graph.get(n, []))

            groups.append(sorted(group))

    return groups


def _find_subsettable(map_df: pd.DataFrame) -> Tuple[pd.DataFrame, Mapping]:
    """
    Get subset group information.

    Args:
        map_df (pd.DataFrame): mapping information

    Returns:
        map_df (pd.DataFrame): filtered mapping information
        subset_map (Mapping): subset mapping information
    """
    # Prepare dataframe and matrix
    peptide_df, protein_df = _get_df(map_df)
    _, protein_mat = _get_matrix(map_df, peptide_df, protein_df)

    # Get subset matrix
    inclusion_mat = protein_mat == protein_mat.max(axis=0)
    subset_mat = inclusion_mat & (inclusion_mat.T != inclusion_mat)

    # Get subset mappings
    subset_map = _get_subset_map(subset_mat, protein_df)

    # Update protein
    map_df["protein"] = map_df["protein"].map(subset_map["repr"]).fillna(map_df["protein"])
    map_df = map_df.drop_duplicates().reset_index(drop=True)
    peptide_df, protein_df = _get_df(map_df)

    removed_subsets = len(subset_map["repr"])
    print(f"- Removed subsettable: {removed_subsets}", flush=True)

    return map_df, subset_map


def _get_subset_map(
    subset_mat: np.ndarray,
    protein_df: pd.DataFrame,
) -> Mapping:
    """
    Returns subset group information.

    Args:
        subset_mat (np.ndarray): subset matrix
        protein_df (pd.DataFrame): protein information

    Returns:
        subset_map (Mapping): subset mapping information
    """
    # Initialize mappings
    subset_repr_map = {}
    subset_memb_map = {}

    # Build hierarchy
    hierarchy = _build_hierarchy(subset_mat)

    # Get subset members
    for r_idx, m_idx in hierarchy.items():
        repr_prot = protein_df.loc[r_idx, "protein"]
        memb_prot = protein_df.loc[m_idx, "protein"].values

        for memb in memb_prot:
            subset_repr_map[memb] = repr_prot

        subset_memb_map[repr_prot] = ";".join(memb_prot)

    return Mapping(repr=subset_repr_map, memb=subset_memb_map)


def _build_hierarchy(matrix: np.ndarray) -> dict[int, np.ndarray]:
    """
    Build hierarchy from subset matrix.

    Args:
        matrix (np.ndarray): subset matrix

    Returns:
        hierarchy (dict[int, np.ndarray]): hierarchy mapping
    """
    p_idx, c_idx = np.where(matrix)

    # Get unique parents & children
    parents = np.unique(p_idx)
    children = np.unique(c_idx)

    # Find root nodes (parents that are NOT children)
    root_mask = ~np.isin(parents, children)
    root_nodes = parents[root_mask]

    # Store children for each parent
    hierarchy = {}
    for parent in root_nodes:
        hierarchy[parent] = np.sort(c_idx[p_idx == parent])

    return hierarchy


def _find_subsumable(map_df: pd.DataFrame) -> Tuple[pd.DataFrame, Mapping]:
    """
    Get subsumable protein information.

    Args:
        map_df (pd.DataFrame): mapping information

    Returns:
        map_df (pd.DataFrame): filtered mapping information
        subsum_map (Mapping): subsum mapping information
    """
    # Prepare dataframe and matrix
    peptide_df, protein_df = _get_df(map_df)
    peptide_mat, protein_mat = _get_matrix(map_df, peptide_df, protein_df)

    # Get subsum mappings
    subsum_map, removed_proteins = _get_subsum_map(peptide_mat, protein_mat, protein_df)

    # Remove subsumable proteins
    map_df = map_df[~map_df["protein"].isin(removed_proteins)].reset_index(drop=True)

    # Update protein
    map_df["protein"] = map_df["protein"].map(subsum_map["repr"]).fillna(map_df["protein"])
    map_df = map_df.drop_duplicates().reset_index(drop=True)

    removed_subsumables = len(subsum_map["repr"]) - len(subsum_map["memb"]) + len(removed_proteins)
    print(f"- Removed subsumable: {removed_subsumables}", flush=True)

    return map_df, subsum_map, removed_proteins


def _get_subsum_map(
    peptide_mat: np.ndarray,
    protein_mat: np.ndarray,
    protein_df: pd.DataFrame,
) -> Mapping:
    """
    Returns subsumable group information.

    Args:
        peptide_mat (np.ndarray): peptide-protein matrix
        protein_df (pd.DataFrame): protein information

    Returns:
        subsum_map (Mapping): subsumable mapping information
    """
    # Initialize mappings
    subsum_repr_map = {}
    subsum_memb_map = {}
    removed_proteins = []

    # Get connections
    subsum_indices = protein_df.loc[protein_df["unique_peptides"] == 0].index
    connections = _build_connection(protein_mat, subsum_indices)

    # Get mappings
    for protein_idx in connections:
        # Make a connection dataframe
        protein_names = protein_df.loc[protein_idx, "protein"].values
        connection_mat = peptide_mat[protein_idx, :].toarray()
        connection_mat = connection_mat[:, np.sum(connection_mat, axis=0) > 0]

        # Boolean mask that are subsumable
        is_subsumable = np.array([i in subsum_indices for i in protein_idx])

        # Merge all subsumables into a single protein group
        connection_mat_subsum = np.any(connection_mat[is_subsumable, :], axis=0)
        connection_mat_unique = connection_mat[~is_subsumable, :]
        connection_mat = np.vstack([connection_mat_subsum, connection_mat_unique])

        # If there is no unique peptide, remove the protein
        if np.all(connection_mat[:, connection_mat_subsum].sum(axis=0) != 1):
            [removed_proteins.append(p) for p in protein_names[is_subsumable]]
            continue

        subsum_group = protein_names[is_subsumable]
        subsum_group_name = ",".join(subsum_group)

        for protein in subsum_group:
            subsum_repr_map[protein] = subsum_group_name

        subsum_memb_map[subsum_group_name] = ";".join(subsum_group)

    return Mapping(repr=subsum_repr_map, memb=subsum_memb_map), removed_proteins


def _build_connection(protein_mat: np.ndarray, indices: list[int]) -> list[Tuple[list[int], list[int]]]:
    """
    Build connections from peptide-protein matrix.

    Args:
        protein_mat (np.ndarray): protein-protein matrix
        indices (list[int]): list of indices

    Returns:
        connections (list[Tuple[list[int], list[int]]]): list of connections
    """
    np.fill_diagonal(protein_mat, 0)
    protein_mat = protein_mat.astype(bool)
    protein_mat_csr = sp.csr_matrix(protein_mat)
    n_components, labels = sp.csgraph.connected_components(csgraph=protein_mat_csr, directed=False, return_labels=True)
    components = [np.where(labels == i)[0].tolist() for i in range(n_components)]
    connections = [comp for comp in components if (len(comp) > 1) & (any([i in indices for i in comp]))]

    return connections


def select_canon_prot(protein_group: str, protein_info: pd.DataFrame) -> str:
    """
    DEPRECATED: Use `select_representative` instead.

    Select canonical protein from protein list based on priority.
    canonical > swissprot > trembl > contam

    Args:
        protein_group (str): protein group (uniprot entry)

    Returns:
        protein_group (str): canonical protein group
    """

    warnings.warn(
        "select_canon_prot is deprecated. Use select_representative instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return select_representative(protein_group=protein_group, protein_info=protein_info)


def select_representative(protein_group: str, protein_info: dict[str, str]) -> str:
    """
    Select canonical protein from protein list based on priority.
    canonical > swissprot > trembl > contam

    Args:
        protein_list (list[str]): list of proteins (uniprot entry)
        protein_info (pd.DataFrame): DataFrame of protein info from mdata.uns['protein_info']

    Returns:
        protein_group (str): canonical protein group
    """
    protein_list = re.split(";|,", protein_group)
    concated_protein_list: list[str] = [protein_info[k] for k in protein_list]

    swissprot_canon_ls = [prot for prot in concated_protein_list if prot.startswith("sp") and "-" not in prot]
    if swissprot_canon_ls:
        return ",".join(swissprot_canon_ls).replace("sp_", "")

    swissprot_ls = [prot for prot in concated_protein_list if prot.startswith("sp")]
    if swissprot_ls:
        return ",".join(swissprot_ls).replace("sp_", "")

    trembl_ls = [prot for prot in concated_protein_list if prot.startswith("tr")]
    if trembl_ls:
        return ",".join(trembl_ls).replace("tr_", "")

    contam_ls = [prot for prot in concated_protein_list if prot.startswith("contam")]
    if contam_ls:
        return ",".join(contam_ls)

    return ""


def _make_peptide_map(map_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create peptide mapping.

    Args:
        map_df (pd.DataFrame): mapping information

    Returns:
        peptide_map (pd.DataFrame): peptide mapping information
    """
    # Make peptide mapping
    peptide_map = (
        map_df[["peptide", "protein"]]
        .drop_duplicates()
        .groupby("peptide", as_index=False, observed=False)["protein"]
        .agg(lambda x: ";".join(x))
        .rename(columns={"protein": "protein_group"})
    )

    return peptide_map


def _make_protein_map(
    initial_protein_df: pd.DataFrame,
    subset_repr_map: dict[str, str],
    indist_repr_map: dict[str, str],
    subsum_repr_map: dict[str, str],
) -> pd.DataFrame:
    """
    Create protein mapping.

    Args:
        initial_protein_df (pd.DataFrame): initial protein information
        subset_repr_map (dict[str, str]): subset representative map
        indist_repr_map (dict[str, str]): indistinguishable representative map
        subsum_repr_map (dict[str, str]): subsumable representative map

    Returns:
        protein_map (pd.DataFrame): protein mapping information
    """
    # Map protein groups
    protein_indist = initial_protein_df["protein"].map(indist_repr_map).fillna(initial_protein_df["protein"])
    protein_subset = protein_indist.map(subset_repr_map).fillna(protein_indist)
    protein_subsum = protein_subset.map(subsum_repr_map).fillna(protein_subset)

    # Make protein mapping
    protein_map = pd.DataFrame(
        {
            "initial_protein": initial_protein_df["protein"],
            "protein_group": protein_subsum,
            "indistinguishable": initial_protein_df["protein"].isin(indist_repr_map.keys()),
            "subsetted": protein_indist.isin(subset_repr_map.keys()),
            "subsumable": protein_subset.isin(subsum_repr_map.keys()),
        }
    ).drop_duplicates()

    # Return protein_map
    return protein_map


def _get_final_output(
    map_df: pd.DataFrame,
    initial_protein_df: pd.DataFrame,
    subset_repr_map: dict[str, str],
    indist_repr_map: dict[str, str],
    subsum_repr_map: dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns final output dataframes.

    Args:
        map_df (pd.DataFrame): mapping information
        initial_protein_df (pd.DataFrame): protein information
        subset_repr_map (dict[str, str]): subset member map
        indist_repr_map (dict[str, str]): identical representative map
        subsum_repr_map (dict[str, str]): subsumable representative map

    Returns:
        peptide_map (pd.DataFrame): peptide mapping information
        protein_map (pd.DataFrame): protein mapping information
        protein_info (pd.DataFrame): protein information
    """
    # Make peptide mapping
    peptide_map = _make_peptide_map(map_df)

    # Make protein mapping
    protein_map = _make_protein_map(
        initial_protein_df=initial_protein_df,
        subset_repr_map=subset_repr_map,
        indist_repr_map=indist_repr_map,
        subsum_repr_map=subsum_repr_map,
    )

    print("- Total protein groups:", map_df["protein"].nunique(), flush=True)

    return peptide_map, protein_map
