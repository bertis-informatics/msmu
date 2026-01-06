import logging
import re
import warnings
from collections import deque
from typing import TypedDict

import mudata as md
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .._utils import uns_logger
from .._read_write._mdata_status import MuDataStatus
from .._read_write._reader_registry import read_h5mu


logger = logging.getLogger(__name__)


class Mapping(TypedDict):
    repr: dict[str, str]
    memb: dict[str, str]


@uns_logger
def infer_protein(
    mdata: md.MuData,
    modality: str = "peptide",
    protein_colname: str = "proteins",
    peptide_colname: str = "stripped_peptide",
    propagated_from: md.MuData | str | None = None,
) -> md.MuData:
    """
    Infer protein group mappings and annotate peptides with uniqueness.

    Parameters:
        mdata: MuData object to update
        modality: modality holding peptide-level data
        protein_colname: column in var with semicolon-delimited protein accessions
        peptide_colname: column in var with stripped peptide sequences
        propagated_from: optional MuData or path to reuse existing mappings (e.g., global reference for PTM work)

    Returns:
        MuData object with updated protein mappings and peptide annotations
    """
    logger.info("Starting protein inference")

    mdata = mdata.copy()
    mstatus = MuDataStatus(mdata)

    if mstatus.peptide is None:
        return mdata

    if propagated_from is None:
        # Start from current peptide-level assignments; include decoys to keep mapping consistent
        target_peptides = mdata[modality].var[peptide_colname]
        target_proteins = mdata[modality].var[protein_colname]

        if mstatus.peptide.has_decoy:
            decoy_peptides = mdata[modality].uns["decoy"][peptide_colname]
            peptides = pd.concat([target_peptides, decoy_peptides], ignore_index=False)
            decoy_proteins = mdata[modality].uns["decoy"][protein_colname]
            proteins = pd.concat([target_proteins, decoy_proteins], ignore_index=False)
        else:
            peptides = target_peptides
            proteins = target_proteins

        # Derive peptide-to-protein group mapping from raw relationships
        peptide_map, protein_map = get_protein_mapping(peptides, proteins)

    elif isinstance(propagated_from, md.MuData):
        # Reuse mapping from an existing MuData (e.g., global reference for PTM work)
        peptide_map = propagated_from.uns["peptide_map"]
        protein_map = propagated_from.uns["protein_map"]

    elif isinstance(propagated_from, str):
        # Load external mapping from disk
        propagated_mdata = read_h5mu(propagated_from)
        peptide_map = propagated_mdata.uns["peptide_map"]
        protein_map = propagated_mdata.uns["protein_map"]

    # Store mapping information in MuData object
    mdata.uns["peptide_map"] = peptide_map
    mdata.uns["protein_map"] = protein_map

    # Remap proteins and classify peptides by uniqueness within the updated groups
    mdata[modality].var["protein_group"] = (
        mdata[modality].var[peptide_colname].map(peptide_map.set_index("peptide").to_dict()["protein_group"])
    )
    mdata[modality].var["peptide_type"] = [
        "unique" if len(x.split(";")) == 1 else "shared" for x in mdata[modality].var["protein_group"]
    ]

    if mstatus.peptide.has_decoy:
        # Apply the same mapping logic to decoys to keep QC and FDR flows aligned
        mdata[modality].uns["decoy"]["protein_group"] = (
            mdata[modality]
            .uns["decoy"][peptide_colname]
            .map(peptide_map.set_index("peptide").to_dict()["protein_group"])
        )
        mdata[modality].uns["decoy"]["peptide_type"] = [
            "unique" if len(x.split(";")) == 1 else "shared" for x in mdata[modality].uns["decoy"]["protein_group"]
        ]

    return mdata


def get_protein_mapping(
    peptides: pd.Series,
    proteins: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Infer peptide-to-protein group relationships from peptide and protein columns.

    Parameters:
        peptides: peptide identifiers aligned to proteins
        proteins: semicolon-separated accessions aligned to peptides

    Returns:
        peptide mapping information
        protein mapping information
    """
    # Initial load
    map_df = _get_map_df(peptides, proteins)
    _, initial_protein_df = _get_df(map_df)
    logger.info("Initial proteins: %d", len(initial_protein_df))

    # Find indistinguishable proteins
    map_df, indist_map = _find_indistinguishable(map_df)

    # Find subsettable proteins
    map_df, subset_map = _find_subsettable(map_df)

    # Find subsumable proteins
    map_df, subsum_map, removed_proteins = _find_subsumable(map_df)
    # Flatten merged names to drop fully subsumed proteins from the original list
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
    Build a long-form peptide-to-protein mapping table from aligned peptide and protein columns.

    Parameters:
        peptides: peptide information
        proteins: protein information

    Returns:
        mapping information
    """
    # Split proteins and explode the DataFrame
    map_df = pd.DataFrame({"protein": proteins, "peptide": peptides})
    map_df["protein"] = map_df["protein"].str.split(";")
    map_df = map_df.explode("protein").drop_duplicates().reset_index(drop=True)

    return map_df.sort_values("protein").reset_index(drop=True)


def _get_peptide_df(map_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize peptides and mark whether each peptide is unique to a single protein group.

    Parameters:
        map_df: mapping information

    Returns:
        peptide information
    """
    # Group by peptide and count the number of proteins
    peptide_df = map_df[["protein", "peptide"]]
    peptide_df = peptide_df.groupby("peptide", as_index=False, observed=False).count()
    peptide_df["is_unique"] = peptide_df["protein"] == 1

    return peptide_df


def _get_protein_df(map_df: pd.DataFrame, peptide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize proteins with counts of shared and unique peptides.

    Parameters:
        map_df: mapping information
        peptide_df: peptide information

    Returns:
        protein information
    """
    PEP_COLS = ["peptide", "is_unique"]
    GROUP_COLS = ["protein"]

    data = map_df.merge(peptide_df[PEP_COLS], on="peptide", how="left")

    # Count shared & unique peptides for each protein
    protein_df = (
        data.groupby(GROUP_COLS, observed=False)
        .agg(
            shared_peptides=("is_unique", lambda x: (~x).sum()),
            unique_peptides=("is_unique", "sum"),
        )
        .reset_index()
    )

    return protein_df


def _get_df(map_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper to produce both peptide and protein summaries.

    Parameters:
        map_df: mapping information

    Returns:
        peptide information
        protein information
    """
    peptide_df = _get_peptide_df(map_df=map_df)
    protein_df = _get_protein_df(map_df=map_df, peptide_df=peptide_df)

    return peptide_df, protein_df


def _get_matrix(
    map_df: pd.DataFrame,
    peptide_df: pd.DataFrame,
    protein_df: pd.DataFrame,
) -> tuple[sp.csr_array, np.ndarray]:
    """
    Calculate protein group inclusion relationships.

    Parameters:
        map_df: peptide-to-protein map
        peptide_df: peptide summary
        protein_df: protein summary

    Returns:
        peptide-protein matrix
        protein-protein matrix
    """
    # Build dense indexers so we can place entries in sparse matrices efficiently
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

    # Get protein-protein matrix (pairwise shared peptide counts)
    protein_mat = peptide_mat.dot(peptide_mat.T).toarray()

    return peptide_mat, protein_mat


def _find_indistinguishable(
    map_df: pd.DataFrame,
) -> tuple[pd.DataFrame, Mapping]:
    """
    Identify protein groups that share identical peptide sets and merge them under representative IDs.

    Parameters:
        map_df: mapping information

    Returns:
        filtered mapping information
        indistinguishable mapping information
    """
    # Prepare dataframe and matrix
    peptide_df, protein_df = _get_df(map_df)
    _, protein_mat = _get_matrix(map_df, peptide_df, protein_df)

    # Get indistinguishable matrix
    inclusion_mat = protein_mat == protein_mat.max(axis=0)
    indist_mat = inclusion_mat & (inclusion_mat.T == inclusion_mat)

    # Get indistinguishable mappings
    indist_map = _get_indist_map(indist_mat, protein_df)

    # Update protein IDs to their indistinguishable representative names
    map_df["protein"] = map_df["protein"].map(indist_map["repr"]).fillna(map_df["protein"])
    map_df = map_df.drop_duplicates().reset_index(drop=True)
    peptide_df, protein_df = _get_df(map_df)

    removed_indist = len(indist_map["repr"]) - len(indist_map["memb"])
    logger.info("Removed indistinguishable: %d", removed_indist)

    return map_df, indist_map


def _get_indist_map(
    indist_mat: np.ndarray,
    protein_df: pd.DataFrame,
) -> Mapping:
    """
    Build mappings that collapse indistinguishable proteins into comma-joined representatives and track members.

    Parameters:
        indist_mat: indistinguishable matrix
        protein_df: protein information

    Returns:
        indistinguishable mapping information
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
    Build an undirected graph of proteins that are indistinguishable from one another.

    Parameters:
        indist_mat: indistinguishable matrix

    Returns:
        graph representation
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
    Find connected components in the indistinguishability graph.

    Parameters:
        graph: graph representation

    Returns:
        list of groups
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


def _find_subsettable(map_df: pd.DataFrame) -> tuple[pd.DataFrame, Mapping]:
    """
    Collapse proteins whose peptide sets are strict subsets of others.

    Parameters:
        map_df: mapping information

    Returns:
        filtered mapping information
        subset mapping information
    """
    # Prepare dataframe and matrix
    peptide_df, protein_df = _get_df(map_df)
    _, protein_mat = _get_matrix(map_df, peptide_df, protein_df)

    # Get subset matrix
    inclusion_mat = protein_mat == protein_mat.max(axis=0)
    subset_mat = inclusion_mat & (inclusion_mat.T != inclusion_mat)

    # Get subset mappings
    subset_map = _get_subset_map(subset_mat, protein_df)

    # Redirect subsetted proteins to their parent representatives
    map_df["protein"] = map_df["protein"].map(subset_map["repr"]).fillna(map_df["protein"])
    map_df = map_df.drop_duplicates().reset_index(drop=True)
    peptide_df, protein_df = _get_df(map_df)

    removed_subsets = len(subset_map["repr"])
    logger.info("Removed subsettable: %d", removed_subsets)

    return map_df, subset_map


def _get_subset_map(
    subset_mat: np.ndarray,
    protein_df: pd.DataFrame,
) -> Mapping:
    """
    Build mappings that redirect subset proteins to their representative parents and record membership.

    Parameters:
        subset_mat: subset matrix
        protein_df: protein information

    Returns:
        subset mapping information
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
    Build a parent-to-children hierarchy from subset relationships.

    Parameters:
        matrix: subset matrix

    Returns:
        hierarchy mapping
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


def _find_subsumable(map_df: pd.DataFrame) -> tuple[pd.DataFrame, Mapping, list[str]]:
    """
    Merge proteins that only share peptides within a connected component and lack unique evidence.

    Parameters:
        map_df: mapping information

    Returns:
        filtered mapping information
        subsum mapping information
        proteins dropped because they lack unique evidence
    """
    # Prepare dataframe and matrix
    peptide_df, protein_df = _get_df(map_df)
    peptide_mat, protein_mat = _get_matrix(map_df, peptide_df, protein_df)

    # Get subsum mappings
    subsum_map, removed_proteins = _get_subsum_map(peptide_mat, protein_mat, protein_df)

    # Remove subsumable proteins
    map_df = map_df[~map_df["protein"].isin(removed_proteins)].reset_index(drop=True)

    # Redirect subsumable proteins to their merged representatives
    map_df["protein"] = map_df["protein"].map(subsum_map["repr"]).fillna(map_df["protein"])
    map_df = map_df.drop_duplicates().reset_index(drop=True)

    removed_subsumables = len(subsum_map["repr"]) - len(subsum_map["memb"]) + len(removed_proteins)
    logger.info("Removed subsumable: %d", removed_subsumables)

    return map_df, subsum_map, removed_proteins


def _get_subsum_map(
    peptide_mat: sp.csr_array,
    protein_mat: np.ndarray,
    protein_df: pd.DataFrame,
) -> tuple[Mapping, list[str]]:
    """
    Build mappings that merge subsumable proteins and list any removed due to absent unique peptides.

    Parameters:
        peptide_mat: peptide-protein matrix
        protein_mat: protein-protein matrix
        protein_df: protein information

    Returns:
        subsumable mapping information
        proteins removed for lacking unique support
    """
    # Initialize mappings
    subsum_repr_map = {}
    subsum_memb_map = {}
    removed_proteins: list[str] = []

    # Get connections
    subsum_indices = protein_df.loc[protein_df["unique_peptides"] == 0].index.tolist()
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
            # Entire component lacks discriminating peptides; drop subsumable members
            [removed_proteins.append(p) for p in protein_names[is_subsumable]]
            continue

        subsum_group = protein_names[is_subsumable]
        subsum_group_name = ",".join(subsum_group)

        for protein in subsum_group:
            subsum_repr_map[protein] = subsum_group_name

        subsum_memb_map[subsum_group_name] = ";".join(subsum_group)

    return Mapping(repr=subsum_repr_map, memb=subsum_memb_map), removed_proteins


def _build_connection(protein_mat: np.ndarray, indices: list[int]) -> list[list[int]]:
    """
    Build connected components from the protein–protein overlap graph and keep only those containing subsumable candidates.

    Parameters:
        protein_mat: protein-protein matrix
        indices: indices to consider for connectivity

    Returns:
        list of connected component indices
    """
    np.fill_diagonal(protein_mat, 0)
    protein_mat = protein_mat.astype(bool)
    protein_mat_csr = sp.csr_array(protein_mat)
    n_components, labels = sp.csgraph.connected_components(csgraph=protein_mat_csr, directed=False, return_labels=True)
    components = [np.where(labels == i)[0].tolist() for i in range(n_components)]
    # Keep only components that have at least one candidate subsumable protein
    connections = [comp for comp in components if (len(comp) > 1) & (any([i in indices for i in comp]))]

    return connections


def select_canon_prot(protein_group: str, protein_info: dict[str, str]) -> str:
    """
    > DEPRECATED: Use `select_representative` instead.

    Choose a representative protein accession from a group using priority:

    `canonical > swissprot > trembl > contam`.

    Parameters:
        protein_group: semicolon or comma-separated proteins (uniprot entries)
        protein_info: mapping from accession to annotated identifier (e.g., sp_*, tr_*, contam_*)

    Returns:
        canonical protein group
    """
    warnings.warn(
        "select_canon_prot is deprecated. Use select_representative instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return select_representative(protein_group=protein_group, protein_info=protein_info)


def select_representative(protein_group: str, protein_info: dict[str, str]) -> str:
    """
    Choose a representative protein accession from a group using priority:

    `canonical > swissprot > trembl > contam`.

    Parameters:
        protein_group: semicolon or comma-separated proteins (uniprot entries)
        protein_info: mapping from accession to annotated identifier (e.g., sp_*, tr_*, contam_*)

    Returns:
        canonical protein group
    """
    protein_list = re.split(";|,", protein_group)
    annotated_protein_list: list[str] = [protein_info[k] for k in protein_list]

    swissprot_canon_ls = [prot for prot in annotated_protein_list if prot.startswith("sp") and "-" not in prot]
    if swissprot_canon_ls:
        return ",".join(swissprot_canon_ls).replace("sp_", "")

    swissprot_ls = [prot for prot in annotated_protein_list if prot.startswith("sp")]
    if swissprot_ls:
        return ",".join(swissprot_ls).replace("sp_", "")

    trembl_ls = [prot for prot in annotated_protein_list if prot.startswith("tr")]
    if trembl_ls:
        return ",".join(trembl_ls).replace("tr_", "")

    contam_ls = [prot for prot in annotated_protein_list if prot.startswith("contam")]
    if contam_ls:
        return ",".join(contam_ls)

    return ""


def _make_peptide_map(map_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the final peptide map, concatenating all protein IDs per peptide into a group string.

    Parameters:
        map_df: mapping information

    Returns:
        peptide mapping information
    """
    # Make peptide mapping
    peptide_map = (
        map_df[["peptide", "protein"]]
        .drop_duplicates()
        .groupby("peptide", as_index=False, observed=False)
        .agg({"protein": lambda x: ";".join(x)})
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
    Create the protein mapping table with representative groups and flags indicating how each protein was handled.

    Parameters:
        initial_protein_df: initial protein information
        subset_repr_map: subset representative map
        indist_repr_map: indistinguishable representative map
        subsum_repr_map: subsumable representative map

    Returns:
        protein mapping information
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce the final peptide and protein mapping tables after all grouping steps.

    Parameters:
        map_df: mapping information
        initial_protein_df: protein information
        subset_repr_map: subset member map
        indist_repr_map: identical representative map
        subsum_repr_map: subsumable representative map

    Returns:
        peptide mapping information
        protein mapping information
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

    logger.info("Total protein groups: %d", map_df["protein"].nunique())

    return peptide_map, protein_map
