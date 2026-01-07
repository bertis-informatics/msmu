import re
import pandas as pd
from Bio import SeqIO
import logging

import mudata as md


logger = logging.getLogger(__name__)


def attach_fasta(mdata: md.MuData, fasta_file: str | None) -> md.MuData:
    """
    Attach FASTA metadata to the MuData object.

    Parameters:
        mdata: MuData object to attach FASTA metadata to.
        fasta_file: Path to the FASTA file. If None, fetch from UniProt (not implemented).

    Returns:
        MuData object with attached FASTA metadata.
    """
    mdata = mdata.copy()
    if fasta_file is not None:
        fasta_meta = _get_protein_info_from_fasta(fasta_file)
        mdata.uns["protein_info"] = fasta_meta
    else:
        fasta_meta = _get_protein_info_from_uniprot()
        mdata.uns["protein_info"] = fasta_meta

    return mdata


def _get_protein_info_from_fasta(fasta_file: str) -> pd.DataFrame:
    fasta_dict: dict = dict()
    for record in SeqIO.parse(fasta_file, "fasta"):
        desc: str = record.description
        ref_uniprot: str = record.id.split("|")[1]
        seq_: str = str(record.seq)

        entry_type, accession, protein_id = _split_uniprot_fasta_entry(record.id)
        if entry_type.startswith("contam_"):
            ref_uniprot = "contam_" + ref_uniprot

        gene_name_search = re.search(r"GN=([^\s]+)", desc)
        gene_name: str = gene_name_search.group(1) if gene_name_search else "Unknown"

        organism_search = re.search(r"OS=([^\s].+?) OX=", desc)
        organism: str = organism_search.group(1) if organism_search else "Unknown"

        description = desc.split(" OS=")[0].split(" ", 1)[1] if " OS=" in desc else desc

        fasta_dict[ref_uniprot] = {
            "Entry Type": entry_type,
            "Accession": accession,
            "Protein ID": protein_id,
            "Gene": gene_name,
            "Description": description,
            "Organism": organism,
            "Sequence": seq_,
        }
    fasta_meta = pd.DataFrame.from_dict(fasta_dict, orient="index")

    return fasta_meta


def _get_protein_info_from_uniprot() -> pd.DataFrame:
    logger.error("Fetching protein info from UniProt is not implemented.")
    raise


def parse_uniprot_accession(proteins: pd.Series) -> pd.DataFrame:
    protein_df: pd.DataFrame = pd.DataFrame(proteins)
    col_ = protein_df.columns[0]
    protein_df["index"] = range(len(protein_df))
    protein_df["protein"] = protein_df[col_].apply(lambda x: x.split(";"))
    protein_df = protein_df.explode("protein")

    uniprot_id_category: list = ["source", "accession", "protein_name"]
    for idx, cat_ in enumerate(uniprot_id_category):
        protein_df[cat_] = protein_df["protein"].apply(lambda x: _split_uniprot_fasta_entry(x)[idx])

    protein_df["accession"] = protein_df.apply(
        lambda x: (f"rev_{x['accession']}" if x["protein"].startswith("rev_") else x["accession"]),
        axis=1,
    )
    protein_df["accession"] = protein_df.apply(
        lambda x: (f"contam_{x['accession']}" if x["protein"].startswith("contam_") else x["accession"]),
        axis=1,
    )

    protein_df = protein_df.groupby(["index", col_], as_index=False).agg(";".join)
    protein_df = protein_df.sort_values("index")

    parsed_accessions: list[str] = protein_df["accession"].tolist()

    return parsed_accessions


def _split_uniprot_fasta_entry(entry: str) -> tuple[str, str, str]:
    """
    Splits a Uniprot FASTA entry into its accession and protein name.

    Parameters:
        entry: The Uniprot FASTA entry.

    Returns:
        protein entry type
        protein accession
        protein name
    """
    parts = entry.split("|")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    else:
        return "", parts[0], ""  # Handle cases where the format is different


def _map_fasta(protein_group: str, fasta_meta: pd.DataFrame, category: str) -> str:
    """
    Map protein groups to gene names using a FASTA metadata DataFrame.

    Parameters:
        protein_group: Protein group.
        fasta_meta: DataFrame containing fasta metadata.
        category: Category to map from fasta metadata.
    Returns:
        str containing gene names.
    """
    groups = protein_group.split(";")
    transformed_groups = []

    for group in groups:
        members = group.split(",")
        transformed_members = [fasta_meta[category].get(member, None) for member in members]
        transformed_groups.append(",".join(set(filter(None, transformed_members))))

    return ";".join(transformed_groups)


def map_fasta(
    mdata: md.MuData,
    modality: str,
    categories: list[str] = ["Protein ID", "Gene", "Description", "Organism"],
) -> md.MuData:
    """
    Map protein groups to gene names using a FASTA metadata DataFrame.

    Parameters:
        mdata: MuData object containing the modality to map.
        modality: The modality in the MuData object to map.
        categories: List of categories to map from fasta metadata.

    Returns:
        MuData object with updated modality var.
    """
    mdata = mdata.copy()
    fasta_meta = mdata.uns["protein_info"]

    for category in categories:
        if category not in fasta_meta.columns:
            logger.info(f"Category {category} not found in fasta metadata. Skipping mapping for this category.")
            continue

        if modality == "protein":
            mdata[modality].var[category] = mdata[modality].var.index.map(lambda x: _map_fasta(x, fasta_meta, category))
        else:
            mdata[modality].var[category] = (
                mdata[modality].var["protein_group"].map(lambda x: _map_fasta(x, fasta_meta, category))
            )

    return mdata
