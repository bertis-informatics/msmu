import anndata as ad
import mudata as md
from pathlib import Path
import pandas as pd
import re


def get_modality_dict(
    mdata: md.MuData,
    level: str | None = None,
    modality: str | None = None,
) -> dict[str, ad.AnnData]:
    """Get modality data from MuData object"""

    if (level == None) & (modality == None):
        raise ValueError("Either level or modality must be provided")

    if (level != None) & (modality != None):
        print("Both level and modality are provided. Using level prior to modality.")

    mod_dict: dict = dict()
    if level != None:
        for mod_name in mdata.mod_names:
            if mdata[mod_name].uns["level"] == level:
                mod_dict[mod_name] = mdata[mod_name].copy()

    elif modality != None:
        mod_dict[modality] = mdata[modality].copy()

    return mod_dict


def get_fasta_meta(
    fasta: str | None = None,
) -> list[str]:
    """
    Parse a FASTA file to extract protein symbols, descriptions, and gene names.

    Args:
        fasta (str | None): Path to the FASTA file.

    Returns:
        pd.DataFrame: DataFrame containing protein symbols, descriptions, and gene names.
    """

    with open(fasta, "r") as f:
        headers = [line.strip() for line in f if line.startswith(">")]

        pattern = re.compile(r">(\w+)\|(\S+)\|(\S+) (.+?) OS=(.+?) OX=(\d+) GN=(\S+)")
        pattern_sub = re.compile(r">(\w+)\|(\S+)\|(\S+) (.+?) OS=(.+?) GN=(\S+)")

    parsed_data = []
    for header in headers:
        match = pattern.match(header)
        if match:
            entry_type, accession, protein_id, protein_name, organism, taxonomy_id, gene = match.groups()
            parsed_data.append(
                {
                    "Entry Type": entry_type,  # sp or tr
                    "Accession": accession,
                    "Protein ID": protein_id,
                    "Protein Name": protein_name,
                    "Organism": organism if organism else "Unknown",
                    "Taxonomy ID": taxonomy_id if taxonomy_id else "Unknown",
                    "Gene": gene if gene else "Unknown",
                }
            )
        else:
            match = pattern_sub.match(header)
            if match:
                entry_type, accession, protein_id, protein_name, organism, gene = match.groups()
                parsed_data.append(
                    {
                        "Entry Type": entry_type,
                        "Accession": accession,
                        "Protein ID": protein_id,
                        "Protein Name": protein_name,
                        "Organism": organism if organism else "Unknown",
                        "Taxonomy ID": "Unknown",
                        "Gene": gene if gene else "Unknown",
                    }
                )

    fasta_meta = pd.DataFrame(parsed_data)
    fasta_meta.index = fasta_meta["Entry Type"] + "|" + fasta_meta["Accession"] + "|" + fasta_meta["Protein ID"]

    return fasta_meta


def _map_fasta(protein_group: str, fasta_meta: pd.DataFrame) -> pd.Series:
    """
    Map protein groups to gene names using a FASTA metadata DataFrame.

    Args:
        protein_group (str): Protein group.
        fasta_meta (pd.DataFrame): DataFrame containing fasta metadata.

    Returns:
        pd.Series: Series containing gene names.
    """
    groups = protein_group.split(";")
    transformed_groups = []

    for group in groups:
        members = group.split(",")
        transformed_members = [fasta_meta["Gene"].get(member, None) for member in members]
        transformed_groups.append(",".join(set(filter(None, transformed_members))))

    return ";".join(transformed_groups)


def map_fasta(protein_groups: pd.Series, fasta_meta: pd.DataFrame) -> pd.Series:
    """
    Map protein groups to gene names using a FASTA metadata DataFrame.

    Args:
        protein_groups (pd.Series): Series containing protein groups.
        fasta_meta (pd.DataFrame): DataFrame containing fasta metadata.

    Returns:
        pd.Series: Series containing gene names.
    """
    return protein_groups.map(lambda x: _map_fasta(x, fasta_meta))
