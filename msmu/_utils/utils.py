from typing import Iterable
import anndata as ad
import mudata as md
from pathlib import Path
import pandas as pd
import re
import functools
import datetime


def uns_logger(func):
    @functools.wraps(func)
    def wrapper(mdata, *args, **kwargs):
        # Run the function
        result = func(mdata, *args, **kwargs)

        # Only log if mdata and result are MuData objects
        if not isinstance(mdata, md.MuData) or not isinstance(result, md.MuData):
            return result

        # Create log entry
        log_entry = {
            "function": func.__name__,
            "timestamp": datetime.datetime.now().isoformat(),
            "args": list(args),
            "kwargs": kwargs,
        }

        # Initialize the log list if needed
        if "_cmd" not in result.uns_keys():
            result.uns["_cmd"] = {}

        # Append log
        result.uns["_cmd"][str(len(result.uns["_cmd"]))] = log_entry

        return result

    return wrapper


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


def get_label(mdata: md.MuData) -> str:
    psm_mdatas: Iterable[ad.AnnData] = get_modality_dict(mdata=mdata, level="psm").values()
    label_list: list[str] = [x.uns["label"] for x in psm_mdatas]

    if len(set(label_list)) == 1:
        label: str = label_list[0]
    else:
        raise ValueError("Multiple Label in Adatas! Please check label argument for reading search outputs!")

    return label


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
