import re
import pandas as pd
from Bio import SeqIO

import mudata as md

from ..logging_utils import get_logger

logger = get_logger(__name__)

# Database tags of a UniProt-style entry ("sp|P02769|ALBU_BOVIN"). Anything sitting in front of
# one of these is a decoy tag: those differ per search engine (Sage "rev_", MaxQuant "REV__",
# Percolator "DECOY_") and Sage's is a config value, so they are never enumerated here.
UNIPROT_DATABASE_TAGS = ("sp", "tr")

# Contaminant markers seen in practice. Our search pipeline emits "contam_sp|P02769|ALBU_BOVIN"
# (marker in front), the Hao Lab universal contaminant FASTA emits "sp|Cont_P00722|BGAL_ECOLI"
# (marker inside the accession), MaxQuant emits "CON__".
CONTAMINANT_MARKERS = ("contam_", "Cont_", "CON__")

CANONICAL_CONTAMINANT_PREFIX = "Cont_"
CANONICAL_DECOY_PREFIX = "rev_"


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
        seq_: str = str(record.seq)

        entry_type, _raw_accession, protein_id = _split_uniprot_fasta_entry(record.id)
        # Index on the canonical accession so the mapping keys match what the readers parse out
        # of the search output, whichever contaminant FASTA convention the search used.
        ref_uniprot, _is_contaminant = _parse_protein_entry(record.id)
        accession, _ = _strip_contaminant_marker(_raw_accession, search_anywhere=False)

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


def _strip_contaminant_marker(text: str, *, search_anywhere: bool) -> tuple[str, bool]:
    """Remove the first contaminant marker from ``text`` and report whether one was found.

    ``search_anywhere`` is for the database field, where a decoy tag of unknown shape may sit in
    front of the marker ("rev_contam_sp"); the accession field is matched from the start only,
    since the Hao Lab marker is always leading there.
    """
    for marker in CONTAMINANT_MARKERS:
        if search_anywhere and marker in text:
            return text.replace(marker, "", 1), True
        if not search_anywhere and text.startswith(marker):
            return text[len(marker) :], True
    return text, False


def _parse_protein_entry(protein_entry: str) -> tuple[str, bool]:
    """Parse one protein entry into its canonical accession and its contaminant status.

    The canonical form is ``[rev_][Cont_]<accession>``, so entries reaching msmu through
    different search engines and contaminant FASTA conventions collapse onto one spelling.

    The protein name field is never inspected: real names such as ``CONA_CANLI`` would otherwise
    match a contaminant marker.
    """
    fields = protein_entry.split("|")
    if len(fields) != 3:
        # Not a UniProt-style entry (e.g. a bare accession); only the marker can be recovered.
        accession, is_contaminant = _strip_contaminant_marker(protein_entry, search_anywhere=True)
        prefix = CANONICAL_CONTAMINANT_PREFIX if is_contaminant else ""
        return prefix + accession, is_contaminant

    database_field, accession, _protein_name = fields
    database_field, marker_in_database_field = _strip_contaminant_marker(database_field, search_anywhere=True)
    accession, marker_in_accession = _strip_contaminant_marker(accession, search_anywhere=False)

    # A recognised database tag carrying something in front of it is a decoy entry. Decoy status
    # itself comes from the reader's flag column; the prefix is kept only so that decoy accessions
    # stay distinct from their target counterparts.
    is_decoy = any(database_field.endswith(tag) and database_field != tag for tag in UNIPROT_DATABASE_TAGS)
    is_contaminant = marker_in_database_field or marker_in_accession

    prefix = (CANONICAL_DECOY_PREFIX if is_decoy else "") + (CANONICAL_CONTAMINANT_PREFIX if is_contaminant else "")
    return prefix + accession, is_contaminant


def parse_uniprot_accession_group(protein_group: str) -> tuple[str, bool]:
    """Parse one semicolon-delimited protein group into its accession string.

    Returns the accession string and whether any member is a contaminant, so callers take the
    contaminant flag from the parser instead of re-matching the marker on the parsed string.

    Kept as a per-value function so callers can deduplicate (map over distinct protein groups)
    instead of parsing every PSM row -- protein groups repeat heavily across PSMs.
    """
    group_accessions: list[str] = []
    group_has_contaminant = False
    for protein in protein_group.split(";"):
        accession, is_contaminant = _parse_protein_entry(protein)
        group_accessions.append(accession)
        group_has_contaminant = group_has_contaminant or is_contaminant

    return ";".join(group_accessions), group_has_contaminant


def parse_uniprot_accession(proteins: pd.Series) -> list[str]:
    # Keep parsing in a tight Python loop; this avoids expensive explode + row-wise apply.
    return [parse_uniprot_accession_group(protein_group)[0] for protein_group in proteins]


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
            mdata.mod[modality].var[category] = mdata.mod[modality].var.index.map(
                lambda x: _map_fasta(x, fasta_meta, category)
            )
        else:
            mdata.mod[modality].var[category] = (
                mdata.mod[modality].var["protein_group"].map(lambda x: _map_fasta(x, fasta_meta, category))
            )

    return mdata
