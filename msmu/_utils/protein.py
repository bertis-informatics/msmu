import re
from collections.abc import Mapping

from .._core._provenance import log_provenance
from ._mudata import copy_mudata
from .._core._status import MuDataStatus
from .fasta import CANONICAL_CONTAMINANT_PREFIX, CANONICAL_DECOY_PREFIX
from ._anndata import _resolve_accession_column

# for type hints
import mudata as md

SWISSPROT_ENTRY_TYPE = "sp"
TREMBL_ENTRY_TYPE = "tr"
ISOFORM_SEPARATOR = "-"


@log_provenance
def select_repr_protein(mdata: md.MuData, modality: str) -> md.MuData:
    """
    Select canonical protein from protein list based on priority.
    canonical > swissprot > trembl > contam

    Parameters:
        mdata: `MuData` object with protein groups inferred
        modality: Modality name for protein data

    Returns:
        `MuData` object with representative proteins selected
    """

    mdata = copy_mudata(mdata)
    mstatus = MuDataStatus(mdata)

    if modality not in mstatus.mod_names:
        raise ValueError(f"{modality} modality not found in MuData object.")
    if "protein_info" not in mdata.uns:
        raise ValueError("protein_info not found in mdata.uns; attach a FASTA with mm.utils.attach_fasta first.")

    # An entry without a database tag counts as Swiss-Prot.
    entry_type_by_accession = mdata.uns["protein_info"]["Entry Type"].replace("", SWISSPROT_ENTRY_TYPE).to_dict()

    if modality == "protein":
        mdata.mod["protein"].var["repr_protein"] = mdata.mod["protein"].var.index.map(
            lambda x: select_representative(x, entry_type_by_accession)
        )
    else:
        accession_column = _resolve_accession_column(mdata.mod[modality].var, context=f"{modality}.var")
        mdata.mod[modality].var["repr_protein"] = (
            mdata.mod[modality].var[accession_column].apply(lambda x: select_representative(x, entry_type_by_accession))
        )

    return mdata


def select_representative(protein_group: str, entry_type_by_accession: Mapping[str, str]) -> str:
    """
    Select canonical protein from protein list based on priority.
    canonical > swissprot > trembl > contam

    A decoy member (``decoy_``-prefixed) takes the entry type of the accession it was made from.

    Args:
        protein_group: semicolon or comma-separated proteins (uniprot entries)
        entry_type_by_accession: FASTA entry type (``sp``, ``tr``) per accession

    Returns:
        the members of the highest-ranked tier, comma-separated

    Raises:
        ValueError: A member is not in the FASTA, i.e. the attached FASTA is not the search FASTA.
    """
    protein_list = re.split(";|,", protein_group)
    accession_list = [protein.removeprefix(CANONICAL_DECOY_PREFIX) for protein in protein_list]
    missing_proteins = [
        protein for protein, accession in zip(protein_list, accession_list) if accession not in entry_type_by_accession
    ]
    if missing_proteins:
        raise ValueError(
            f"Proteins {missing_proteins} are not in mdata.uns['protein_info']; "
            "attach the FASTA used for the search with mm.utils.attach_fasta."
        )

    # Contaminants rank last whatever their database tag says: with the Hao Lab convention the
    # marker sits inside the accession ("sp|Cont_P1|..."), so they would otherwise be picked up
    # by the swissprot tiers below.
    contaminant_proteins = [protein for protein in protein_list if CANONICAL_CONTAMINANT_PREFIX in protein]
    ranked_proteins = [
        (protein, entry_type_by_accession[accession])
        for protein, accession in zip(protein_list, accession_list)
        if protein not in contaminant_proteins
    ]
    tiers = (
        [
            protein
            for protein, entry_type in ranked_proteins
            if entry_type.startswith(SWISSPROT_ENTRY_TYPE) and ISOFORM_SEPARATOR not in protein
        ],
        [protein for protein, entry_type in ranked_proteins if entry_type.startswith(SWISSPROT_ENTRY_TYPE)],
        [protein for protein, entry_type in ranked_proteins if entry_type.startswith(TREMBL_ENTRY_TYPE)],
        contaminant_proteins,
    )
    for tier in tiers:
        if tier:
            return ",".join(tier)
    return ""
