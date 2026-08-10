import re

from .._core._status import MuDataStatus
from .fasta import CANONICAL_CONTAMINANT_PREFIX
from ..logging_utils import get_logger

# for type hints
import mudata as md


logger = get_logger(__name__)


def select_repr_protein(mdata: md.MuData, modality: str) -> md.MuData:
    """
    Select canonical protein from protein list based on priority.
    canonical > swissprot > trembl > contam

    Parameters:
        mdata: MuData object with protein groups inferred
        modality: Modality name for protein data

    Returns:
        MuData object with representative proteins selected
    """

    mdata = mdata.copy()
    mstatus = MuDataStatus(mdata)

    if modality not in mstatus.mod_names:
        logger.error(f"{modality} modality not found in MuData object.")
        raise
    elif "protein_info" not in mdata.uns:
        logger.error("protein_info not found in mdata.uns.")
        raise
    else:
        protein_info = mdata.uns["protein_info"].copy()
        protein_info.loc[protein_info["Entry Type"] == "", "Entry Type"] = "sp"
        protein_info["concated_accession"] = protein_info["Entry Type"] + "_" + protein_info.index
        protein_info = protein_info[["concated_accession"]]

        protein_info_dict = protein_info.to_dict(orient="dict")["concated_accession"]

        if modality == "protein":
            mdata.mod["protein"].var["repr_protein"] = mdata.mod["protein"].var.index.map(
                lambda x: select_representative(x, protein_info_dict)
            )
        else:
            mdata.mod[modality].var["repr_protein"] = (
                mdata.mod[modality].var["protein_group"].apply(lambda x: select_representative(x, protein_info_dict))
            )

        return mdata


def select_representative(protein_group: str, protein_info: dict[str, str]) -> str:
    """
    Select canonical protein from protein list based on priority.
    canonical > swissprot > trembl > contam

    Args:
        protein_group: semicolon or comma-separated proteins (uniprot entries)
        protein_info: mapping from accession to annotated identifier (e.g., sp_*, tr_*, *_Cont_*)

    Returns:
        canonical protein group
    """
    protein_list = re.split(";|,", protein_group)
    concated_protein_list: list[str] = [protein_info[k] for k in protein_list]

    # Contaminants rank last whatever their database tag says: with the Hao Lab convention the
    # marker sits inside the accession ("sp|Cont_P1|..."), so they would otherwise be picked up
    # by the swissprot branches below.
    contam_ls = [prot for prot in concated_protein_list if CANONICAL_CONTAMINANT_PREFIX in prot]
    ranked_protein_list = [prot for prot in concated_protein_list if prot not in contam_ls]

    swissprot_canon_ls = [prot for prot in ranked_protein_list if prot.startswith("sp") and "-" not in prot]
    if swissprot_canon_ls:
        return ",".join(swissprot_canon_ls).replace("sp_", "")

    swissprot_ls = [prot for prot in ranked_protein_list if prot.startswith("sp")]
    if swissprot_ls:
        return ",".join(swissprot_ls).replace("sp_", "")

    trembl_ls = [prot for prot in ranked_protein_list if prot.startswith("tr")]
    if trembl_ls:
        return ",".join(trembl_ls).replace("tr_", "")

    if contam_ls:
        return ",".join(_strip_entry_type_prefix(prot) for prot in contam_ls)

    return ""


def _strip_entry_type_prefix(annotated_identifier: str) -> str:
    """Drop the leading ``<entry type>_`` from an annotated identifier ("contam_sp_Cont_P1")."""
    marker_position = annotated_identifier.find(CANONICAL_CONTAMINANT_PREFIX)
    return annotated_identifier[marker_position:] if marker_position != -1 else annotated_identifier
