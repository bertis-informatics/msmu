import logging
import re

from .._read_write._mdata_status import MuDataStatus

# for type hints
import mudata as md


logger = logging.getLogger(__name__)


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
            mdata["protein"].var["repr_protein"] = mdata["protein"].var.index.map(
                lambda x: _select_representative(x, protein_info_dict)
            )
        else:
            mdata[modality].var["repr_protein"] = (
                mdata[modality].var["protein_group"].apply(lambda x: _select_representative(x, protein_info_dict))
            )

        return mdata


def _select_representative(protein_group: str, protein_info: dict[str, str]) -> str:
    """
    Select canonical protein from protein list based on priority.
    canonical > swissprot > trembl > contam

    Args:
        protein_list: list of proteins (uniprot entry)
        protein_info: DataFrame of protein info from mdata.uns['protein_info']

    Returns:
        canonical protein group
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
        return ",".join(contam_ls).replace("contam_sp_", "")

    return ""
