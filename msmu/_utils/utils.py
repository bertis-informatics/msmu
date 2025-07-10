from typing import Iterable
import anndata as ad
import mudata as md
from collections.abc import Mapping, Sequence
import pandas as pd
import re
import functools
import datetime
import numpy as np

from .._read_write._readers import add_modality


def serialize(obj):
    if isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, Mapping):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, Sequence) and not isinstance(obj, str):
        return [serialize(v) for v in obj]
    else:
        return obj


def uns_logger(func):
    @functools.wraps(func)
    def wrapper(mdata, *args, **kwargs):
        # Run the function
        result = func(mdata, *args, **kwargs)

        if not isinstance(mdata, md.MuData) or not isinstance(result, md.MuData):
            return result

        # Filter kwargs (e.g., None values)
        filtered_kwargs = {k: str(v) for k, v in kwargs.items() if v is not None}

        # Create log entry
        log_entry = {
            "function": func.__name__,
            "timestamp": datetime.datetime.now().isoformat(),
            "args": serialize(args),
            "kwargs": serialize(filtered_kwargs),
        }

        # Initialize log structure if not present
        if "_cmd" not in result.uns_keys():
            result.uns["_cmd"] = {}

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
    psm_mdatas: Iterable[ad.AnnData] = get_modality_dict(mdata=mdata, modality="feature").values()
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


def add_quant(mdata: md.MuData, quant_data: str | pd.DataFrame, quant_tool: str) -> md.MuData:
    # mdata_quant = mdata.copy()
    if isinstance(quant_data, str):
        quant = pd.read_csv(quant_data, sep="\t")
    elif isinstance(quant_data, pd.DataFrame):
        quant = quant_data.copy()
    else:
        raise ValueError("quant_data must be file for dataframe")

    if quant_tool == "flashlfq":
        quant = quant.set_index("Sequence", drop=True)
        quant = quant.rename_axis(index=None, columns=None)
        intensity_cols = [x for x in quant.columns if x.startswith("Intensity_")]
        input_arr = quant[intensity_cols]
        input_arr.columns = [x.split("Intensity_")[1] for x in intensity_cols]
        input_arr = input_arr.replace(0, np.nan)

        obs_df = mdata.obs.copy()
        filename = [x.split(".mzML")[0] for x in obs_df["tag"]]
        rename_dict = {k: v for k, v in zip(filename, obs_df.index)}
        col_order = list(rename_dict.values())
        input_arr = input_arr.rename(columns=rename_dict)
        input_arr = input_arr[col_order]
        input_arr = input_arr.dropna(how="all")

        peptide_adata = ad.AnnData(X=input_arr.T)
        peptide_adata.uns["level"] = "peptide"

        mdata = add_modality(mdata=mdata, adata=peptide_adata, mod_name="peptide", parent_mods=["feature"])

    mdata.update_obs()

    return mdata


def rename_obs(
    mdata: md.MuData,
    map: dict[str, str] | pd.Series | pd.DataFrame,
) -> md.MuData:
    """
    Rename an observation (obs) column in the MuData object.

    Parameters
    ----------
    mdata : md.MuData
        The MuData object containing the observation to rename.
    obs_name : str
        The current name of the observation to rename.
    new_obs_name : str
        The new name for the observation.

    Returns
    -------
    md.MuData
        The modified MuData object with the renamed observation.
    """
    mdata = mdata.copy()

    if isinstance(map, pd.Series):
        map = map.to_dict()
    elif isinstance(map, pd.DataFrame):
        if len(map.columns) != 2:
            raise ValueError("DataFrame must have exactly two columns.")
        map = map.set_index(map.columns[0])[map.columns[1]].to_dict()
    elif not isinstance(map, dict):
        raise ValueError("Map must be a dictionary, pandas Series, or DataFrame.")

    if not (set(mdata.obs.index) <= set(map.keys())):
        raise ValueError("Map keys must contain the index of mdata.obs")

    mdata.obs.index = mdata.obs.index.map(map)

    for mod in mdata.mod:
        mdata[mod].obs.index = mdata[mod].obs.index.map(map)

    return mdata
