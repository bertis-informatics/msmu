"""Pandas-path reader transforms -- the delete-at-cutover unit (BID-79).

Every function here is the pandas implementation of a reader transform whose polars twin lives on
the reader class. They are grouped in one module so that, once the polars path is the only path,
the whole file is deleted and each dispatcher collapses to its polars call -- nothing here survives
the cutover. Functions take the reader instance (``reader``) so they can read its settings and set
the same side effects (``_mbr``, ``search_settings.has_decoy``, ``_cols_to_stringify``) the inline
branches did; the reader-side helpers they call (``_map_unique``, ``_extract_scan_number``,
``_make_unique_index``, ``_set_mbr``/``_set_decoy``, ``_label_decoy``) are likewise pandas-only and
go away with this module.

Nothing in this file should be imported by the polars path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._utils.fasta import parse_uniprot_accession, parse_uniprot_accession_group


# --------------------------------------------------------------------------------------------------
# DIA-NN
# --------------------------------------------------------------------------------------------------
def diann_identification(reader, identification_df: pd.DataFrame) -> pd.DataFrame:
    """Pandas twin of ``DiannReader._identification_columns_polars`` (build the feature frame)."""
    reader._set_mbr(identification_df)  # sets reader._mbr (selects the q-value column to rename)
    reader._set_decoy(identification_df)

    # object columns of the raw frame -> stringified into varm when kept
    for col in identification_df.columns:
        if identification_df[col].dtype == "object":
            reader._cols_to_stringify.append(col)

    feature_df = pd.DataFrame(index=identification_df.index)
    feature_df["proteins"] = parse_uniprot_accession(identification_df["Protein.Ids"])
    feature_df["missed_cleavages"] = identification_df["Stripped.Sequence"].str.count(r"(?<=[KR])(?!P)")
    feature_df["peptide_length"] = identification_df["Stripped.Sequence"].str.len()
    feature_df["PEP"] = identification_df["PEP"]
    feature_df["Modified.Sequence"] = identification_df["Modified.Sequence"]  # -> peptide
    feature_df["Stripped.Sequence"] = identification_df["Stripped.Sequence"]  # -> stripped_peptide
    feature_df["Run"] = identification_df["Run"]  # -> filename
    feature_df["Precursor.Charge"] = identification_df["Precursor.Charge"]  # -> charge
    feature_df["RT"] = identification_df["RT"]  # -> rt
    feature_df["Precursor.Id"] = identification_df["Precursor.Id"]  # for _make_unique_index (dropped at subset)
    q_value_source = "Lib.Q.Value" if reader._mbr else "Global.Q.Value"
    feature_df[q_value_source] = identification_df[q_value_source]  # -> q_value
    if reader.search_settings.has_decoy:
        feature_df["Decoy"] = identification_df["Decoy"]  # -> decoy
    else:
        feature_df["decoy"] = 0

    return feature_df


# --------------------------------------------------------------------------------------------------
# MaxQuant
# --------------------------------------------------------------------------------------------------
def maxquant_identification(reader, identification_df: pd.DataFrame) -> pd.DataFrame:
    """Pandas twin of ``MaxQuantReader._identification_columns_polars``."""
    feature_df = pd.DataFrame(index=identification_df.index)

    decoy = identification_df["Reverse"].apply(lambda x: 1 if x == "+" else 0)
    feature_df["decoy"] = decoy
    feature_df["contaminant"] = identification_df["Potential contaminant"].apply(lambda x: 1 if x == "+" else 0)

    proteins = identification_df["Proteins"].copy()
    proteins.loc[decoy == 1] = identification_df.loc[decoy == 1, "Leading proteins"]
    feature_df["proteins"] = proteins

    feature_df["Sequence"] = identification_df["Sequence"]  # -> stripped_peptide
    feature_df["Modified sequence"] = identification_df["Modified sequence"]  # -> peptide
    feature_df["Length"] = identification_df["Length"]  # -> peptide_length
    feature_df["Missed cleavages"] = identification_df["Missed cleavages"]  # -> missed_cleavages
    feature_df["Charge"] = identification_df["Charge"]  # -> charge
    feature_df["Raw file"] = identification_df["Raw file"]  # -> filename
    feature_df["MS/MS Scan Number"] = identification_df["MS/MS Scan Number"]  # -> scan_num
    feature_df["Retention time"] = identification_df["Retention time"]  # -> rt
    feature_df["PEP"] = identification_df["PEP"]

    return feature_df


def maxtmt_quant(reader, raw_identification_df: pd.DataFrame) -> pd.DataFrame:
    """Pandas twin of the polars branch of ``MaxTmtReader._extract_quant_from_raw``."""
    reporter_cols = [c for c in raw_identification_df.columns if c.startswith("Reporter intensity corrected")]
    quant = pd.DataFrame(index=raw_identification_df.index)
    quant["filename"] = raw_identification_df["Raw file"]
    quant["scan_num"] = raw_identification_df["MS/MS Scan Number"]
    for col in reporter_cols:
        quant[col] = raw_identification_df[col]
    quant = reader._make_unique_index(quant)
    quant = quant.drop(columns=["filename", "scan_num"])

    return quant


def maxlfq_quant(reader, raw_identification_df: pd.DataFrame) -> pd.DataFrame:
    """Pandas twin of the polars branch of ``MaxLfqReader._extract_quant_from_raw``."""
    pep_quant_df = pd.DataFrame(
        {
            "filename": raw_identification_df["Raw file"].to_numpy(),
            "peptide": raw_identification_df["Modified sequence"].to_numpy(),
            "Intensity": raw_identification_df["Intensity"].to_numpy(),
        }
    )
    pep_quant_df = pep_quant_df.pivot_table(index="peptide", columns="filename", values="Intensity", aggfunc="sum")
    pep_quant_df = pep_quant_df.rename_axis(index=None, columns=None)

    return pep_quant_df


# --------------------------------------------------------------------------------------------------
# Sage
# --------------------------------------------------------------------------------------------------
def sage_identification(reader, identification_df: pd.DataFrame) -> pd.DataFrame:
    """Pandas twin of ``SageReader._identification_columns_polars``."""
    feature_df = pd.DataFrame(index=identification_df.index)
    feature_df["proteins"] = reader._map_unique(identification_df["proteins"], parse_uniprot_accession_group)
    feature_df["peptide"] = identification_df["peptide"]
    feature_df["filename"] = reader._map_unique(identification_df["filename"], reader._strip_filename)
    feature_df["scan_num"] = reader._map_unique(identification_df["scannr"], reader._extract_scan_number)
    feature_df["stripped_peptide"] = reader._map_unique(identification_df["peptide"], reader._make_stripped_peptide)
    feature_df["charge"] = identification_df["charge"]
    feature_df["peptide_len"] = identification_df["peptide_len"]
    feature_df["expmass"] = identification_df["expmass"]
    feature_df["calcmass"] = identification_df["calcmass"]
    feature_df["rt"] = identification_df["rt"]
    feature_df["missed_cleavages"] = identification_df["missed_cleavages"]
    feature_df["semi_enzymatic"] = identification_df["semi_enzymatic"]
    feature_df["decoy"] = (identification_df["label"] == -1).astype(int)
    feature_df["contaminant"] = feature_df["proteins"].str.contains("contam_", regex=False).astype(int)
    feature_df["PEP"] = np.power(10, identification_df["posterior_error"])  # convert log10 PEP to PEP
    feature_df["hyperscore"] = identification_df["hyperscore"]
    feature_df["spectrum_q"] = identification_df["spectrum_q"]

    return feature_df


def tmt_sage_quant(reader, quantification_df: pd.DataFrame) -> pd.DataFrame:
    """Pandas twin of ``TmtSageReader._quantification_columns_polars``."""
    quantification_df["filename"] = reader._map_unique(quantification_df["filename"], reader._strip_filename)
    quantification_df["scan_num"] = quantification_df["scannr"].apply(reader._extract_scan_number)
    quantification_df = reader._make_unique_index(quantification_df)
    quantification_df = quantification_df.drop(["filename", "scannr", "scan_num", "ion_injection_time"], axis=1)

    return quantification_df


# --------------------------------------------------------------------------------------------------
# FragPipe
# --------------------------------------------------------------------------------------------------
def fragpipe_identification(reader, identification_df: pd.DataFrame) -> pd.DataFrame:
    """Pandas twin of ``FragPipeReader._identification_columns_polars``."""
    feature_df = pd.DataFrame(index=identification_df.index)

    feature_df["filename"] = identification_df["Spectrum"].apply(lambda x: x.split(".")[0])
    feature_df["scan_num"] = identification_df["Spectrum"].apply(lambda x: int(x.split(".")[1]))

    proteins = identification_df["Protein"].astype(str) + "," + identification_df["Mapped Proteins"].astype(str)
    proteins = proteins.apply(lambda x: [y.strip() for y in x.split(",") if y != "nan"])
    proteins = proteins.apply(lambda x: ",".join(x))
    proteins = proteins.apply(lambda x: x.replace(",", ";"))
    feature_df["proteins"] = proteins

    peptide = identification_df["Modified Peptide"].copy()
    peptide.loc[peptide.isna()] = identification_df.loc[peptide.isna(), "Peptide"]
    feature_df["peptide"] = peptide

    feature_df["decoy"] = feature_df["proteins"].apply(reader._label_decoy)
    if feature_df["decoy"].unique().tolist() == [0]:
        reader.search_settings.has_decoy = False

    feature_df["rt"] = identification_df["Retention"] / 60.0  # convert to minutes

    feature_df["Peptide"] = identification_df["Peptide"]  # -> stripped_peptide
    feature_df["Charge"] = identification_df["Charge"]  # -> charge
    feature_df["Peptide Length"] = identification_df["Peptide Length"]  # -> peptide_length
    feature_df["Number of Missed Cleavages"] = identification_df["Number of Missed Cleavages"]  # -> missed_cleavages
    feature_df["Calculated Peptide Mass"] = identification_df["Calculated Peptide Mass"]  # -> calcmass
    feature_df["observed mass"] = identification_df["observed mass"]  # -> expmass
    feature_df["Hyperscore"] = identification_df["Hyperscore"]  # -> score

    return feature_df


def tmt_fragpipe_quant(reader, raw_identification_df: pd.DataFrame, quant_cols: list[str]) -> pd.DataFrame:
    """Pandas twin of the polars branch of ``TmtFragPipeReader._extract_quant_from_raw``."""
    quant = pd.DataFrame(index=raw_identification_df.index)
    quant["filename"] = raw_identification_df["Spectrum"].apply(lambda x: x.split(".")[0])
    quant["scan_num"] = raw_identification_df["Spectrum"].apply(lambda x: int(x.split(".")[1]))
    for col in quant_cols:
        quant[col] = raw_identification_df[col]
    quant = reader._make_unique_index(quant)
    quant = quant.drop(columns=["filename", "scan_num"])

    return quant


# --------------------------------------------------------------------------------------------------
# DELPI
# --------------------------------------------------------------------------------------------------
def delpi_identification(reader, identification_df: pd.DataFrame) -> pd.DataFrame:
    """Pandas twin of ``DelpiReader._identification_columns_polars``."""
    feature_df = pd.DataFrame(index=identification_df.index)
    feature_df["proteins"] = parse_uniprot_accession(identification_df["fasta_id"])
    feature_df["peptide"] = (
        identification_df["peptide"].str.strip("<").str.strip(">").str.strip(".").str.strip("_")
    )  # -> stripped_peptide
    feature_df["modified_sequence"] = (
        identification_df["modified_sequence"].str.strip("<").str.strip(">").str.strip(".").str.strip("_")
    )  # -> peptide
    feature_df["run_name"] = identification_df["run_name"]  # -> filename
    feature_df["frame_num"] = identification_df["frame_num"]  # -> scan_num
    feature_df["precursor_charge"] = identification_df["precursor_charge"]  # -> charge
    feature_df["sequence_length"] = identification_df["sequence_length"]  # -> peptide_length
    feature_df["posterior_error"] = identification_df["posterior_error"]  # -> PEP
    feature_df["global_precursor_q_value"] = identification_df["global_precursor_q_value"]  # -> q_value
    feature_df["score"] = identification_df["score"]
    feature_df["is_decoy"] = identification_df["is_decoy"]  # -> decoy
    feature_df["pmsm_index"] = identification_df["pmsm_index"]  # for _make_unique_index (dropped at subset)

    return feature_df
