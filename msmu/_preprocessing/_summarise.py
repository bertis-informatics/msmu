import warnings
import logging

import anndata as ad
import pandas as pd

from .._utils.utils import uns_logger
from .._read_write._reader_utils import add_modality
from .._read_write._mdata_status import MuDataStatus
from ._summarisation import SummarisationPrep, PtmSummarisationPrep, Aggregator
from .._statistics._target_decoy_q import estimate_q_values
from .._preprocessing._filter import add_filter, apply_filter

# for type checking only
import mudata as md
from typing import Literal

# ignore warnings in this module
warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
warnings.filterwarnings(action="ignore", message="Mean of empty slice")


logger = logging.getLogger(__name__)


@uns_logger
def to_peptide(
    mdata: md.MuData,
    layer: str | None = None,
    agg_method: Literal["median", "mean", "sum"] = "median",
    score_method: Literal["best_pep"] = "best_pep",
    purity_threshold: float | None = 0.7,  # for tmt data
    top_n: int | None = None,
    rank_method: Literal["total_intensity", "max_intensity", "median_intensity"] = "total_intensity",
    calculate_q: bool = True,
) -> md.MuData:
    """Summarise PSM-level data to peptide-level data.

    Usage:
        mdata = mm.pp.to_peptide(
            mdata,
            agg_method="median",
            calculate_q=True,
            score_method="best_pep",
            purity_threshold=0.7,
        )

    Parameters:
        mdata: MuData object containing PSM-level data.
        layer: Layer to use for quantification aggregation. If None, the default layer (.X) will be used. Defaults to None.
        agg_method: Aggregation method for quantification to use. Defaults to "median".
        calculate_q: Whether to calculate q-values. Defaults to True.
        score_method: Method to combine scores. Defaults to "best_pep".
        purity_threshold: Purity threshold for TMT data quantification aggregation (does not filter out features). If None, no filtering is applied. Defaults to 0.7.
        top_n: Number of top features to consider for summarisation. If None, all features are used. Defaults to None.
        rank_method: Method to rank features when selecting top_n. Defaults to "total_intensity".

    Returns:
        MuData object containing peptide-level data.
    """
    adata_to_summarise: ad.AnnData = mdata["psm"].copy()
    if layer is not None:
        adata_to_summarise.X = adata_to_summarise.layers[layer]

    mstatus = MuDataStatus(mdata)
    _peptide_col: str = "peptide"
    _protein_col: str = "proteins"

    # Preparation
    summarisation_prep = SummarisationPrep(
        adata_to_summarise, col_to_groupby=_peptide_col, has_decoy=mstatus.psm.has_decoy
    )

    # Filtering for TMT purity in peptide quantification
    if mstatus.psm.label == "tmt":
        if mstatus.psm.has_purity == False:
            logger.warning("Purity column not found in psm modality for TMT data. Skipping purity filtering.")
        elif purity_threshold is None:
            logger.info("No purity threshold provided. Skipping purity filtering.")
        else:
            summarisation_prep.filter_dict = {"purity": ("gt", purity_threshold)}

    # Ranking for top_n features
    if top_n is not None:
        summarisation_prep.rank_tuple = (rank_method, top_n)  # e.g. ("total_intensity", 3)

    identification_df, quantification_df, decoy_df = summarisation_prep.prep()

    # Aggregation
    aggregator = Aggregator.peptide(
        identification_df=identification_df,
        quantification_df=quantification_df,
        decoy_df=decoy_df,
        agg_method=agg_method,
        score_method=score_method,
        protein_col=_protein_col,
        peptide_col=_peptide_col,
    )

    # Aggregate identification and quantification data
    if mstatus.psm.has_var:
        ident_df_agg = aggregator.aggregate_identification()
    else:
        logger.error("var is empty in psm modality. Cannot aggregate identification data.")
        raise

    # Aggregate decoy data if present
    if mstatus.psm.has_decoy:
        decoy_df_agg = aggregator.aggregate_decoy()
    else:
        logger.warning("Decoy data not found. Skipping decoy aggregation.")

    # q-value calculation
    if calculate_q:
        if mstatus.psm.has_decoy is False:
            logger.warning("Decoy data not found. Skipping q-value calculation.")
        elif mstatus.psm.has_pep is False:
            logger.warning("PEP column not found in identification data. Skipping q-value calculation.")
        else:
            ident_df_agg, decoy_df_agg = estimate_q_values(
                identification_df=ident_df_agg,
                decoy_df=decoy_df_agg,
            )
            logger.info(
                f"Peptide-level identifications: {len(ident_df_agg)} ({sum(ident_df_agg['q_value'] < 0.01)} at 1% FDR)"
            )

    # Aggregate quantification data
    quant_df_agg = aggregator.aggregate_quantification()

    # build peptide-level anndata
    if (mstatus.psm.has_quant == False) & (
        "peptide" in mstatus.mod_names
    ):  # for lfq (dda) data with peptide quantification already existing
        print("Using existing peptide quantification data.")
        quant_df_agg = mdata["peptide"].to_df().T
        quant_df_agg = pd.merge(ident_df_agg[[]], quant_df_agg, left_index=True, right_index=True, how="left")
        peptide_adata = ad.AnnData(
            X=quant_df_agg.T,
            var=ident_df_agg,
        )
        mdata = mdata[:, [v not in mdata["peptide"].var_names for v in mdata.var_names]].copy()
        mdata.mod["peptide"] = peptide_adata
        # mdata["peptide"].var = ident_df_agg

    else:  # all other cases
        print("Building new peptide quantification data.")
        peptide_adata = ad.AnnData(
            X=quant_df_agg.T,
            var=ident_df_agg,
        )

        # add modality
        mdata = add_modality(mdata=mdata, adata=peptide_adata, mod_name="peptide", parent_mods=["psm"])
    mdata["peptide"].uns["level"] = "peptide"

    if mstatus.psm.has_decoy:
        mdata["peptide"].uns["decoy"] = decoy_df_agg

    return mdata


@uns_logger
def to_protein(
    mdata: md.MuData,
    layer: str | None = None,
    agg_method: Literal["median", "mean", "sum"] = "median",
    score_method: Literal["best_pep"] = "best_pep",
    top_n: int | None = 3,
    rank_method: Literal["total_intensity", "max_intensity", "median_intensity"] = "total_intensity",
    calculate_q: bool = True,
    _shared_peptide: Literal["discard"] = "discard",
) -> md.MuData:
    """Summarise peptide-level data to protein-level data. By default, uses `top 3` peptides in their `total_intensity` and `unique` (_shared_peptide = "discard") per protein_group for quantification aggregation with median.

    Parameters:
        mdata: MuData object containing Peptide-level data.
        layer: Layer to use for quantification aggregation. If None, the default layer (.X) will be used. Defaults to None.
        agg_method: Aggregation method to use. Defaults to "median".
        calculate_q: Whether to calculate q-values. Defaults to True.
        score_method: Method to combine scores (PEP). Defaults to "best_pep".
        top_n: Number of top peptides to consider for summarisation. If None, all peptides are used. Defaults to None.
        rank_method: Method to rank features when selecting top_n. Defaults to "total_intensity".
        _shared_peptide: How to handle shared peptides. Currently only "discard" is implemented. Defaults to "discard".

    Returns:
        MuData object containing protein-level data.
    """
    original_mdata = mdata.copy()

    mstatus = MuDataStatus(original_mdata)
    _protein_col: str = "protein_group"

    # Handle shared peptides
    # use unique peptides only
    if _shared_peptide == "discard":
        mdata = add_filter(
            mdata=original_mdata,
            modality="peptide",
            column="peptide_type",
            keep="eq",
            value="unique",
        )
        mdata = apply_filter(
            mdata=mdata,
            modality="peptide",
        )
    else:
        mdata = original_mdata

    adata_to_summarise: ad.AnnData = mdata["peptide"].copy()
    if layer is not None:
        adata_to_summarise.X = adata_to_summarise.layers[layer]

    # Preparation
    summarisation_prep = SummarisationPrep(
        adata=adata_to_summarise, col_to_groupby=_protein_col, has_decoy=mstatus.peptide.has_decoy
    )

    # Ranking for top_n features
    if top_n is not None:
        summarisation_prep.rank_tuple = (rank_method, top_n)  # e.g ("total_intensity", 3)

    identification_df, quantification_df, decoy_df = summarisation_prep.prep()

    # Aggregation
    aggregator = Aggregator.protein(
        identification_df=identification_df,
        quantification_df=quantification_df,
        decoy_df=decoy_df,
        agg_method=agg_method,
        score_method=score_method,
        protein_col=_protein_col,
    )

    # Aggregate identification
    ident_df_agg = aggregator.aggregate_identification()
    if mstatus.peptide.has_decoy:
        agg_decoy_df = aggregator.aggregate_decoy()
    else:
        logger.warning("Decoy data not found. Skipping decoy aggregation.")

    # q-value calculation
    if calculate_q:
        if mstatus.peptide.has_decoy is False:
            logger.warning("Decoy data not found. Skipping q-value calculation.")
        elif mstatus.peptide.has_pep is False:
            logger.warning("PEP column not found in identification data. Skipping q-value calculation.")
        else:
            ident_df_agg, agg_decoy_df = estimate_q_values(
                identification_df=ident_df_agg,
                decoy_df=agg_decoy_df,
            )
            logger.info(
                f"Protein-level identifications :  {len(ident_df_agg)} ({sum(ident_df_agg['q_value'] < 0.01)} at 1% FDR)"
            )

    quant_df_agg = aggregator.aggregate_quantification()

    # build protein-level anndata
    protein_adata = ad.AnnData(
        X=quant_df_agg.T,
        var=ident_df_agg,
    )

    # add modality
    mdata = add_modality(mdata=original_mdata, adata=protein_adata, mod_name="protein", parent_mods=["peptide"])
    mdata["protein"].uns["level"] = "protein"

    if mstatus.peptide.has_decoy:
        mdata["protein"].uns["decoy"] = agg_decoy_df

    return mdata


@uns_logger
def to_ptm(
    mdata: md.MuData,
    modi_name: str,
    modification: str,
    layer: str | None = None,
    agg_method: Literal["median", "mean", "sum"] = "median",
    top_n: int | None = None,
    rank_method: Literal["total_intensity", "max_intensity"] = "total_intensity",
) -> md.MuData:
    """Summarise peptide-level data to PTM-level data.

    Parameters:
        mdata: MuData object containing peptide-level data.
        modi_name: Name of the PTM to summarise (e.g., "phospho"). Will be used in the output modality name (eg. phospho_site).
        modification: Modification string (e.g., "[+79.96633]", "(unimod:21)").
        layer: Layer to use for quantification aggregation. If None, the default layer (.X) will be used. Defaults to None.
        agg_method: Aggregation method to use. Defaults to "median".
        top_n: Number of top features to consider for summarisation. If None, all features are used. Defaults to None.
        rank_method: Method to rank features when selecting top_n. Defaults to "total_intensity".

    Returns:
        MuData: MuData object containing PTM-level data.
    """
    adata_to_summarise: ad.AnnData = mdata["peptide"].copy()
    if layer is not None:
        adata_to_summarise.X = adata_to_summarise.layers[layer]

    modality_name = f"{modi_name}_site"
    mstatus = MuDataStatus(mdata)

    # Preparation

    if "protein_info" not in mdata.uns:
        logger.error("protein_info not found in mdata.uns. Attach fasta to mdata with mm.utils.attach_fasta().")
        raise
    summarisation_prep = PtmSummarisationPrep(
        adata_to_summarise,
        modi_identifier=modification,
        fasta=mdata.uns["protein_info"],
    )

    # Ranking for top_n features
    if top_n is not None:
        summarisation_prep.rank_tuple = (rank_method, top_n)  # e.g. ("total_intensity", 3)

    identification_df, quantification_df = summarisation_prep.prep()

    # Aggregation
    aggregator = Aggregator.ptm_site(
        identification_df=identification_df,
        quantification_df=quantification_df,
        agg_method=agg_method,
    )

    # Aggregate identification and quantification data
    if mstatus.peptide.has_var:
        ident_df_agg = aggregator.aggregate_identification()
    else:
        logger.error("var is empty in peptide modality. Cannot aggregate identification data.")
        raise

    logger.info(f"{modi_name} site level identifications: {len(ident_df_agg)}")

    # Aggregate quantification data
    quant_df_agg = aggregator.aggregate_quantification()

    # build ptm-level anndata
    logger.info(f"Building new {modality_name} AnnData.")
    ptm_adata = ad.AnnData(
        X=quant_df_agg.T,
        var=ident_df_agg,
    )

    # add modality
    mdata = add_modality(mdata=mdata, adata=ptm_adata, mod_name=modality_name, parent_mods=["peptide"])
    mdata[modality_name].uns["level"] = "ptm_site"

    return mdata
