import warnings

import anndata as ad
import pandas as pd

from .._utils._mudata import add_modality, get_anndata_mod, get_mudata, get_mudata_mod_as_mutable
from .._core._provenance import log_provenance
from .._core._status import MuDataStatus
from ..logging_utils import get_logger
from ._summarisation import (
    MATRIX_ROLLUP_METHODS,
    Aggregator,
    MultisiteHandling,
    PtmSummarisationPrep,
    SummarisationPrep,
    warn_if_not_log_scale,
)
from .._statistics._target_decoy_q import estimate_q_values
from .._preprocessing._filter import add_filter, apply_filter

# for type checking only
import mudata as md
from collections.abc import Sequence
from typing import Literal

# ignore warnings in this module
warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
warnings.filterwarnings(action="ignore", message="Mean of empty slice")


logger = get_logger(__name__)

# median_polish and directlfq model per-feature response factors, which is meaningful when
# combining distinct peptides into a protein but not when combining PSMs of the same peptide
# (replicate measurements, typically 1-3 per peptide). PSM->peptide is restricted to the
# column-wise reductions; the matrix rollups belong to the peptide->protein step (to_protein).
PEPTIDE_AGG_METHODS: tuple[str, ...] = ("median", "mean", "sum")


@log_provenance
def to_peptide(
    mdata: md.MuData,
    layer: str | None = None,
    agg_method: Literal["median", "mean", "sum"] = "median",
    purity_threshold: float | None = 0.7,  # for tmt data
    top_n: int | None = None,
    rank_method: Literal["median_intensity", "total_intensity", "max_intensity", "mean_intensity"] = "median_intensity",
    calculate_q: bool = True,
) -> md.MuData:
    """Summarise PSM-level data to peptide-level data.

    Usage:
        mdata = mm.pp.to_peptide(
            mdata,
            agg_method="median",
            purity_threshold=0.7,
            calculate_q=True,
        )

    Parameters:
        mdata: MuData object containing PSM-level data.
        layer: Layer to use for quantification aggregation. If None, the default layer (.X) will be used. Defaults to None.
        agg_method: Aggregation method for quantification to use. One of "median", "mean", or "sum". Defaults to "median". The matrix rollups "median_polish" and "directlfq" are not offered here; they model per-peptide response factors and belong to the peptide-to-protein step (to_protein).
        purity_threshold: Purity threshold for TMT data quantification aggregation (does not filter out features). If None, no filtering is applied. Defaults to 0.7.
        top_n: Number of top features to consider for summarisation. If None, all features are used. Defaults to None.
        rank_method: Method to rank features when selecting top_n. Defaults to "median_intensity".
        calculate_q: Whether to calculate q-values. Defaults to True.

    Returns:
        MuData object containing peptide-level data.
    """
    if agg_method not in PEPTIDE_AGG_METHODS:
        raise ValueError(
            f"to_peptide supports agg_method in {PEPTIDE_AGG_METHODS}, got {agg_method!r}. "
            "The matrix rollups 'median_polish' and 'directlfq' model per-peptide response "
            "factors and are intended for the peptide-to-protein step (to_protein)."
        )

    mdata = mdata.copy()
    adata_to_summarise: ad.AnnData = get_anndata_mod(mdata, "psm")
    if layer is not None:
        adata_to_summarise.X = adata_to_summarise.layers[layer]
        logger.debug("Using layer '%s' for peptide summarisation.", layer)

    mstatus = MuDataStatus(mdata)
    _peptide_col: str = "peptide"
    _protein_col: str = "proteins"

    # Preparation
    summarisation_prep = SummarisationPrep(
        adata_to_summarise, col_to_groupby=_peptide_col, has_decoy=mstatus.psm.has_decoy
    )

    # Filtering for TMT purity in peptide quantification
    if mstatus.psm.label == "tmt":
        if not mstatus.psm.has_purity:
            logger.warning("Purity column not found in psm modality for TMT data. Skipping purity filtering.")
        elif purity_threshold is None:
            logger.debug("No purity threshold provided for TMT data. Skipping purity filtering.")
        else:
            summarisation_prep.filter_dict = {"purity": ("gt", purity_threshold)}

    # Ranking for top_n features
    if top_n is not None:
        summarisation_prep.rank_tuple = (
            rank_method,
            top_n,
        )  # e.g. ("total_intensity", 3)

    identification_df, quantification_df, decoy_df = summarisation_prep.prep()
    logger.debug(
        "Prepared peptide summarisation inputs: ident=%s quant=%s decoy=%s",
        identification_df.shape,
        quantification_df.shape,
        None if decoy_df is None else decoy_df.shape,
    )

    # Aggregation
    score_method: str = "best_pep"
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
                "Peptide-level identifications: %d (%d at 1%% FDR)",
                len(ident_df_agg),
                int((ident_df_agg["q_value"] < 0.01).sum()),
            )

    # Aggregate quantification data
    quant_df_agg = aggregator.aggregate_quantification()
    logger.debug("Aggregated peptide quantification shape: %s", quant_df_agg.shape)

    # build peptide-level anndata
    if (
        not mstatus.psm.has_quant and "peptide" in mstatus.mod_names
    ):  # for lfq (dda) data with peptide quantification already existing
        logger.info("Using existing peptide quantification data.")
        existing_peptide_adata = get_anndata_mod(mdata, "peptide")
        quant_df_agg = existing_peptide_adata.to_df().T
        quant_df_agg = pd.merge(
            ident_df_agg[[]],
            quant_df_agg,
            left_index=True,
            right_index=True,
            how="left",
        )
        logger.debug("Merged existing peptide quantification shape: %s", quant_df_agg.shape)
        peptide_adata = ad.AnnData(
            X=quant_df_agg.T,
            var=ident_df_agg,
        )
        mdata = get_mudata(mdata[:, [v not in existing_peptide_adata.var_names for v in mdata.var_names]].copy())
        get_mudata_mod_as_mutable(mdata)["peptide"] = peptide_adata
        # mdata["peptide"].var = ident_df_agg

    else:  # all other cases
        logger.info("Building new peptide quantification data.")
        peptide_adata = ad.AnnData(
            X=quant_df_agg.T,
            var=ident_df_agg,
        )

        # add modality
        mdata = add_modality(mdata=mdata, adata=peptide_adata, mod_name="peptide")
    peptide_mod = get_anndata_mod(mdata, "peptide")
    peptide_mod.uns["level"] = "peptide"

    if mstatus.psm.has_decoy:
        peptide_mod.uns["decoy"] = decoy_df_agg

    return mdata


@log_provenance
def to_protein(
    mdata: md.MuData,
    layer: str | None = None,
    agg_method: Literal["median", "mean", "sum", "median_polish", "directlfq"] = "median",
    top_n: int | None = 3,
    rank_method: Literal["median_intensity", "total_intensity", "max_intensity", "mean_intensity"] = "median_intensity",
    calculate_q: bool = True,
    _shared_peptide: Literal["discard"] = "discard",
) -> md.MuData:
    """Summarise peptide-level data to protein-level data. By default, uses `top 3` peptides in their `median_intensity` and `unique` (_shared_peptide = "discard") per protein_group for quantification aggregation with median.

    Parameters:
        mdata: MuData object containing Peptide-level data.
        layer: Layer to use for quantification aggregation. If None, the default layer (.X) will be used. Defaults to None.
        agg_method: Aggregation method to use. One of "median", "mean", "sum", "median_polish", or "directlfq". Defaults to "median". "median_polish" applies Tukey's median polish per protein group and "directlfq" applies the DirectLFQ rollup per protein group; both assume the quantification is in log2 space (apply log2_transform first).
        top_n: Number of top peptides to consider for summarisation. If None, all peptides are used. Defaults to None.
        rank_method: Method to rank features when selecting top_n. Defaults to "median_intensity".
        calculate_q: Whether to calculate q-values. Defaults to True.
        _shared_peptide: How to handle shared peptides. Currently only "discard" is implemented. Defaults to "discard".

    Returns:
        MuData object containing protein-level data.
    """
    original_mdata = mdata.copy()
    mdata = original_mdata.copy()

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
            on="var",
        )
        mdata = apply_filter(
            mdata=mdata,
            modality="peptide",
            on="var",
        )
    else:
        mdata = original_mdata

    adata_to_summarise: ad.AnnData = get_anndata_mod(mdata, "peptide")
    if layer is not None:
        adata_to_summarise.X = adata_to_summarise.layers[layer]
        logger.debug("Using layer '%s' for protein summarisation.", layer)

    # Preparation
    summarisation_prep = SummarisationPrep(
        adata=adata_to_summarise,
        col_to_groupby=_protein_col,
        has_decoy=mstatus.peptide.has_decoy,
    )

    # Ranking for top_n features
    if top_n is not None:
        summarisation_prep.rank_tuple = (
            rank_method,
            top_n,
        )  # e.g ("total_intensity", 3)

    identification_df, quantification_df, decoy_df = summarisation_prep.prep()
    logger.debug(
        "Prepared protein summarisation inputs: ident=%s quant=%s decoy=%s",
        identification_df.shape,
        quantification_df.shape,
        None if decoy_df is None else decoy_df.shape,
    )

    # Aggregation
    score_method: str = "best_pep"
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
                "Protein-level identifications: %d (%d at 1%% FDR)",
                len(ident_df_agg),
                int((ident_df_agg["q_value"] < 0.01).sum()),
            )

    quant_df_agg = aggregator.aggregate_quantification()
    logger.debug("Aggregated protein quantification shape: %s", quant_df_agg.shape)

    # build protein-level anndata
    protein_adata = ad.AnnData(
        X=quant_df_agg.T,
        var=ident_df_agg,
    )

    # add modality
    mdata = add_modality(mdata=original_mdata, adata=protein_adata, mod_name="protein")
    protein_mod = get_anndata_mod(mdata, "protein")
    protein_mod.uns["level"] = "protein"

    if mstatus.peptide.has_decoy:
        protein_mod.uns["decoy"] = agg_decoy_df

    return mdata


@log_provenance
def to_ptm(
    mdata: md.MuData,
    modi_name: str,
    modification: str | Sequence[str],
    layer: str | None = None,
    agg_method: Literal["median", "mean", "sum", "median_polish", "directlfq"] = "median_polish",
    top_n: int | None = None,
    rank_method: Literal["median_intensity", "total_intensity", "max_intensity", "mean_intensity"] = "median_intensity",
    multisite_handling: MultisiteHandling = "each_site",
) -> md.MuData:
    """Summarise peptide-level data to PTM-level data.

    Sites are localised from the peptide's own ``proteins`` accessions and the attached FASTA, so
    this step does not need an inferred ``protein_group`` and the resulting site ids do not depend on
    any other dataset.

    A site's position counts residues only -- the text inside a modification tag is never counted --
    so the same peptidoform yields the same site in every notation. Each peptidoform is parsed and
    checked against its ``stripped_peptide``; a notation msmu cannot read raises rather than produce
    misplaced sites.

    Parameters:
        mdata: MuData object containing peptide-level data.
        modi_name: Name of the PTM to summarise (e.g., "phospho"). Will be used in the output modality name (eg. phospho_site).
        modification: The modification tag, exactly as it appears in ``peptide`` (brackets and case
            included), or several of them to summarise into one modality. A tag may be qualified by
            its residue to match that residue only. Examples: ``"[+79.9663]"`` (Sage; the decimals
            follow the search settings), ``"(UniMod:21)"`` (DIA-NN), ``"(Phospho (STY))"``
            (MaxQuant), ``["S[167]", "T[181]", "Y[243]"]`` (FragPipe, which writes the modified
            residue's total mass). If nothing matches, the error lists the tags the data contains.
        layer: Layer to use for quantification aggregation. If None, the default layer (.X) will be used. Defaults to None.
        agg_method: Aggregation method to use. One of "median", "mean", "sum", "median_polish", or "directlfq". Defaults to "median_polish", which models a per-peptidoform effect and so is not perturbed when the set of peptidoforms supporting a site changes between samples; for a site backed by a single peptidoform it is identical to "median". "median_polish" applies Tukey's median polish per group and "directlfq" applies the DirectLFQ rollup per group; both assume the quantification is in log2 space (apply log2_transform first).
        top_n: Number of top features to consider for summarisation. If None, all features are used. Defaults to None.
        rank_method: Method to rank features when selecting top_n. Defaults to "median_intensity".
        multisite_handling: What a peptidoform carrying the target modification on several residues
            quantifies. ``"each_site"`` (default) gives its value to each of its sites, so a site
            pools singly and multiply modified peptidoforms and one measurement can reach several
            sites. ``"site_combination"`` gives it to the set of sites it carries, as one feature
            (``"P1|S5+S8"``): a multiply modified peptide's change cannot be attributed to one of
            its sites, the way a shared peptide's cannot be attributed to one protein, so it is
            reported as the group it belongs to. Every peptidoform then feeds exactly one feature,
            and the features with ``var["count_site"] == 1`` are the site table built from singly
            modified peptidoforms alone. Peptidoforms differing only in other modifications or
            missed cleavages still share a feature either way.

    Returns:
        MuData: MuData object containing PTM-level data.
    """
    adata_to_summarise: ad.AnnData = get_anndata_mod(mdata, "peptide").copy()
    if layer is not None:
        adata_to_summarise.X = adata_to_summarise.layers[layer]
        logger.debug("Using layer '%s' for PTM summarisation.", layer)

    if agg_method in MATRIX_ROLLUP_METHODS:
        warn_if_not_log_scale(adata_to_summarise.X, context=f"to_ptm(agg_method='{agg_method}')")

    modality_name = f"{modi_name}_site"
    mstatus = MuDataStatus(mdata)

    # Preparation

    if "protein_info" not in mdata.uns:
        logger.error("protein_info not found in mdata.uns. Attach fasta to mdata with mm.utils.attach_fasta().")
        raise
    summarisation_prep = PtmSummarisationPrep(
        adata_to_summarise,
        modification=modification,
        fasta=mdata.uns["protein_info"],
        multisite_handling=multisite_handling,
    )

    # Ranking for top_n features
    if top_n is not None:
        summarisation_prep.rank_tuple = (
            rank_method,
            top_n,
        )  # e.g. ("total_intensity", 3)

    identification_df, quantification_df = summarisation_prep.prep()
    logger.debug(
        "Prepared PTM summarisation inputs: ident=%s quant=%s",
        identification_df.shape,
        quantification_df.shape,
    )

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

    logger.info("%s site level identifications: %d", modi_name, len(ident_df_agg))

    # Aggregate quantification data
    quant_df_agg = aggregator.aggregate_quantification()

    # build ptm-level anndata
    logger.info("Building new %s AnnData.", modality_name)
    ptm_adata = ad.AnnData(
        X=quant_df_agg.T,
        var=ident_df_agg,
    )

    # add modality
    mdata = add_modality(mdata=mdata, adata=ptm_adata, mod_name=modality_name)
    get_anndata_mod(mdata, modality_name).uns["level"] = "ptm_site"

    return mdata
