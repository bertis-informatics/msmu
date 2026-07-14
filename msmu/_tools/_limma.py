import mudata as md
import pandas as pd

from .._utils._mudata import get_anndata_mod
from ..logging_utils import get_logger
from .._statistics._de_base import DeaResult, StatTestResult
from .._statistics._limma import limma_de
from .._statistics._statistics import _measure_central_tendency, _get_pct_expression

logger = get_logger(__name__)


def _run_limma(
    mdata: md.MuData,
    modality: str,
    category: str,
    ctrl: str,
    expr: str,
    interaction: str | None = None,
    interaction_levels: list | None = None,
    covariates: list[str] | None = None,
    min_pct: float = 0.5,
    layer: str | None = None,
) -> DeaResult:
    """
    Run limma moderated-t differential expression on a MuData modality.

    Internal engine behind ``run_de(stat_method="limma")``. Positive log2 fold change
    means higher in ``expr``. With ``interaction`` unset this tests the main effect
    ``expr - ctrl``. With ``interaction`` set it tests the interaction /
    difference-in-differences of ``expr - ctrl`` across two levels of the second
    factor, e.g. "does the drug effect differ by genotype". The design matrix and
    contrast are built internally; the caller never specifies raw contrast coefficients.
    """
    mod_adata = get_anndata_mod(mdata, modality)
    if layer is not None:
        data = pd.DataFrame(
            mod_adata.layers[layer],
            index=mod_adata.obs_names,
            columns=mod_adata.var_names,
        )
    else:
        data = mod_adata.to_df()
    obs = mod_adata.obs
    expr_matrix = data.T  # features x samples

    result = limma_de(
        expr_matrix=expr_matrix,
        obs=obs,
        category=category,
        ctrl=ctrl,
        expr=expr,
        interaction=interaction,
        interaction_levels=interaction_levels,
        covariates=covariates,
        min_pct=min_pct,
    )
    logger.debug(
        "limma DEA for modality '%s' contrast '%s' produced %d features.",
        modality,
        result.contrast_label,
        result.features.size,
    )

    test_res = StatTestResult(stat_method="limma", p_value=result.p_value, q_value=result.q_value)
    de_res = DeaResult(test_res)
    de_res.stat_method = "limma"
    de_res.ctrl = ctrl
    de_res.expr = expr
    de_res.features = result.features
    de_res.log2fc = result.log2fc
    de_res.contrast_label = result.contrast_label

    # Representative central tendency / detection percentage for the pooled expr-vs-ctrl
    # groups, for volcano hover context (well-defined for main effect and interaction alike).
    ctrl_samples = obs.index[obs[category] == ctrl]
    expr_samples = obs.index[obs[category] == expr]
    ctrl_arr = data.loc[ctrl_samples].to_numpy()
    expr_arr = data.loc[expr_samples].to_numpy()
    de_res.repr_ctrl = _measure_central_tendency(ctrl_arr, "median")
    de_res.repr_expr = _measure_central_tendency(expr_arr, "median")
    de_res.pct_ctrl = _get_pct_expression(ctrl_arr)
    de_res.pct_expr = _get_pct_expression(expr_arr)

    return de_res
