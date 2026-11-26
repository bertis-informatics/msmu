import pandas as pd

from .._utils._anndata import _get_target_feature_mask


def estimate_q_values(identification_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate q-values by target-decoy competition over the rows of one table.

    Parameters:
        identification_df: DataFrame with a 'PEP' column and a 'decoy' column (1 for decoys).

    Returns:
        A copy of ``identification_df`` with a 'q_value' column for every row, targets and decoys.
    """
    target_decoy = pd.DataFrame(
        {"PEP": identification_df["PEP"].to_numpy(), "is_decoy": ~_get_target_feature_mask(identification_df)},
        index=identification_df.index,
    )

    identification_with_q = identification_df.copy()
    identification_with_q["q_value"] = compute_fdr_q(target_decoy)["q_value"].to_numpy()

    return identification_with_q


def compute_fdr_q(target_decoy: pd.DataFrame) -> pd.DataFrame:
    """
    Compute q-values at complete, exact-PEP group boundaries.

    Every member of a PEP group receives the same q-value, independent of row
    order. Missing PEPs form a final group, as in the previous NaN-last sort.
    Boundaries with no cumulative targets retain an undefined (NaN) q-value.

    Parameters:
        target_decoy: DataFrame with 'PEP' and 'is_decoy' columns.

    Returns:
        DataFrame with 'is_decoy' and 'q_value' columns in the original row order.
    """
    q_offset = 1

    groups = (
        target_decoy["is_decoy"].astype(bool)
        .groupby(target_decoy["PEP"], sort=True, dropna=False, observed=True)
        .agg(["size", "sum"])
    )
    cum_target = (groups["size"] - groups["sum"]).cumsum()
    cum_decoy = groups["sum"].cumsum()
    fdr = ((cum_decoy + q_offset) / cum_target.where(cum_target > 0)).clip(upper=1.0)
    group_q = fdr.iloc[::-1].cummin().iloc[::-1]

    result = target_decoy[["is_decoy"]].copy()
    result["q_value"] = target_decoy["PEP"].map(group_q)
    return result
