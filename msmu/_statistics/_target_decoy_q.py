import pandas as pd


def estimate_q_values(identification_df: pd.DataFrame, decoy_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate q-values for target and decoy identifications using target-decoy competition.

    Parameters:
        identification_df: DataFrame containing target identifications with 'PEP' column.
        decoy_df: DataFrame containing decoy identifications with 'PEP' column.

    Returns:
        identification_with_q
        decoy_with_q
    """
    target_decoy = concat_target_decoy(identification_df, decoy_df)

    q_vals = compute_fdr_q(target_decoy)

    identification_df, decoy_df = retrieve_target_decoy_with_q_values(identification_df, decoy_df, q_vals)

    return identification_df, decoy_df


def concat_target_decoy(identification_df: pd.DataFrame, decoy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate target and decoy DataFrames with an 'is_decoy' column.

    Parameters:
        identification_df: DataFrame containing target identifications.
        decoy_df: DataFrame containing decoy identifications.

    Returns:
        Concatenated DataFrame with 'is_decoy' column.
    """
    identification_df = identification_df.copy()
    decoy_df = decoy_df.copy()

    identification_df["is_decoy"] = 0
    decoy_df["is_decoy"] = 1

    combined_df = pd.concat([identification_df, decoy_df], ignore_index=False)

    return combined_df


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


def retrieve_target_decoy_with_q_values(
    identification_df: pd.DataFrame, decoy_df: pd.DataFrame, q_vals: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieve target and decoy DataFrames with assigned q-values.

    Parameters:
        identification_df: DataFrame containing target identifications.
        decoy_df: DataFrame containing decoy identifications.
        q_vals: DataFrame with 'is_decoy' and 'q_value' columns.

    Returns:
        identification_with_q
        decoy_with_q
    """
    identification_with_q = identification_df.copy()
    decoy_with_q = decoy_df.copy()

    identification_with_q["q_value"] = q_vals["q_value"][q_vals["is_decoy"] == 0]
    decoy_with_q["q_value"] = q_vals["q_value"][q_vals["is_decoy"] == 1]

    return identification_with_q, decoy_with_q
