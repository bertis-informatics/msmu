import numpy as np
import pandas as pd


def estimate_q_values(identification_df: pd.DataFrame, decoy_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate q-values for target and decoy identifications using target-decoy competition.

    Parameters:
        identification_df: DataFrame containing target identifications with 'score' column.
        decoy_df: DataFrame containing decoy identifications with 'score' column.

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
    Compute FDR and q-values for the picked target-decoy pairs.

    Parameters:
        target_decoy: DataFrame with 'score' and 'is_decoy' columns.

    Returns:
        DataFrame with 'is_decoy' and 'q_value' columns.
    """
    q_offset = 1

    df = target_decoy.sort_values("PEP", ascending=True)

    # 누적 타겟/데코이 수
    df["cum_target"] = (~df["is_decoy"].astype(bool)).cumsum()
    df["cum_decoy"] = (df["is_decoy"].astype(bool)).cumsum()

    # FDR 계산
    df["fdr"] = np.nan
    valid = df["cum_target"] > 0
    df.loc[valid, "fdr"] = (df.loc[valid, "cum_decoy"] + q_offset) / df.loc[valid, "cum_target"]

    # FDR과 q-value는 확률로 clip (0~1)
    df["fdr"] = df["fdr"].clip(upper=1.0)
    df["q_value"] = df["fdr"].iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)

    return df[["is_decoy", "q_value"]]


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
