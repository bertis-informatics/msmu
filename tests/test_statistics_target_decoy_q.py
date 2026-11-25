import numpy as np
import pandas as pd
import pytest

from msmu._statistics._target_decoy_q import compute_fdr_q, estimate_q_values


def test_compute_fdr_q_bounds():
    df = pd.DataFrame({"PEP": [1, 2, 3, 4], "is_decoy": [0, 1, 0, 1]})
    q_vals = compute_fdr_q(df)
    assert np.all((q_vals["q_value"] >= 0) & (q_vals["q_value"] <= 1))


def test_estimate_q_values_scores_targets_and_decoys_of_one_table():
    identifications = pd.DataFrame(
        {"PEP": [1, 2, 3, 4, 5, 6], "decoy": [0, 1, 0, 0, 1, 0]},
        index=["t1", "d1", "t2", "t3", "d2", "t4"],
    )

    with_q = estimate_q_values(identifications)

    expected_q_values = pd.Series(
        {"t1": 2 / 3, "d1": 2 / 3, "t2": 2 / 3, "t3": 2 / 3, "d2": 3 / 4, "t4": 3 / 4}, dtype=float
    )
    assert with_q.index.equals(identifications.index)
    np.testing.assert_allclose(with_q["q_value"].to_numpy(), expected_q_values.to_numpy())
    pd.testing.assert_frame_equal(with_q.drop(columns="q_value"), identifications)
    assert "q_value" not in identifications


def test_tied_pep_is_independent_of_input_order():
    # The old row-wise calculation gives targets 0.2 or 0.4 depending on decoy order.
    df = pd.DataFrame({"PEP": [0.1] * 6, "is_decoy": [0] * 5 + [1]}, index=list("abcdef"))
    for order in (df, df.iloc[::-1], df.sample(frac=1, random_state=42)):
        actual = compute_fdr_q(order)
        assert actual.index.equals(order.index)
        np.testing.assert_array_equal(actual["q_value"], [0.4] * 6)
        assert not (actual["q_value"] < 0.3).any()


def test_group_boundaries_and_row_order():
    # Group FDRs: 1/2, 2/3, 2/5. Reverse cumulative minima: 2/5 for every group.
    # The next representable PEP must remain a separate group (FDR 3/5).
    identifications = pd.DataFrame(
        {"PEP": [0.3, 0.1, 0.2, 0.1, 0.3, np.nextafter(0.3, 1), 0.2], "decoy": [0, 0, 0, 0, 0, 1, 1]},
        index=["e", "d", "c", "b", "a", "decoy_a", "decoy_b"],
    )
    for seed in (0, 1, 2):
        shuffled = identifications.sample(frac=1, random_state=seed)
        with_q = estimate_q_values(shuffled)
        assert with_q.index.equals(shuffled.index)
        np.testing.assert_array_equal(with_q.loc[list("edcba"), "q_value"], [0.4] * 5)
        np.testing.assert_array_equal(with_q.loc[["decoy_a", "decoy_b"], "q_value"], [0.6, 0.4])


@pytest.mark.parametrize(
    ("peps", "decoys", "expected"),
    [
        ([], [], []),
        ([0.1, 0.1], [1, 1], [np.nan, np.nan]),
        ([0.1, 0.2, 0.2], [1, 0, 0], [np.nan, 1, 1]),
        ([0.1, 0.1], [0, 0], [0.5, 0.5]),
        ([0.1, np.nan, np.nan], [0, 0, 1], [1, 1, 1]),
    ],
)
def test_grouped_q_empty_and_zero_target_boundaries(peps, decoys, expected):
    actual = compute_fdr_q(pd.DataFrame({"PEP": peps, "is_decoy": decoys}))
    np.testing.assert_allclose(actual["q_value"], expected, equal_nan=True)
