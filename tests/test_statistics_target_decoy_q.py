import numpy as np
import pandas as pd
import pytest

from msmu._statistics._target_decoy_q import compute_fdr_q, estimate_q_values


def test_compute_fdr_q_bounds():
    df = pd.DataFrame({"PEP": [1, 2, 3, 4], "is_decoy": [0, 1, 0, 1]})
    q_vals = compute_fdr_q(df)
    assert np.all((q_vals["q_value"] >= 0) & (q_vals["q_value"] <= 1))


def test_estimate_q_values():
    target = pd.DataFrame({"PEP": [1, 3, 4, 6]}, index=["t1", "t2", "t3", "t4"])
    decoy = pd.DataFrame({"PEP": [2, 5]}, index=["d1", "d2"])
    target_q, decoy_q = estimate_q_values(target, decoy)

    expected_target = pd.Series(
        {
            "t1": 2 / 3,
            "t2": 2 / 3,
            "t3": 2 / 3,
            "t4": 3 / 4,
        },
        dtype=float,
    )
    expected_decoy = pd.Series(
        {
            "d1": 2 / 3,
            "d2": 3 / 4,
        },
        dtype=float,
    )

    assert np.allclose(
        target_q["q_value"].loc[expected_target.index].to_numpy(),
        expected_target.to_numpy(),
    )
    assert np.allclose(
        decoy_q["q_value"].loc[expected_decoy.index].to_numpy(),
        expected_decoy.to_numpy(),
    )


def test_tied_pep_is_independent_of_input_order():
    # The old row-wise calculation gives targets 0.2 or 0.4 depending on decoy order.
    df = pd.DataFrame({"PEP": [0.1] * 6, "is_decoy": [0] * 5 + [1]}, index=list("abcdef"))
    for order in (df, df.iloc[::-1], df.sample(frac=1, random_state=42)):
        actual = compute_fdr_q(order)
        assert actual.index.equals(order.index)
        np.testing.assert_array_equal(actual["q_value"], [0.4] * 6)
        assert not (actual["q_value"] < 0.3).any()


def test_group_boundaries_and_original_target_decoy_indices():
    # Group FDRs: 1/2, 2/3, 2/5. Reverse cumulative minima: 2/5 for every group.
    # The next representable PEP must remain a separate group (FDR 3/5).
    target = pd.DataFrame({"PEP": [0.3, 0.1, 0.2, 0.1, 0.3]}, index=list("edcba"))
    decoy = pd.DataFrame({"PEP": [np.nextafter(0.3, 1), 0.2]}, index=["a", "b"])
    for seed in (0, 1, 2):
        targets = target.sample(frac=1, random_state=seed)
        decoys = decoy.sample(frac=1, random_state=seed)
        target_q, decoy_q = estimate_q_values(targets, decoys)
        assert target_q.index.equals(targets.index)
        assert decoy_q.index.equals(decoys.index)
        np.testing.assert_array_equal(target_q["q_value"], [0.4] * 5)
        np.testing.assert_array_equal(decoy_q.loc[["a", "b"], "q_value"], [0.6, 0.4])
    assert "q_value" not in target and "q_value" not in decoy


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
