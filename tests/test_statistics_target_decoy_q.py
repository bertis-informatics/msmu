import numpy as np
import pandas as pd

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

    assert np.allclose(target_q["q_value"].loc[expected_target.index].to_numpy(), expected_target.to_numpy())
    assert np.allclose(decoy_q["q_value"].loc[expected_decoy.index].to_numpy(), expected_decoy.to_numpy())
