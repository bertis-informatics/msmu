import logging

import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from mudata import MuData

from msmu._core._blockdiag import to_dense_df, to_observed_sparse
from msmu._preprocessing._batch_correction import correct_batch_effect


def _mdata_with_batch(x, obs_names, var_names, batch) -> MuData:
    # correct_batch_effect reads the batch label from the MuData-level ``mdata.obs``.
    mdata = MuData({"psm": AnnData(X=x, obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names))})
    mdata.obs["batch"] = np.asarray(batch)
    return mdata


def test_correct_batch_effect_gis_drops_samples(simple_mdata):
    out = correct_batch_effect(
        simple_mdata,
        modality="psm",
        method="gis",
        category="batch",
        gis_samples=["gis1", "gis2"],
        drop_gis=True,
    )
    assert out["psm"].n_obs == 4  # 2 GIS samples dropped


def test_correct_batch_effect_gis_keep_samples(simple_mdata):
    out = correct_batch_effect(
        simple_mdata,
        modality="psm",
        method="gis",
        category="batch",
        gis_samples=["gis1", "gis2"],
        drop_gis=False,
    )
    assert out["psm"].n_obs == 6  # 2 GIS samples kept


def test_correct_batch_effect_gis(simple_mdata):
    # Both features appear in both batches, so the IRS scale is restored: the reference-relative
    # correction ([[-5, -4], ...]) plus the per-feature geomean reference target [8.5, 9.0].
    out = correct_batch_effect(
        simple_mdata,
        modality="psm",
        method="gis",
        category="batch",
        gis_samples=["gis1", "gis2"],
        log_transformed=True,
        drop_gis=False,
    )
    assert np.allclose(
        out["psm"].X,
        np.array(
            [
                [3.5, 5.0],
                [5.5, 7.0],
                [8.5, 9.0],
                [4.5, 5.0],
                [6.5, 7.0],
                [8.5, 9.0],
            ]
        ),
    )


def test_correct_batch_effect_gis_protein(simple_mdata_protein):
    out = correct_batch_effect(
        simple_mdata_protein,
        modality="protein",
        method="gis",
        category="batch",
        gis_samples=["gis1", "gis2"],
        log_transformed=True,
        drop_gis=False,
    )
    # protein reference target (geomean of per-batch GIS levels) = [(5 + 11) / 2, (6 + 12) / 2] = [8, 9]
    assert np.allclose(
        out["protein"].X,
        np.array(
            [
                [4.0, 5.0],
                [6.0, 7.0],
                [8.0, 9.0],
                [4.0, 5.0],
                [6.0, 7.0],
                [8.0, 9.0],
            ]
        ),
    )
    # correcting the protein modality leaves psm untouched
    assert np.allclose(
        out["psm"].X,
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
                [11.0, 12.0],
            ]
        ),
    )


def test_correct_batch_effect_gis_shared_restores_orphan():
    # The matrix has cross-batch structure (v1 has a GIS reference in both batches), so every feature
    # is restored -- including v2, whose reference exists only in b1. v2 is restored against its own
    # b1 level (back to its abundance) instead of being stranded at the reference-relative origin, so
    # it stays on the same scale as v1. Pins the global (any-vs-none) restore decision.
    x = np.array(
        [
            [1.0, 10.0],  # s1   (b1)
            [4.0, 8.0],  # gis1 (b1 reference)
            [5.0, 20.0],  # s2   (b2)
            [12.0, np.nan],  # gis2 (b2 reference; v2 unobserved)
        ]
    )
    mdata = _mdata_with_batch(x, ["s1", "gis1", "s2", "gis2"], ["v1", "v2"], ["b1", "b1", "b2", "b2"])
    out = correct_batch_effect(
        mdata,
        modality="psm",
        method="gis",
        category="batch",
        gis_samples=["gis1", "gis2"],
        drop_gis=False,
    )
    # v1 target 8 (mean of 4, 12); v2 target 8 (its only b1 level) -> both on the abundance scale.
    expected = np.array(
        [
            [5.0, 10.0],
            [8.0, 8.0],
            [1.0, np.nan],
            [8.0, np.nan],
        ]
    )
    np.testing.assert_allclose(out["psm"].X, expected, equal_nan=True)


def test_correct_batch_effect_median_center(simple_mdata):
    # Both features in both batches -> restored: per-batch-median correction plus the per-feature
    # overall median [6.5, 7.0].
    out = correct_batch_effect(
        simple_mdata,
        modality="psm",
        method="median_center",
        category="batch",
        log_transformed=True,
    )
    assert np.allclose(
        out["psm"].X,
        np.array(
            [
                [4.5, 5.0],
                [6.5, 7.0],
                [9.5, 9.0],
                [4.5, 5.0],
                [6.5, 7.0],
                [8.5, 9.0],
            ]
        ),
    )


def test_correct_batch_effect_median_center_blockdiag_skips_restore():
    # Fully block-diagonal: every feature lives in one batch only, so no cross-batch centre exists.
    # Restore must be skipped -- the output stays centred (reference-relative), NaN never filled.
    x = np.array(
        [
            [1.0, 2.0, np.nan, np.nan],  # s1 (b1)
            [3.0, 4.0, np.nan, np.nan],  # s2 (b1)
            [np.nan, np.nan, 5.0, 6.0],  # s3 (b2)
            [np.nan, np.nan, 7.0, 8.0],  # s4 (b2)
        ]
    )
    mdata = _mdata_with_batch(x, ["s1", "s2", "s3", "s4"], ["v1", "v2", "v3", "v4"], ["b1", "b1", "b2", "b2"])
    out = correct_batch_effect(mdata, modality="psm", method="median_center", category="batch")
    expected = np.array(
        [
            [-1.0, -1.0, np.nan, np.nan],
            [1.0, 1.0, np.nan, np.nan],
            [np.nan, np.nan, -1.0, -1.0],
            [np.nan, np.nan, 1.0, 1.0],
        ]
    )
    np.testing.assert_allclose(out["psm"].X, expected, equal_nan=True)


def test_correct_batch_effect_median_center_shared_restores_orphan():
    # v1 spans both batches; v2 is only in b1. Because the matrix has cross-batch structure (v1),
    # every feature is restored -- v2 too, against its own overall median (back to its abundance)
    # rather than left at the centred origin, so it stays on v1's scale. Global restore decision.
    x = np.array(
        [
            [1.0, 10.0],  # s1 (b1)
            [3.0, 12.0],  # s2 (b1)
            [5.0, np.nan],  # s3 (b2)
            [7.0, np.nan],  # s4 (b2)
        ]
    )
    mdata = _mdata_with_batch(x, ["s1", "s2", "s3", "s4"], ["v1", "v2"], ["b1", "b1", "b2", "b2"])
    out = correct_batch_effect(mdata, modality="psm", method="median_center", category="batch")
    # v1 target 4, v2 target 11 (its overall median) -> v2 back to ~its abundance, not the origin.
    expected = np.array(
        [
            [3.0, 10.0],
            [5.0, 12.0],
            [3.0, np.nan],
            [5.0, np.nan],
        ]
    )
    np.testing.assert_allclose(out["psm"].X, expected, equal_nan=True)


def test_correct_batch_effect_gis_missing_raises(simple_mdata):
    with pytest.raises(ValueError, match="as GIS not found in obs"):
        correct_batch_effect(
            simple_mdata,
            modality="psm",
            method="gis",
            category="batch",
            gis_samples=["gis4", "gis5"],
        )


def test_correct_batch_effect_invalid_method(simple_mdata):
    with pytest.raises(ValueError, match="not recognised"):
        correct_batch_effect(
            simple_mdata,
            modality="psm",
            category="batch",
            method="nope",
        )


def test_correct_batch_effect_combat(simple_mdata):
    out = correct_batch_effect(
        simple_mdata,
        modality="psm",
        method="combat",
        category="batch",
        log_transformed=True,
    )
    assert out["psm"].X.shape == (6, 2)


def test_correct_batch_effect_continuous(simple_mdata):
    out = correct_batch_effect(
        simple_mdata,
        modality="psm",
        method="continuous",
        category="batch",
        log_transformed=True,
    )
    assert out["psm"].X.shape == (6, 2)


def test_correct_batch_effect_gis_no_reference_feature_dropped(caplog):
    # v2 has NO GIS reference in any batch (gis1/gis2 are NaN for it) though s1/s2 carry values, so
    # the correction subtracts a missing reference and v2 becomes NaN -- reported at the correction
    # as a warning, not as a restore "reference-relative" scale choice. v1 (reference in both) is
    # restored normally.
    x = np.array(
        [
            [1.0, 5.0],  # s1   (b1)
            [4.0, np.nan],  # gis1 (b1 reference; v2 unobserved)
            [5.0, 6.0],  # s2   (b2)
            [12.0, np.nan],  # gis2 (b2 reference; v2 unobserved)
        ]
    )
    mdata = _mdata_with_batch(x, ["s1", "gis1", "s2", "gis2"], ["v1", "v2"], ["b1", "b1", "b2", "b2"])

    # The package logger does not propagate, so attach caplog's handler directly to capture the warning.
    batch_logger = logging.getLogger("msmu._preprocessing._batch_correction")
    batch_logger.addHandler(caplog.handler)
    previous_level = batch_logger.level
    batch_logger.setLevel(logging.WARNING)
    try:
        out = correct_batch_effect(
            mdata,
            modality="psm",
            method="gis",
            category="batch",
            gis_samples=["gis1", "gis2"],
            drop_gis=False,
        )
    finally:
        batch_logger.removeHandler(caplog.handler)
        batch_logger.setLevel(previous_level)

    # v1 restored to target 8 (mean of 4, 12); v2 has no reference in any batch -> all NaN.
    expected = np.array(
        [
            [5.0, np.nan],
            [8.0, np.nan],
            [1.0, np.nan],
            [8.0, np.nan],
        ]
    )
    np.testing.assert_allclose(out["psm"].X, expected, equal_nan=True)
    assert "no GIS reference" in caplog.text


@pytest.mark.parametrize(
    "method, extra_kwargs",
    [
        ("gis", {"gis_samples": ["gis1", "gis2"], "drop_gis": False}),
        ("median_center", {}),
    ],
)
def test_correct_batch_effect_sparse_matches_dense_and_stays_sparse(method, extra_kwargs):
    """gis / median_center apply the correction subtractively, so a sparse input must yield a sparse
    output equal to the dense result (absent cells preserved, never densified out)."""
    x = np.array(
        [
            [1.0, 10.0],  # s1   (b1)
            [4.0, 8.0],  # gis1 (b1 reference)
            [5.0, 20.0],  # s2   (b2)
            [12.0, np.nan],  # gis2 (b2 reference; v2 unobserved)
        ]
    )
    obs_names, var_names, batch = ["s1", "gis1", "s2", "gis2"], ["v1", "v2"], ["b1", "b1", "b2", "b2"]

    dense_out = correct_batch_effect(
        _mdata_with_batch(x, obs_names, var_names, batch), modality="psm", method=method, category="batch", **extra_kwargs
    )
    sparse_out = correct_batch_effect(
        _mdata_with_batch(to_observed_sparse(x, dtype=np.float64), obs_names, var_names, batch),
        modality="psm",
        method=method,
        category="batch",
        **extra_kwargs,
    )

    assert sp.issparse(sparse_out["psm"].X), f"{method} should re-sparsify a sparse input"
    np.testing.assert_allclose(
        to_dense_df(sparse_out["psm"]).to_numpy(), to_dense_df(dense_out["psm"]).to_numpy(), equal_nan=True
    )


def test_correct_batch_effect_combat_densifies_sparse_input(simple_mdata):
    """ComBat replaces the whole array (may fill NaN), so it is excluded from re-sparsify: a sparse
    input yields a dense output. Pins the intended boundary."""
    mdata = simple_mdata.copy()
    mdata.mod["psm"].X = to_observed_sparse(np.asarray(mdata.mod["psm"].X, dtype=np.float64))
    out = correct_batch_effect(mdata, modality="psm", method="combat", category="batch", log_transformed=True)
    assert not sp.issparse(out["psm"].X)
