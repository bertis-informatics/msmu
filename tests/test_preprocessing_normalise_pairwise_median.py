"""``normalise(method="pairwise_median")``: samples are aligned on the features they observe in
common, so a sample that lost its low-abundance features is not pushed down for it (which is what
``median`` does, centring every sample on the median of whatever it observed).

Also covers what every ``normalise`` call now records in ``adata.uns["normalisation"]``.
"""

from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData
from mudata import MuData

from msmu._core._blockdiag import to_dense_df
from msmu._preprocessing import _normalisation
from msmu._preprocessing._normalisation import (
    UnsharedSamplePairsError,
    estimate_pairwise_median_shifts,
    normalise_pairwise_median,
)
from msmu._preprocessing._normalise import normalise

SAMPLES = [f"s{i}" for i in range(1, 7)]
OFFSETS = np.array([0.5, -0.3, 0.0, 0.8, -1.0, 0.0])
FEATURE_COUNT = 400


def _log2_values(seed: int = 0, noise_sd: float = 0.0, missing_fraction: float = 0.3) -> np.ndarray:
    """(samples x features) log2 matrix: feature levels + per-sample offset (+ noise), MCAR missing."""
    rng = np.random.default_rng(seed)
    feature_levels = rng.normal(20.0, 2.0, size=FEATURE_COUNT)
    values = feature_levels[None, :] + OFFSETS[:, None] + rng.normal(0.0, noise_sd, size=(len(SAMPLES), FEATURE_COUNT))
    values[rng.random(values.shape) < missing_fraction] = np.nan
    return values


def _sparse_with_absent_cells(dense: np.ndarray) -> sp.csr_matrix:
    rows, cols = np.nonzero(~np.isnan(dense))
    return sp.csr_matrix((dense[rows, cols], (rows, cols)), shape=dense.shape)


def _mdata(layer_array, sample_names=SAMPLES) -> MuData:
    feature_count = layer_array.shape[1]
    adata = AnnData(
        X=np.zeros((len(sample_names), feature_count)),
        obs=pd.DataFrame({"batch": ["a", "a", "a", "b", "b", "b"][: len(sample_names)]}, index=list(sample_names)),
        var=pd.DataFrame({"fraction": ["f1"] * (feature_count // 2) + ["f2"] * (feature_count - feature_count // 2)},
                         index=[f"v{j}" for j in range(feature_count)]),
    )
    adata.layers["raw"] = layer_array
    return MuData({"peptide": adata})


def _normalised_layer(mdata: MuData) -> np.ndarray:
    return to_dense_df(mdata["peptide"], layer="raw").to_numpy()


def _applied_shifts(mdata: MuData) -> np.ndarray:
    return mdata["peptide"].uns["normalisation"]["blocks"]["raw|all|all"]["shift_log2"]


def test_fixture_pairs_all_share_features():
    values = _log2_values()
    shared = (~np.isnan(values)).astype(float) @ (~np.isnan(values)).astype(float).T
    assert shared[~np.eye(len(SAMPLES), dtype=bool)].min() > 0


def test_recovers_noise_free_offsets_exactly_with_zero_sum():
    out = normalise(_mdata(_log2_values()), method="pairwise_median", modality="peptide", layer="raw")
    shifts = _applied_shifts(out)
    np.testing.assert_allclose(shifts, OFFSETS - OFFSETS.mean(), atol=1e-10)
    assert abs(shifts.sum()) < 1e-10


def test_shift_is_the_equal_weight_least_squares_solution():
    """The row mean of the pair-median matrix must equal the least-squares fit of
    ``f_s - f_t = m_st`` over all pairs with ``sum(f) = 0`` (n in the denominator, not n - 1)."""
    estimate = estimate_pairwise_median_shifts(_log2_values(noise_sd=0.2))
    sample_count = len(SAMPLES)
    rows, targets = [], []
    for first in range(sample_count):
        for second in range(first + 1, sample_count):
            row = np.zeros(sample_count)
            row[first], row[second] = 1.0, -1.0
            rows.append(row)
            targets.append(estimate.pair_median_log2[first, second])
    rows.append(np.ones(sample_count))
    targets.append(0.0)
    least_squares_shifts = np.linalg.lstsq(np.array(rows), np.array(targets), rcond=None)[0]
    np.testing.assert_allclose(estimate.shift_log2, least_squares_shifts, atol=1e-10)


def test_recovers_shift_of_truncated_sample_where_median_misses():
    """One sample loses its lowest 30% of features and is shifted by -1.0. ``pairwise_median``
    recovers the shift; ``median`` reads the truncated sample as high and misses by more than 0.2."""
    rng = np.random.default_rng(1)
    feature_levels = rng.normal(20.0, 2.0, size=FEATURE_COUNT)
    values = feature_levels[None, :] + rng.normal(0.0, 0.1, size=(len(SAMPLES), FEATURE_COUNT))
    truncated_row = 0
    cutoff = np.quantile(feature_levels, 0.3)
    values[truncated_row, feature_levels < cutoff] = np.nan
    values[truncated_row] -= 1.0
    expected = np.zeros(len(SAMPLES))
    expected[truncated_row] = -1.0
    expected -= expected.mean()

    pairwise = normalise(_mdata(values.copy()), method="pairwise_median", modality="peptide", layer="raw")
    median = normalise(_mdata(values.copy()), method="median", modality="peptide", layer="raw")
    pairwise_shifts = _applied_shifts(pairwise)
    median_shifts = _applied_shifts(median)
    median_shifts = median_shifts - median_shifts.mean()

    assert np.abs(pairwise_shifts - expected).max() < 0.05
    assert abs(median_shifts[truncated_row] - expected[truncated_row]) > 0.2


def test_result_is_invariant_to_sample_order():
    values = _log2_values(noise_sd=0.2)
    permutation = np.array([3, 0, 5, 1, 4, 2])
    out = _normalised_layer(normalise(_mdata(values), method="pairwise_median", modality="peptide", layer="raw"))
    permuted_out = _normalised_layer(
        normalise(_mdata(values[permutation], [SAMPLES[i] for i in permutation]), method="pairwise_median",
                  modality="peptide", layer="raw")
    )
    np.testing.assert_allclose(permuted_out[np.argsort(permutation)], out, atol=1e-10)


def test_two_samples_split_the_pair_median_and_one_sample_is_unchanged():
    two = _log2_values()[:2]
    pair_median = np.nanmedian(two[0] - two[1])
    estimate = estimate_pairwise_median_shifts(two)
    np.testing.assert_allclose(estimate.shift_log2, [pair_median / 2, -pair_median / 2], atol=1e-10)

    one = _log2_values()[:1]
    np.testing.assert_array_equal(normalise_pairwise_median(one.T), one.T)


def test_all_nan_sample_is_excluded_dense_and_sparse():
    values = _log2_values()
    values[5] = np.nan
    dense_out = normalise(_mdata(values.copy()), method="pairwise_median", modality="peptide", layer="raw")
    sparse_layer = _sparse_with_absent_cells(values)
    sparse_out = normalise(_mdata(sparse_layer), method="pairwise_median", modality="peptide", layer="raw")
    without = normalise(_mdata(values[:5].copy(), SAMPLES[:5]), method="pairwise_median", modality="peptide", layer="raw")

    for out in (dense_out, sparse_out):
        layer = _normalised_layer(out)
        assert np.isnan(layer[5]).all()
        np.testing.assert_allclose(layer[:5], _normalised_layer(without), atol=1e-10, equal_nan=True)
        assert list(out["peptide"].uns["normalisation"]["summary"]["sample"]) == SAMPLES[:5]


def test_nan_positions_are_preserved():
    values = _log2_values()
    dense_out = normalise(_mdata(values.copy()), method="pairwise_median", modality="peptide", layer="raw")
    np.testing.assert_array_equal(np.isnan(_normalised_layer(dense_out)), np.isnan(values))
    sparse_layer = _sparse_with_absent_cells(values)
    sparse_out = normalise(_mdata(sparse_layer), method="pairwise_median", modality="peptide", layer="raw")
    out_layer = sparse_out["peptide"].layers["raw"]
    assert sp.issparse(out_layer)
    np.testing.assert_array_equal(out_layer.tocsr().indices, sparse_layer.indices)
    np.testing.assert_array_equal(out_layer.tocsr().indptr, sparse_layer.indptr)


@pytest.mark.parametrize("grouping", [{}, {"group_obs": "batch"}, {"group_var": "fraction"}, {"group_obs": "batch", "group_var": "fraction"}])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_sparse_matches_dense_and_stays_sparse(grouping, dtype):
    values = _log2_values(noise_sd=0.2).astype(dtype)
    sparse_out = normalise(_mdata(_sparse_with_absent_cells(values)), method="pairwise_median", modality="peptide",
                           layer="raw", **grouping)
    dense_out = normalise(_mdata(values.copy()), method="pairwise_median", modality="peptide", layer="raw", **grouping)
    tolerance = 1e-10 if dtype is np.float64 else 1e-4
    np.testing.assert_allclose(_normalised_layer(sparse_out), _normalised_layer(dense_out), atol=tolerance, equal_nan=True)
    assert sp.issparse(sparse_out["peptide"].layers["raw"])
    assert sparse_out["peptide"].layers["raw"].dtype == dtype


def test_grouping_changes_the_result():
    values = _log2_values(noise_sd=0.2)
    ungrouped = _normalised_layer(normalise(_mdata(values.copy()), method="pairwise_median", modality="peptide", layer="raw"))
    grouped = _normalised_layer(
        normalise(_mdata(values.copy()), method="pairwise_median", modality="peptide", layer="raw", group_obs="batch")
    )
    assert not np.allclose(ungrouped, grouped, equal_nan=True)


def _block_diagonal_values() -> np.ndarray:
    values = np.full((6, 12), np.nan)
    for sample in range(6):
        values[sample, 2 * sample : 2 * sample + 2] = 20.0
    return values


def test_unshared_sample_pairs_raise_naming_the_samples_and_pass_within_groups():
    values = _block_diagonal_values()
    values[:3, 0] = 20.0  # batch a shares v0, batch b shares v6
    values[3:, 6] = 20.0
    for layer in (values.copy(), _sparse_with_absent_cells(values)):
        with pytest.raises(ValueError, match="Samples share no observed feature") as info:
            normalise(_mdata(layer), method="pairwise_median", modality="peptide", layer="raw")
        assert "'s1' x 's4'" in str(info.value)
        assert "group_obs" in str(info.value)
        normalise(_mdata(layer), method="pairwise_median", modality="peptide", layer="raw", group_obs="batch")


def test_estimator_error_carries_block_positions_and_survives_pickling():
    with pytest.raises(UnsharedSamplePairsError) as info:
        estimate_pairwise_median_shifts(_block_diagonal_values())
    restored = pickle.loads(pickle.dumps(info.value))
    assert restored.sample_position_pairs == info.value.sample_position_pairs
    assert (0, 1) in restored.sample_position_pairs


def test_block_diagonal_sparse_raises_before_densifying(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("densified a block whose samples share no feature")

    monkeypatch.setattr(_normalisation, "_densify_shared_columns", fail)
    monkeypatch.setattr("msmu._preprocessing._normalise.dense_block", fail)
    with pytest.raises(ValueError, match="Samples share no observed feature"):
        normalise(_mdata(_sparse_with_absent_cells(_block_diagonal_values())), method="pairwise_median",
                  modality="peptide", layer="raw")


@pytest.mark.parametrize("method", ["median", "median_center", "total_sum", "quantile", "pairwise_median"])
def test_every_method_records_a_summary(method):
    values = _log2_values(noise_sd=0.2)
    out = normalise(_mdata(values.copy()), method=method, modality="peptide", layer="raw", group_var="fraction")
    record = out["peptide"].uns["normalisation"]
    summary = record["summary"]
    assert list(summary["sample"]) == SAMPLES * 2
    assert set(summary["var_group"]) == {"f1", "f2"}
    assert (summary["method"] == method).all() and (summary["layer"] == "raw").all()
    assert set(record["blocks"]) == {"raw|all|f1", "raw|all|f2"}
    if method in ("median", "median_center", "total_sum", "pairwise_median"):
        layer = _normalised_layer(out)
        for block_key, columns in (("raw|all|f1", slice(0, 200)), ("raw|all|f2", slice(200, 400))):
            applied = np.nanmedian(values[:, columns] - layer[:, columns], axis=1)
            np.testing.assert_allclose(record["blocks"][block_key]["shift_log2"], applied, atol=1e-10)


def test_pairwise_record_holds_the_pair_matrices_and_survives_h5mu(tmp_path):
    out = normalise(_mdata(_log2_values(noise_sd=0.2)), method="pairwise_median", modality="peptide", layer="raw")
    block = out["peptide"].uns["normalisation"]["blocks"]["raw|all|all"]
    np.testing.assert_allclose(block["pair_median_log2"].mean(axis=1).to_numpy(), block["shift_log2"], atol=1e-10)
    assert list(block["pair_median_log2"].index) == SAMPLES
    assert (np.diag(block["pair_shared_count"].to_numpy()) == block["n_observed_features"]).all()

    out.write_h5mu(tmp_path / "out.h5mu")
    from mudata import read_h5mu

    restored = read_h5mu(tmp_path / "out.h5mu")["peptide"].uns["normalisation"]
    pd.testing.assert_frame_equal(restored["summary"], out["peptide"].uns["normalisation"]["summary"])
    pd.testing.assert_frame_equal(restored["blocks"]["raw|all|all"]["pair_median_log2"], block["pair_median_log2"])


def test_records_of_other_layers_are_kept_and_same_layer_is_replaced():
    mdata = _mdata(_log2_values(noise_sd=0.2))
    mdata["peptide"].layers["other"] = _log2_values(seed=2, noise_sd=0.2)
    out = normalise(mdata, method="median", modality="peptide", layer="other")
    out = normalise(out, method="pairwise_median", modality="peptide", layer="raw")
    out = normalise(out, method="median_center", modality="peptide", layer="raw")
    record = out["peptide"].uns["normalisation"]
    assert set(record["blocks"]) == {"other|all|all", "raw|all|all"}
    assert record["blocks"]["raw|all|all"]["method"] == "median_center"
    assert set(record["summary"]["method"]) == {"median", "median_center"}


def test_weakly_shared_pair_is_warned_about(caplog):
    values = _log2_values(noise_sd=0.2)
    values[0, 20:] = np.nan  # s1 keeps 20 features: shares far fewer than the other pairs
    with caplog.at_level(logging.WARNING, logger="msmu"):
        normalise(_mdata(values), method="pairwise_median", modality="peptide", layer="raw")
    warnings = [record for record in caplog.records if "share only" in record.getMessage()]
    assert len(warnings) == 1 and "'s1'" in warnings[0].getMessage()

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="msmu"):
        normalise(_mdata(_log2_values(noise_sd=0.2)), method="pairwise_median", modality="peptide", layer="raw")
    assert not [record for record in caplog.records if "share only" in record.getMessage()]
