import logging

import numpy as np
import pytest

from msmu._statistics._permutation import PermutationTest, min_achievable_q


def _make_arrays(n_ctrl: int, n_expr: int, n_features: int = 8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return rng.normal(size=(n_ctrl, n_features)), rng.normal(size=(n_expr, n_features))


class TestMinAchievableQ:
    def test_balanced_3v3_floor_matches_observed_value(self):
        # C(6,3)=20 splits, and both the observed labelling and its complement are among them,
        # so fp >= 2*tp and q bottoms out at PI0_LOWER_BOUND * 3/((20+1)*2).
        assert np.isclose(min_achievable_q(3, 3, n_permutations=20, fdr="empirical"), 0.95 * 3 / 42)

    def test_balanced_3v3_cannot_reach_conventional_cutoff(self):
        assert min_achievable_q(3, 3, n_permutations=20, fdr="empirical") > 0.05

    def test_bh_floor_is_higher_than_empirical_at_3v3(self):
        # BH keeps self_inclusion/B = 2/20; the empirical estimator's /(tp+1) divisor puts its
        # floor lower. Neither clears 0.05.
        bh = min_achievable_q(3, 3, n_permutations=20, fdr="bh")
        empirical = min_achievable_q(3, 3, n_permutations=20, fdr="empirical")
        assert bh > empirical > 0.05

    def test_unbalanced_design_escapes_the_complement_penalty(self):
        # For 3v4 the complement of a 3-subset is a 4-subset, which is not itself a valid ctrl
        # split, so only the observed labelling is self-included and fp >= tp.
        assert min_achievable_q(3, 4, n_permutations=35, fdr="empirical") < 0.05

    def test_floor_falls_as_permutations_grow(self):
        floors = [min_achievable_q(n, n, n_permutations=b, fdr="empirical") for n, b in [(3, 20), (4, 70), (5, 252)]]
        assert floors == sorted(floors, reverse=True)
        assert floors[0] > 0.05 > floors[1]


class TestSignificanceUnreachableWarning:
    def test_warns_for_3v3(self, caplog):
        ctrl, expr = _make_arrays(3, 3)
        with caplog.at_level(logging.WARNING):
            PermutationTest(ctrl_arr=ctrl, expr_arr=expr, n_resamples=20, _force_resample=False, fdr="empirical")
        assert "cannot produce q < 0.05" in caplog.text
        assert "0.068" in caplog.text
        assert "limma" in caplog.text

    def test_silent_for_4v4(self, caplog):
        ctrl, expr = _make_arrays(4, 4)
        with caplog.at_level(logging.WARNING):
            PermutationTest(ctrl_arr=ctrl, expr_arr=expr, n_resamples=70, _force_resample=False, fdr="empirical")
        assert "cannot produce" not in caplog.text

    def test_silent_when_fdr_disabled(self, caplog):
        ctrl, expr = _make_arrays(3, 3)
        with caplog.at_level(logging.WARNING):
            PermutationTest(ctrl_arr=ctrl, expr_arr=expr, n_resamples=20, _force_resample=False, fdr=False)
        assert "cannot produce" not in caplog.text

    @pytest.mark.parametrize("fdr", ["empirical", "bh"])
    def test_warns_for_both_correction_methods(self, caplog, fdr):
        ctrl, expr = _make_arrays(3, 3)
        with caplog.at_level(logging.WARNING):
            PermutationTest(ctrl_arr=ctrl, expr_arr=expr, n_resamples=20, _force_resample=False, fdr=fdr)
        assert f"fdr='{fdr}'" in caplog.text


class TestResamplesIgnoredWarning:
    def test_warns_when_n_resamples_exceeds_available_splits(self, caplog):
        ctrl, expr = _make_arrays(3, 3)
        with caplog.at_level(logging.WARNING):
            perm = PermutationTest(
                ctrl_arr=ctrl, expr_arr=expr, n_resamples=1000, _force_resample=False, fdr="empirical"
            )
        assert "n_resamples=1000 exceeds the 20 distinct" in caplog.text
        assert perm.permutation_method == "exact"
        assert perm.n_permutations_used == 20

    def test_silent_when_n_resamples_fits(self, caplog):
        ctrl, expr = _make_arrays(6, 6)
        with caplog.at_level(logging.WARNING):
            perm = PermutationTest(
                ctrl_arr=ctrl, expr_arr=expr, n_resamples=100, _force_resample=False, fdr="empirical"
            )
        assert "exceeds" not in caplog.text
        assert perm.permutation_method == "randomised"
        assert perm.n_permutations_used == 100

    def test_forced_resampling_reports_the_requested_count(self, caplog):
        ctrl, expr = _make_arrays(3, 3)
        with caplog.at_level(logging.WARNING):
            perm = PermutationTest(
                ctrl_arr=ctrl, expr_arr=expr, n_resamples=1000, _force_resample=True, fdr="empirical"
            )
        assert perm.permutation_method == "randomised"
        assert perm.n_permutations_used == 1000
        assert "exceeds" not in caplog.text
