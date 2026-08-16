"""
Tests for the statistical core.

These check properties rather than golden numbers. A permutation test is
stochastic; asserting it returns 0.0413 would be asserting the seed, not the
mathematics. What must hold is that identical groups look identical, separated
groups look separated, the p-value can never claim more precision than the
resample count supports, and the FDR correction obeys its own definition.
"""

from __future__ import annotations

import math

import pytest

from hindsight import stats


class TestPermutationTest:
    def test_identical_groups_are_not_significant(self):
        a = [50.0] * 40
        b = [50.0] * 40
        assert stats.permutation_test(a, b, iters=1000) > 0.9

    def test_clearly_separated_groups_are_significant(self):
        a = [90.0 + i * 0.1 for i in range(40)]
        b = [10.0 + i * 0.1 for i in range(40)]
        assert stats.permutation_test(a, b, iters=2000) < 0.01

    def test_p_value_never_reports_zero(self):
        """
        Phipson & Smyth: with `iters` resamples the smallest supportable
        p-value is 1/(iters+1). Reporting 0 would claim certainty the
        resampling cannot provide.
        """
        a = [1000.0] * 30
        b = [0.0] * 30
        p = stats.permutation_test(a, b, iters=500)
        assert p == pytest.approx(1 / 501, rel=1e-9)
        assert p > 0

    def test_symmetric_in_arguments(self):
        a = [1.0, 5.0, 9.0, 3.0, 7.0] * 6
        b = [2.0, 4.0, 6.0, 8.0, 10.0] * 6
        p1 = stats.permutation_test(a, b, iters=2000, seed=7)
        p2 = stats.permutation_test(b, a, iters=2000, seed=7)
        assert p1 == pytest.approx(p2, abs=0.05)

    def test_empty_group_returns_unity(self):
        assert stats.permutation_test([], [1.0, 2.0]) == 1.0
        assert stats.permutation_test([1.0, 2.0], []) == 1.0

    def test_deterministic_for_a_fixed_seed(self):
        a, b = [3.0, 6.0, 9.0] * 8, [4.0, 5.0, 6.0] * 8
        assert stats.permutation_test(a, b, iters=800, seed=99) == \
               stats.permutation_test(a, b, iters=800, seed=99)


class TestBootstrapCI:
    def test_interval_brackets_the_observed_difference(self):
        a = [60.0 + (i % 5) for i in range(60)]
        b = [40.0 + (i % 5) for i in range(60)]
        lo, hi = stats.bootstrap_ci(a, b, iters=2000)
        observed = sum(a) / len(a) - sum(b) / len(b)
        assert lo < observed < hi

    def test_smaller_samples_give_wider_intervals(self):
        big_a = [50.0 + (i % 20) for i in range(400)]
        big_b = [45.0 + (i % 20) for i in range(400)]
        small_a, small_b = big_a[:20], big_b[:20]

        wide_lo, wide_hi = stats.bootstrap_ci(small_a, small_b, iters=2000)
        tight_lo, tight_hi = stats.bootstrap_ci(big_a, big_b, iters=2000)
        assert (wide_hi - wide_lo) > (tight_hi - tight_lo)

    def test_identical_groups_interval_contains_zero(self):
        a = [10.0, 20.0, 30.0] * 15
        lo, hi = stats.bootstrap_ci(a, list(a), iters=2000)
        assert lo <= 0.0 <= hi


class TestBenjaminiHochberg:
    def test_returns_one_q_per_input_in_order(self):
        ps = [0.001, 0.02, 0.3, 0.9]
        qs = stats.benjamini_hochberg(ps)
        assert len(qs) == len(ps)

    def test_q_values_are_never_below_their_p_values(self):
        ps = [0.001, 0.01, 0.04, 0.2, 0.5, 0.99]
        for p, q in zip(ps, stats.benjamini_hochberg(ps)):
            assert q >= p - 1e-12

    def test_monotone_in_p(self):
        ps = [0.001, 0.008, 0.02, 0.04, 0.2, 0.6]
        qs = stats.benjamini_hochberg(ps)
        assert qs == sorted(qs)

    def test_largest_p_maps_to_itself(self):
        ps = [0.01, 0.2, 0.8]
        assert stats.benjamini_hochberg(ps)[-1] == pytest.approx(0.8)

    def test_correction_suppresses_lone_marginal_result(self):
        """
        One p=0.04 among forty nulls is what multiple testing looks like.
        Uncorrected it is a 'finding'; corrected it must not be.
        """
        ps = [0.04] + [0.5 + i * 0.01 for i in range(39)]
        assert stats.benjamini_hochberg(ps)[0] > 0.05

    def test_strong_signal_survives_correction(self):
        ps = [0.00001] + [0.5 + i * 0.01 for i in range(39)]
        assert stats.benjamini_hochberg(ps)[0] < 0.05

    def test_empty_input(self):
        assert stats.benjamini_hochberg([]) == []

    def test_q_values_clipped_to_one(self):
        assert all(q <= 1.0 for q in stats.benjamini_hochberg([0.9, 0.95, 0.99]))


class TestInverseNormal:
    @pytest.mark.parametrize("p,expected", [
        (0.5, 0.0), (0.975, 1.959964), (0.8, 0.841621),
        (0.95, 1.644854), (0.025, -1.959964), (0.99, 2.326348),
    ])
    def test_known_quantiles(self, p, expected):
        assert stats._inverse_normal_cdf(p) == pytest.approx(expected, abs=1e-4)

    def test_extreme_tails_stay_finite(self):
        assert math.isfinite(stats._inverse_normal_cdf(1e-8))
        assert math.isfinite(stats._inverse_normal_cdf(1 - 1e-8))

    @pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5])
    def test_out_of_range_rejected(self, p):
        with pytest.raises(ValueError):
            stats._inverse_normal_cdf(p)


class TestPowerAnalysis:
    def test_bigger_effects_need_fewer_videos(self):
        assert stats.required_sample_size(28.9, 10.0) < \
               stats.required_sample_size(28.9, 5.0)

    def test_noisier_data_needs_more_videos(self):
        assert stats.required_sample_size(40.0, 5.0) > \
               stats.required_sample_size(20.0, 5.0)

    def test_matches_the_textbook_formula(self):
        """n = 2(z_{1-a/2} + z_power)^2 sigma^2 / delta^2, sigma=28.9, d=5."""
        n = stats.required_sample_size(28.9, 5.0, power=0.80, alpha=0.05)
        expected = 2 * ((1.959964 + 0.841621) ** 2) * (28.9 ** 2) / 25.0
        assert n == math.ceil(expected)

    def test_rejects_non_positive_effect(self):
        with pytest.raises(ValueError):
            stats.required_sample_size(28.9, 0.0)

    def test_zero_variance_needs_no_samples(self):
        assert stats.required_sample_size(0.0, 5.0) == 1

    def test_detectable_effect_inverts_sample_size(self):
        n = stats.required_sample_size(28.9, 8.0)
        assert stats.detectable_effect(n, 28.9) == pytest.approx(8.0, rel=0.02)

    def test_tiny_samples_resolve_nothing(self):
        assert stats.detectable_effect(1, 28.9) == float("inf")


class TestCompareAndFDR:
    def test_compare_reports_group_sizes_and_lift(self):
        a, b = [70.0] * 30, [50.0] * 30
        r = stats.compare(a, b, iters=500)
        assert (r.n_group, r.n_rest) == (30, 30)
        assert r.lift == pytest.approx(20.0)
        assert r.direction == "better"

    def test_significance_uses_corrected_q_not_raw_p(self):
        """A result must not count as significant on its raw p-value alone."""
        r = stats.TestResult(
            n_group=30, n_rest=30, mean_group=60, mean_rest=50,
            lift=10, ci_low=2, ci_high=18, p_value=0.01, q_value=0.4,
        )
        assert r.p_value < stats.config.ALPHA
        assert not r.significant

    def test_apply_fdr_attaches_q_values(self):
        results = {
            "a": stats.compare([90.0] * 40, [10.0] * 40, iters=500),
            "b": stats.compare([50.0] * 40, [50.1] * 40, iters=500),
        }
        corrected = stats.apply_fdr(results)
        assert set(corrected) == {"a", "b"}
        assert corrected["a"].q_value <= corrected["b"].q_value

    def test_apply_fdr_on_empty(self):
        assert stats.apply_fdr({}) == {}
