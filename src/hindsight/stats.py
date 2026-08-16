"""
Statistics, implemented here rather than imported.

Everything in this module is non-parametric. That is a deliberate choice, not
an aesthetic one: YouTube view counts are violently right-skewed (on the
channel this tool was built against, the mean is 337 and the median is 140),
so any method that assumes normally distributed outcomes will read that skew
as signal. Working on cohort *percentiles* and testing them by permutation
sidesteps the distribution entirely -- the null hypothesis becomes "the labels
are exchangeable", which is exactly the question a creator is asking.

Contents:
    permutation_test    two-sided difference-of-means, no distribution assumed
    bootstrap_ci        percentile-bootstrap interval on the same difference
    benjamini_hochberg  false-discovery-rate control across many features
    required_sample_size  power analysis for `hindsight design`

The multiple-comparison step matters more than it looks. Hindsight tests on
the order of forty feature buckets per run; at alpha=0.05 you would expect two
false "findings" per analysis from noise alone. Reporting those to a creator
who then rewrites their whole content strategy around them is the single most
harmful thing this tool could do, so the FDR correction is applied by default
and the uncorrected p-value is kept only for display.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import config


@dataclass(frozen=True)
class TestResult:
    """Outcome of comparing one group against the rest of the catalog."""

    n_group: int
    n_rest: int
    mean_group: float
    mean_rest: float
    lift: float
    ci_low: float
    ci_high: float
    p_value: float
    q_value: float = 1.0  # FDR-corrected; filled in by benjamini_hochberg

    @property
    def significant(self) -> bool:
        """Significant after false-discovery-rate correction, not before."""
        return self.q_value <= config.ALPHA

    @property
    def direction(self) -> str:
        return "better" if self.lift > 0 else "worse"


def permutation_test(
    group: Sequence[float],
    rest: Sequence[float],
    iters: int = config.PERMUTATION_ITERS,
    seed: int = config.RANDOM_SEED,
) -> float:
    """
    Two-sided permutation p-value for the difference in means.

    Under the null the two labels are interchangeable, so we pool the values,
    repeatedly re-split them at random into groups of the original sizes, and
    ask how often a random split produces a gap at least as large as the one
    actually observed.

    The `+1` in the numerator and denominator is Phipson & Smyth's correction:
    without it a test that never sees a more extreme permutation reports p=0,
    which claims more certainty than `iters` resamples can support. The
    smallest p-value this can return is 1/(iters+1).
    """
    a = np.asarray(group, dtype=float)
    b = np.asarray(rest, dtype=float)
    if a.size == 0 or b.size == 0:
        return 1.0

    observed = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    n_a, n_total = a.size, pool.size
    total = pool.sum()

    rng = np.random.default_rng(seed)

    # argpartition is O(n) where a full argsort would be O(n log n); we only
    # need *which* n_a elements land in the first group, not their order.
    noise = rng.random((iters, n_total))
    picks = np.argpartition(noise, n_a - 1, axis=1)[:, :n_a]
    sums_a = pool[picks].sum(axis=1)

    diffs = np.abs(sums_a / n_a - (total - sums_a) / (n_total - n_a))
    hits = int((diffs >= observed - 1e-12).sum())
    return (hits + 1) / (iters + 1)


def bootstrap_ci(
    group: Sequence[float],
    rest: Sequence[float],
    confidence: float = 0.95,
    iters: int = config.BOOTSTRAP_ITERS,
    seed: int = config.RANDOM_SEED,
) -> tuple[float, float]:
    """
    Percentile-bootstrap confidence interval on (mean(group) - mean(rest)).

    Both groups are resampled with replacement at their own sizes, which
    propagates the uncertainty of a small group into a correspondingly wide
    interval. That width is the honest answer to "should I act on this?" and
    is reported alongside every finding.
    """
    a = np.asarray(group, dtype=float)
    b = np.asarray(rest, dtype=float)
    if a.size == 0 or b.size == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    means_a = a[rng.integers(0, a.size, size=(iters, a.size))].mean(axis=1)
    means_b = b[rng.integers(0, b.size, size=(iters, b.size))].mean(axis=1)
    diffs = means_a - means_b

    tail = (1.0 - confidence) / 2.0 * 100.0
    return (
        float(np.percentile(diffs, tail)),
        float(np.percentile(diffs, 100.0 - tail)),
    )


def compare(
    group: Sequence[float],
    rest: Sequence[float],
    iters: int = config.PERMUTATION_ITERS,
    seed: int = config.RANDOM_SEED,
) -> TestResult:
    """Run the permutation test and bootstrap interval together."""
    a = np.asarray(group, dtype=float)
    b = np.asarray(rest, dtype=float)

    mean_a = float(a.mean()) if a.size else 0.0
    mean_b = float(b.mean()) if b.size else 0.0
    lo, hi = bootstrap_ci(a, b, iters=min(iters, config.BOOTSTRAP_ITERS), seed=seed)

    return TestResult(
        n_group=int(a.size),
        n_rest=int(b.size),
        mean_group=mean_a,
        mean_rest=mean_b,
        lift=mean_a - mean_b,
        ci_low=lo,
        ci_high=hi,
        p_value=permutation_test(a, b, iters=iters, seed=seed),
    )


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """
    Benjamini-Hochberg step-up FDR correction.

    Returns q-values in the same order as the input. A q-value of 0.05 means
    "if I act on every finding at or below this threshold, about 5% of the
    ones I act on will be flukes" -- which is the question that matters when
    you are choosing which lever to pull, and is a far more useful guarantee
    than Bonferroni's near-total suppression of real effects.

    The enforced monotonicity (a cumulative minimum walking up from the
    largest p-value) is part of the procedure, not a smoothing hack.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return []

    order = np.argsort(p)
    ranked = p[order]
    scaled = ranked * n / np.arange(1, n + 1)

    # Step-up: q_i is the smallest scaled value at or above rank i.
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)

    q = np.empty(n, dtype=float)
    q[order] = monotone
    return q.tolist()


def apply_fdr(results: dict[str, TestResult]) -> dict[str, TestResult]:
    """Attach FDR-corrected q-values to a keyed collection of test results."""
    if not results:
        return {}
    keys = list(results)
    qs = benjamini_hochberg([results[k].p_value for k in keys])
    return {
        k: TestResult(**{**results[k].__dict__, "q_value": q})
        for k, q in zip(keys, qs)
    }


# --------------------------------------------------------------------------
# Power analysis -- used by `hindsight design` to size the next experiment
# --------------------------------------------------------------------------


def _inverse_normal_cdf(p: float) -> float:
    """
    Inverse standard-normal CDF via Acklam's rational approximation.

    Accurate to about 1.15e-9 across the open interval, which is far beyond
    what a sample-size estimate needs, and it keeps scipy out of the
    dependency list for the sake of two z-scores.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    p_low, p_high = 0.02425, 1 - 0.02425

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


def required_sample_size(
    std_dev: float,
    min_effect: float = config.MIN_DETECTABLE_EFFECT,
    power: float = config.TARGET_POWER,
    alpha: float = config.ALPHA,
) -> int:
    """
    Videos needed *per arm* to detect `min_effect` with the given power.

    The standard two-sample formula:

        n = 2 * (z(1-a/2) + z(power))^2 * sigma^2 / delta^2

    Because the outcome is a cohort percentile, sigma is bounded (a uniform
    distribution over 0-100 has sigma ~ 28.9), so this stays well-behaved even
    on a channel whose raw view counts span three orders of magnitude.

    The number this returns is usually the most sobering output Hindsight
    produces: detecting a 5-percentile effect takes a few hundred videos per
    arm. For a channel posting ten a day that is a fortnight; the point of
    saying it out loud is that it stops people concluding anything from the
    six-video "test" they were about to run.
    """
    if min_effect <= 0:
        raise ValueError("min_effect must be positive")
    if std_dev <= 0:
        return 1

    z_alpha = _inverse_normal_cdf(1 - alpha / 2)
    z_power = _inverse_normal_cdf(power)
    n = 2 * ((z_alpha + z_power) ** 2) * (std_dev ** 2) / (min_effect ** 2)
    return max(2, math.ceil(n))


def detectable_effect(
    n_per_arm: int,
    std_dev: float,
    power: float = config.TARGET_POWER,
    alpha: float = config.ALPHA,
) -> float:
    """
    Inverse of `required_sample_size`: the smallest effect `n` videos can see.

    Used to tell someone who only has 40 videos per arm what their test is
    actually capable of resolving, instead of letting them run it and read
    tea leaves.
    """
    if n_per_arm < 2 or std_dev <= 0:
        return float("inf")
    z_alpha = _inverse_normal_cdf(1 - alpha / 2)
    z_power = _inverse_normal_cdf(power)
    return math.sqrt(2 * ((z_alpha + z_power) ** 2) * (std_dev ** 2) / n_per_arm)
