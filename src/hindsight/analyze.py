"""
Analysis: from a scored catalog to a ranked list of things worth doing.

The pipeline is:

    1. Score every video against its publication-time cohort      (cohort.py)
    2. Bucket every eligible video on every controllable feature  (features.py)
    3. For each bucket with enough videos, test it against the rest of the
       catalog by permutation                                     (stats.py)
    4. Correct every p-value in the run for false discovery       (stats.py)
    5. Classify each feature and rank what survived

Step 5 is where most of the judgement lives, because a feature can fail to
produce a finding in three very different ways and a creator needs to tell
them apart:

    TESTED         the channel varied this lever enough to measure it, and the
                   result -- signal or no signal -- is trustworthy.

    UNTESTED       every video in the catalog made the same choice. There is
                   no finding because there is no variation. This is not a
                   dead end; it is the most valuable output Hindsight
                   produces, because an untested lever is the only kind
                   guaranteed to still have headroom.

    UNDERPOWERED   the lever varied, but the buckets are too small to conclude
                   anything. Reporting "no significant difference" here would
                   be actively misleading -- absence of evidence, presented as
                   evidence of absence, is how creators end up believing a
                   lever does not work when they simply never tested it
                   properly.

A tool that only reported category one would look at a channel with 949
identically-tagged videos and say "tags don't matter". Which would be exactly
backwards.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import config, cohort, features as feat, stats
from .cohort import ScoredVideo
from .features import Feature
from .stats import TestResult

# Feature classification outcomes.
TESTED = "tested"
UNTESTED = "untested"
UNDERPOWERED = "underpowered"


@dataclass
class BucketResult:
    """One value of one feature, and how the videos that used it performed."""

    bucket: str
    n: int
    mean_percentile: float
    median_views: float
    mean_like_rate_percentile: float
    test: TestResult | None = None
    examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def testable(self) -> bool:
        return self.test is not None

    @property
    def significant(self) -> bool:
        return self.test is not None and self.test.significant


@dataclass
class FeatureResult:
    """A controllable lever, its buckets, and what the catalog says about it."""

    feature: Feature
    buckets: list[BucketResult]
    status: str
    note: str = ""

    @property
    def tested_buckets(self) -> list[BucketResult]:
        return [b for b in self.buckets if b.testable]

    @property
    def significant_buckets(self) -> list[BucketResult]:
        return [b for b in self.buckets if b.significant]

    @property
    def best(self) -> BucketResult | None:
        tested = self.tested_buckets
        return max(tested, key=lambda b: b.mean_percentile) if tested else None

    @property
    def worst(self) -> BucketResult | None:
        tested = self.tested_buckets
        return min(tested, key=lambda b: b.mean_percentile) if tested else None

    @property
    def spread(self) -> float:
        """Percentile gap between the best and worst tested bucket."""
        best, worst = self.best, self.worst
        if best is None or worst is None:
            return 0.0
        return best.mean_percentile - worst.mean_percentile

    @property
    def has_finding(self) -> bool:
        return self.status == TESTED and bool(self.significant_buckets)


@dataclass
class Finding:
    """A single actionable, statistically surviving result."""

    feature_key: str
    feature_label: str
    lever: str
    group: str
    bucket: str
    n: int
    lift: float
    ci_low: float
    ci_high: float
    p_value: float
    q_value: float
    mean_percentile: float
    median_views: float
    like_rate_percentile: float
    comparison_bucket: str | None = None
    comparison_percentile: float | None = None

    # Era-split robustness. See `_era_robustness`.
    early_lift: float | None = None
    late_lift: float | None = None
    replicates: bool = False
    era_note: str = ""

    @property
    def direction(self) -> str:
        return "outperforms" if self.lift > 0 else "underperforms"

    @property
    def abs_lift(self) -> float:
        return abs(self.lift)


@dataclass
class AnalysisResult:
    channel: dict[str, Any]
    scored: list[ScoredVideo]
    summary: dict[str, Any]
    feature_results: list[FeatureResult]
    findings: list[Finding]
    untested_levers: list[FeatureResult]
    underpowered_levers: list[FeatureResult]
    params: dict[str, Any]

    @property
    def eligible_count(self) -> int:
        return self.summary["eligible"]


def analyze(
    videos: Sequence[dict[str, Any]],
    channel: dict[str, Any],
    half_width: int = config.COHORT_HALF_WIDTH,
    min_age_days: int = config.MIN_AGE_DAYS,
    min_bucket_n: int = config.MIN_BUCKET_N,
    timezone_offset_h: float = 0.0,
    iters: int = config.PERMUTATION_ITERS,
    seed: int = config.RANDOM_SEED,
) -> AnalysisResult:
    """Run the full analysis over a channel's catalog."""

    # 1-2. Score on both metrics. Views measures distribution; like-rate
    # measures resonance among the people actually reached.
    scored = cohort.score_catalog(
        videos, cohort.views_metric, half_width, min_age_days
    )
    like_scored = cohort.score_catalog(
        videos, cohort.like_rate_metric, half_width, min_age_days
    )
    like_pct = {s.video_id: s.percentile for s in like_scored}

    summary = cohort.summarize(scored)
    eligible = cohort.eligible_only(scored)

    feature_list = feat.build_features(
        [s.video for s in eligible], timezone_offset_h=timezone_offset_h
    )

    # 3. Bucket, then test every bucket that clears the size floor. All tests
    # are collected before any is judged, so the FDR correction sees the whole
    # family of comparisons made in this run.
    pending: dict[tuple[str, str], TestResult] = {}
    feature_buckets: dict[str, list[BucketResult]] = {}

    for feature in feature_list:
        groups: dict[str, list[ScoredVideo]] = defaultdict(list)
        for sv in eligible:
            bucket = feature.extract(sv.video)
            if bucket is not None:
                groups[bucket].append(sv)

        buckets: list[BucketResult] = []
        for name, members in groups.items():
            pcts = [m.percentile for m in members]
            views = sorted(m.metric_value for m in members)
            mid = len(views) // 2
            median_views = (
                views[mid] if len(views) % 2 else (views[mid - 1] + views[mid]) / 2
            ) if views else 0.0

            like_values = [like_pct.get(m.video_id, 50.0) for m in members]

            result = BucketResult(
                bucket=name,
                n=len(members),
                mean_percentile=sum(pcts) / len(pcts),
                median_views=median_views,
                mean_like_rate_percentile=sum(like_values) / len(like_values),
                examples=[
                    {
                        "video_id": m.video_id,
                        "title": m.title,
                        "views": int(m.metric_value),
                        "percentile": round(m.percentile, 1),
                    }
                    for m in sorted(members, key=lambda m: -m.percentile)[:3]
                ],
            )

            # One-vs-rest: this bucket against every other eligible video that
            # the feature applies to. Comparing against the rest of the
            # *catalog* rather than against one chosen rival avoids picking a
            # flattering baseline.
            rest = [
                m.percentile
                for other, ms in groups.items() if other != name
                for m in ms
            ]
            if len(members) >= min_bucket_n and len(rest) >= min_bucket_n:
                pending[(feature.key, name)] = stats.compare(
                    pcts, rest, iters=iters, seed=seed
                )

            buckets.append(result)

        buckets.sort(key=lambda b: -b.mean_percentile)
        feature_buckets[feature.key] = buckets

    # 4. Family-wide false-discovery correction.
    corrected = stats.apply_fdr(pending)
    for (fkey, bname), result in corrected.items():
        for b in feature_buckets[fkey]:
            if b.bucket == bname:
                b.test = result
                break

    # 5. Classify each feature and harvest findings.
    feature_results: list[FeatureResult] = []
    findings: list[Finding] = []

    for feature in feature_list:
        buckets = feature_buckets[feature.key]
        status, note = _classify(buckets, min_bucket_n, len(eligible))
        fr = FeatureResult(feature=feature, buckets=buckets, status=status, note=note)
        feature_results.append(fr)

        if status != TESTED:
            continue

        worst = fr.worst
        for b in fr.significant_buckets:
            findings.append(Finding(
                feature_key=feature.key,
                feature_label=feature.label,
                lever=feature.lever,
                group=feature.group,
                bucket=b.bucket,
                n=b.n,
                lift=b.test.lift,
                ci_low=b.test.ci_low,
                ci_high=b.test.ci_high,
                p_value=b.test.p_value,
                q_value=b.test.q_value,
                mean_percentile=b.mean_percentile,
                median_views=b.median_views,
                like_rate_percentile=b.mean_like_rate_percentile,
                comparison_bucket=worst.bucket if worst and worst is not b else None,
                comparison_percentile=(
                    worst.mean_percentile if worst and worst is not b else None
                ),
            ))

    # Every surviving finding is re-tested independently on the first and
    # second half of the catalog. See `_era_robustness` for why this matters
    # more than the p-value does.
    by_key = {f.key: f for f in feature_list}
    for finding in findings:
        _era_robustness(finding, by_key[finding.feature_key], eligible, min_bucket_n, seed)

    findings.sort(key=lambda f: (-f.abs_lift, f.q_value))

    return AnalysisResult(
        channel=channel,
        scored=scored,
        summary=summary,
        feature_results=feature_results,
        findings=findings,
        untested_levers=[f for f in feature_results if f.status == UNTESTED],
        underpowered_levers=[f for f in feature_results if f.status == UNDERPOWERED],
        params={
            "cohort_half_width": half_width,
            "min_age_days": min_age_days,
            "min_bucket_n": min_bucket_n,
            "permutation_iters": iters,
            "alpha": config.ALPHA,
            "seed": seed,
            "timezone_offset_h": timezone_offset_h,
        },
    )


def _era_robustness(
    finding: Finding,
    feature: Feature,
    eligible: Sequence[ScoredVideo],
    min_bucket_n: int,
    seed: int,
) -> None:
    """
    Re-test a finding separately on the older and newer half of the catalog.

    A p-value answers "could this gap have come from random label shuffling?".
    It cannot answer the question a sceptical creator should actually ask:
    "did this show up because of a habit I had during one period?"

    Consider the finding that 01:00 uploads underperform. If the channel
    posted at 01:00 only during a three-month stretch when it was also small,
    badly targeted, or between formats, then "01:00" is standing in for "that
    era" and rescheduling would achieve nothing. Cohort scoring already
    suppresses most of this -- a video is only ever compared to its immediate
    neighbours -- but it cannot suppress a habit that persisted across whole
    cohorts.

    Splitting the catalog in half and requiring the effect to appear in both
    halves independently is a direct test of that. An effect that replicates
    across two disjoint samples separated by months is a property of the
    choice; one that appears in a single half is a property of the period.

    Findings that do not replicate are kept and reported, not silently
    dropped -- but they are labelled, because the correct response to them is
    to run the experiment rather than to act.
    """
    half = len(eligible) // 2
    halves = (("early", eligible[:half]), ("late", eligible[half:]))
    lifts: dict[str, float | None] = {"early": None, "late": None}
    floor = max(8, min_bucket_n // 2)

    for name, subset in halves:
        group, rest = [], []
        for sv in subset:
            bucket = feature.extract(sv.video)
            if bucket is None:
                continue
            (group if bucket == finding.bucket else rest).append(sv.percentile)

        if len(group) >= floor and len(rest) >= floor:
            lifts[name] = sum(group) / len(group) - sum(rest) / len(rest)

    finding.early_lift = lifts["early"]
    finding.late_lift = lifts["late"]

    early, late = lifts["early"], lifts["late"]
    if early is None or late is None:
        finding.replicates = False
        finding.era_note = (
            "Could not be re-tested on both halves of the catalog -- this "
            "choice was not used often enough in one of the two periods."
        )
    elif (early > 0) == (late > 0) == (finding.lift > 0):
        finding.replicates = True
        finding.era_note = (
            f"Replicates: the effect points the same way in both halves of the "
            f"catalog ({early:+.1f} early, {late:+.1f} late), months apart. "
            f"That is consistent with the choice causing the difference rather "
            f"than a habit from one period."
        )
    else:
        finding.replicates = False
        finding.era_note = (
            f"Does not replicate: the effect is {early:+.1f} in the older half "
            f"and {late:+.1f} in the newer half. It may be an artifact of when "
            f"this choice was being made rather than of the choice itself. "
            f"Treat as a hypothesis to test, not a conclusion."
        )


def _classify(
    buckets: list[BucketResult], min_bucket_n: int, total: int
) -> tuple[str, str]:
    """Decide whether a lever was tested, never varied, or varied too thinly."""
    if len(buckets) <= 1:
        only = buckets[0].bucket if buckets else "nothing"
        return UNTESTED, (
            f"All {total} analysed videos used the same value ({only}). "
            f"This lever has never been tested on this channel."
        )

    testable = [b for b in buckets if b.testable]
    if len(testable) < 2:
        # The blocker is almost always the *minority* bucket: a 837-vs-8 split
        # is a lever that was nominally varied but effectively never tested,
        # and saying "the largest bucket has 837" would read as if the split
        # were fine. Report the small side, which is the side that is short.
        smallest = min(buckets, key=lambda b: b.n)
        majority = max(buckets, key=lambda b: b.n)
        share = majority.n / total * 100 if total else 0.0
        return UNDERPOWERED, (
            f"{majority.n} of {total} videos ({share:.0f}%) used "
            f"'{majority.bucket}', leaving only {smallest.n} in the smallest "
            f"alternative. A comparison needs at least {min_bucket_n} videos on "
            f"both sides, so this lever cannot be measured yet -- which is not "
            f"evidence that it does not matter."
        )

    return TESTED, ""


def top_and_bottom(
    result: AnalysisResult, n: int = 5
) -> tuple[list[ScoredVideo], list[ScoredVideo]]:
    """The best and worst performing eligible videos, for the report."""
    eligible = cohort.eligible_only(result.scored)
    ranked = sorted(eligible, key=lambda s: -s.percentile)
    return ranked[:n], ranked[-n:][::-1]
