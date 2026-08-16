"""
Experiment design and readout -- the half of the loop that makes the other
half worth running.

An analysis of past uploads can only ever produce correlations. The catalog
was not randomised: the creator chose when to post and what to write, and
those choices are tangled with each other and with everything that was
happening to the channel at the time. Cohort scoring and the era-split check
remove a great deal of that, but they cannot manufacture an experiment that
was never run.

So Hindsight designs one.

`design` reads the analysis, picks the lever with the most unrealised value,
and emits a concrete plan: which values to test, how many videos each arm
needs to resolve the effect, which variables to freeze so they cannot
confound the result, and how to interleave the arms so that time itself does
not become the hidden variable. `verdict` reads the channel back afterwards
and calls the result.

WHY THE SAMPLE SIZES LOOK BIG

The number of videos per arm comes from a power calculation, not from taste.
Detecting a five-percentile shift at 80% power needs a few hundred videos per
arm, and there is no way around that -- the variance in a cohort percentile is
simply large. For a channel posting ten videos a day this is a fortnight, and
saying so is the entire point. The alternative, which is what everyone
actually does, is to change something, watch six uploads, and conclude
whatever the first two suggested.

WHY ARMS ARE INTERLEAVED

Running arm A for a week and then arm B for a week does not test A against B;
it tests one week against another. Anything that moved in between -- a
holiday, an algorithm change, a burst of subscribers -- lands entirely on one
arm. Alternating upload by upload puts both arms under identical conditions,
which is the whole reason to bother running an experiment instead of just
reading the catalog.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import numpy as np

from . import config, db, stats
from .analyze import TESTED, UNDERPOWERED, UNTESTED, AnalysisResult, FeatureResult
from .cohort import parse_published


@dataclass
class Arm:
    """One condition in an experiment."""

    name: str
    value: str
    videos: int
    note: str = ""


@dataclass
class ExperimentPlan:
    """A complete, machine-consumable plan for the next test."""

    experiment_id: str
    channel_id: str
    channel_slug: str
    feature: str
    feature_label: str
    lever: str
    hypothesis: str
    rationale: str
    priority: str
    arms: list[Arm]
    videos_per_arm: int
    total_videos: int
    hold_constant: list[dict[str, str]]
    assignment: str
    observed_std_dev: float
    min_detectable_effect: float
    target_power: float
    alpha: float
    uploads_per_day: float
    estimated_days: float
    earliest_readout_utc: str
    created_utc: str
    readout_command: str
    # What this channel's cadence can actually resolve over various budgets.
    feasibility: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["arms"] = [asdict(a) for a in self.arms]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# --------------------------------------------------------------------------
# Choosing what to test next
# --------------------------------------------------------------------------

# Ranking rationale:
#
#   UNTESTED levers come first. A lever with no variation has produced no
#   information at all, so its entire effect -- whatever it is -- is still
#   unclaimed. It is also the only category where you know for certain you
#   have not already captured the value.
#
#   UNDERPOWERED levers come second. They varied by accident rather than by
#   design, which usually means a near-total split (99% one way). A deliberate
#   balanced test converts them into an answer cheaply.
#
#   PROMISING levers -- those that missed the corrected significance threshold
#   but not by much -- come third. There may be something there; an experiment
#   is how you find out rather than guessing from the same data that raised
#   the suspicion.
#
#   CONFIRMED levers come last, and only to re-test. If it already replicated
#   across eras, spend the uploads elsewhere.

PRIORITY_UNTESTED = "untested-lever"
PRIORITY_UNDERPOWERED = "underpowered-lever"
PRIORITY_PROMISING = "promising-but-unconfirmed"
PRIORITY_CONFIRM = "confirm-existing-finding"

_PRIORITY_ORDER = {
    PRIORITY_UNTESTED: 0,
    PRIORITY_UNDERPOWERED: 1,
    PRIORITY_PROMISING: 2,
    PRIORITY_CONFIRM: 3,
}

# A lever whose corrected q-value sits between alpha and this threshold is
# "promising": not established, but not noise-shaped either.
PROMISING_Q = 0.35


def rank_candidates(result: AnalysisResult) -> list[tuple[str, FeatureResult, str]]:
    """
    Rank every lever by how much value an experiment on it would create.

    Returns (priority, feature_result, reason) sorted best-first.
    """
    candidates: list[tuple[str, FeatureResult, str]] = []

    for fr in result.feature_results:
        if fr.status == UNTESTED:
            candidates.append((
                PRIORITY_UNTESTED, fr,
                f"Every one of the {result.eligible_count} analysed videos made "
                f"the same choice here, so this lever has produced no evidence "
                f"in the entire history of the channel. Whatever it is worth is "
                f"still on the table.",
            ))
        elif fr.status == UNDERPOWERED:
            candidates.append((
                PRIORITY_UNDERPOWERED, fr,
                f"This lever varied, but too lopsidedly to measure. {fr.note}",
            ))
        elif fr.status == TESTED:
            best_q = min(
                (b.test.q_value for b in fr.tested_buckets if b.test), default=1.0
            )
            if fr.has_finding:
                candidates.append((
                    PRIORITY_CONFIRM, fr,
                    f"Already produced a surviving finding (best q={best_q:.3f}). "
                    f"Worth a controlled re-test only once the untested levers "
                    f"are exhausted.",
                ))
            elif best_q <= PROMISING_Q and fr.spread >= config.MIN_DETECTABLE_EFFECT:
                candidates.append((
                    PRIORITY_PROMISING, fr,
                    f"Shows a {fr.spread:.1f}-point spread between its best and "
                    f"worst value and a best q of {best_q:.3f} -- suggestive, but "
                    f"it did not survive correction. A balanced test would settle it.",
                ))

    # Within a priority tier, prefer levers that subsume others: one test of
    # "which tags" also answers "how many tags", so it buys more information
    # for the same number of uploads. Spread breaks any remaining ties.
    candidates.sort(
        key=lambda c: (
            _PRIORITY_ORDER[c[0]],
            -len(c[1].feature.supersedes),
            -c[1].spread,
        )
    )

    # Drop candidates that a better-ranked candidate mechanically subsumes.
    # Without this, a channel that has never varied its tags gets told to run
    # a "tag count" experiment and a "tag set" experiment as if they were two
    # independent questions, when running either one answers both.
    superseded: set[str] = set()
    kept: list[tuple[str, FeatureResult, str]] = []
    for priority, fr, reason in candidates:
        if fr.feature.key in superseded:
            continue
        superseded.update(fr.feature.supersedes)
        kept.append((priority, fr, reason))

    return kept


# --------------------------------------------------------------------------
# Proposing the arms
# --------------------------------------------------------------------------


def _propose_arms(
    fr: FeatureResult, result: AnalysisResult, videos_per_arm: int
) -> tuple[list[Arm], list[str]]:
    """
    Propose the conditions to compare for a given lever.

    For a lever that already varies, the arms are its best and worst observed
    values -- the comparison the catalog was already gesturing at. For a lever
    that has never varied there is nothing to read off the data, so Hindsight
    constructs a credible alternative, and for tags it does so from the
    channel's own title vocabulary rather than from a generic list.
    """
    warnings: list[str] = []
    key = fr.feature.key

    if fr.status == UNTESTED:
        current = fr.buckets[0].bucket if fr.buckets else "(current)"

        if key == "tag_set":
            extra = _mine_tag_candidates(result, existing=current, count=10)
            return [
                Arm("control", current, videos_per_arm,
                    "The tag set every video has used so far."),
                Arm("expanded", f"{current}, {', '.join(extra)}", videos_per_arm,
                    "Control tags plus topical terms mined from the titles of "
                    "this channel's own best-performing videos."),
            ], warnings

        if key == "tag_count":
            return [
                Arm("control", current, videos_per_arm, "Current tag count."),
                Arm("expanded", "15 tags", videos_per_arm,
                    "Roughly triple the tags, drawn from the channel's own "
                    "title vocabulary."),
            ], warnings

        if key == "description_cta":
            other = "no CTA" if "has" in current else "has CTA"
            return [
                Arm("control", current, videos_per_arm, "Current description style."),
                Arm("variant", other, videos_per_arm,
                    "Same description with the call to action removed or added."),
            ], warnings

        return [
            Arm("control", current, videos_per_arm, "Current behaviour."),
            Arm("variant", f"anything other than '{current}'", videos_per_arm,
                "Any deliberate alternative -- the point is simply to create "
                "variation where there has never been any."),
        ], warnings

    # Levers that already vary: test the extremes the catalog identified.
    tested = fr.tested_buckets or fr.buckets
    if len(tested) < 2:
        biggest = max(fr.buckets, key=lambda b: b.n)
        smallest = min(fr.buckets, key=lambda b: b.n)
        return [
            Arm("control", biggest.bucket, videos_per_arm,
                f"The dominant choice ({biggest.n} videos so far)."),
            Arm("variant", smallest.bucket, videos_per_arm,
                f"The rare alternative ({smallest.n} videos so far) -- this arm "
                f"is the reason the lever could not be measured."),
        ], warnings

    best = max(tested, key=lambda b: b.mean_percentile)
    worst = min(tested, key=lambda b: b.mean_percentile)
    return [
        Arm("challenger", best.bucket, videos_per_arm,
            f"Best observed value so far ({best.mean_percentile:.1f} mean "
            f"percentile over {best.n} videos)."),
        Arm("baseline", worst.bucket, videos_per_arm,
            f"Worst observed value so far ({worst.mean_percentile:.1f} mean "
            f"percentile over {worst.n} videos)."),
    ], warnings


def _mine_tag_candidates(
    result: AnalysisResult, existing: str, count: int
) -> list[str]:
    """
    Derive candidate tags from the titles of the channel's own top videos.

    Using the channel's above-median content as the source means the proposed
    tags describe what this channel actually makes, which a generic
    "recommended tags for philosophy" list cannot do.
    """
    from collections import Counter

    from .features import content_words

    have = {t.strip().lower() for t in existing.split(",")}
    counts: Counter[str] = Counter()

    for sv in result.scored:
        if sv.eligible and sv.percentile >= 50.0:
            counts.update(set(content_words(sv.title)))

    out: list[str] = []
    for word, n in counts.most_common(count * 6):
        if word in have or len(word) < 4 or n < 5:
            continue
        out.append(word)
        if len(out) >= count:
            break
    return out


# --------------------------------------------------------------------------
# Design
# --------------------------------------------------------------------------


def _uploads_per_day(result: AnalysisResult, window_days: int = 60) -> float:
    """Recent upload cadence, used to convert 'videos needed' into 'days'."""
    dates = [
        parse_published(s.video["published_utc"])
        for s in result.scored if s.video.get("published_utc")
    ]
    if len(dates) < 2:
        return 1.0

    newest = max(dates)
    cutoff = newest - timedelta(days=window_days)
    recent = [d for d in dates if d >= cutoff]
    if len(recent) < 2:
        return 1.0

    span = max((newest - min(recent)).total_seconds() / 86400.0, 1.0)
    return max(len(recent) / span, 0.01)


def design_experiment(
    result: AnalysisResult,
    feature_key: str | None = None,
    min_effect: float | None = None,
    within_days: float = config.DEFAULT_BUDGET_DAYS,
    power: float = config.TARGET_POWER,
    alpha: float = config.ALPHA,
) -> ExperimentPlan:
    """
    Produce the next experiment for a channel.

    If `feature_key` is given, that lever is used; otherwise the highest
    ranked candidate is chosen automatically.

    SIZING

    There are two ways to size a test and they answer different questions.
    Fixing `min_effect` asks "how long until I can detect a shift this small?"
    and can easily return a year. Fixing `within_days` asks "given how fast I
    publish, what is the smallest effect I could detect by then?" -- which is
    the question someone deciding what to do this quarter is actually asking.

    Hindsight defaults to the second. Leave `min_effect` as None and the test
    is sized to read out within `within_days` at the channel's current
    cadence, never claiming to resolve less than MIN_DETECTABLE_EFFECT. Pass
    an explicit `min_effect` to override and accept whatever duration follows.

    Either way the `feasibility` table on the returned plan shows what several
    other time budgets would buy, so the trade is visible rather than buried
    in a default.
    """
    candidates = rank_candidates(result)
    if not candidates:
        raise ValueError(
            "No experiment candidates. Every lever is either already confirmed "
            "or has too little data to reason about -- run `hindsight ingest` "
            "again once more videos have been published."
        )

    if feature_key:
        match = next((c for c in candidates if c[1].feature.key == feature_key), None)
        if match is None:
            known = ", ".join(sorted(c[1].feature.key for c in candidates))
            raise ValueError(
                f"'{feature_key}' is not an experiment candidate. Available: {known}"
            )
        priority, fr, reason = match
    else:
        priority, fr, reason = candidates[0]

    # Power: the spread of cohort percentiles among eligible videos sets how
    # many samples are needed. Percentiles are bounded, so this is stable.
    percentiles = [s.percentile for s in result.scored if s.eligible]
    std_dev = float(np.std(percentiles, ddof=1)) if len(percentiles) > 1 else 28.9
    rate = _uploads_per_day(result)

    warnings: list[str] = []
    n_arms = 2  # every proposer below produces a two-arm test

    if min_effect is None:
        # Size to the time budget: how many videos will exist by the deadline,
        # split across arms, and what can that many resolve?
        affordable = max(2, int(rate * within_days / n_arms))
        min_effect = max(
            config.MIN_DETECTABLE_EFFECT,
            stats.detectable_effect(affordable, std_dev, power, alpha),
        )
        min_effect = round(min_effect, 1)
        warnings.append(
            f"Sized to read out within {within_days:.0f} days at your current "
            f"{rate:.1f} uploads/day, which means it can only resolve effects of "
            f"{min_effect:.1f} percentile points or larger. Smaller real effects "
            f"will not show up -- pass --min-effect to trade time for precision."
        )

    per_arm = stats.required_sample_size(std_dev, min_effect, power, alpha)
    arms, arm_warnings = _propose_arms(fr, result, per_arm)
    warnings.extend(arm_warnings)
    total = per_arm * len(arms)
    days = total / rate if rate > 0 else float("inf")

    # What other budgets would buy, so the trade-off is explicit.
    feasibility = [
        {
            "budget_days": b,
            "videos_per_arm": max(2, int(rate * b / n_arms)),
            "detectable_effect": round(
                stats.detectable_effect(
                    max(2, int(rate * b / n_arms)), std_dev, power, alpha
                ), 1,
            ),
        }
        for b in (14, 30, 60, 90, 180)
    ]

    # Freeze the variables already known to move the outcome. If publish hour
    # matters and you let it drift during a tag test, you have not tested tags.
    #
    # Each confounder is pinned to its own best-performing value, so freezing
    # is not merely defensive -- the experiment runs on the channel's strongest
    # known settings while it answers the new question.
    hold: list[dict[str, str]] = []
    confounders = {f.feature_key for f in result.findings} - {fr.feature.key}

    for key in sorted(confounders):
        other = next(
            (x for x in result.feature_results if x.feature.key == key), None
        )
        if other is None or other.best is None:
            continue
        best_bucket = other.best
        hold.append({
            "feature": key,
            "label": other.feature.label,
            "value": best_bucket.bucket,
            "why": (
                f"This lever is known to move results -- its best and worst "
                f"values differ by {other.spread:.1f} percentile points. "
                f"'{best_bucket.bucket}' is the strongest value observed "
                f"({best_bucket.mean_percentile:.1f} mean percentile over "
                f"{best_bucket.n} videos). Hold it there so it cannot confound "
                f"this test."
            ),
        })

    if days > within_days * 1.5:
        warnings.append(
            f"At {rate:.1f} uploads/day this test needs about {days:.0f} days. "
            f"Consider raising --min-effect: you would be trading the ability to "
            f"detect small effects for a result you can actually act on this "
            f"quarter."
        )
    if not hold:
        warnings.append(
            "No confirmed findings to hold constant, so nothing is being frozen. "
            "Keep every other setting as steady as you can for the duration."
        )

    now = datetime.now(timezone.utc)
    readout = now + timedelta(days=days + config.MIN_AGE_DAYS)
    exp_id = f"exp_{now:%Y%m%d}_{fr.feature.key}"

    return ExperimentPlan(
        experiment_id=exp_id,
        channel_id=result.channel["channel_id"],
        channel_slug=result.channel.get("slug") or result.channel["channel_id"],
        feature=fr.feature.key,
        feature_label=fr.feature.label,
        lever=fr.feature.lever,
        hypothesis=_hypothesis(fr, arms),
        rationale=reason,
        priority=priority,
        arms=arms,
        videos_per_arm=per_arm,
        total_videos=total,
        hold_constant=hold,
        assignment=(
            "Alternate arms upload by upload, in strict rotation. Do not run one "
            "arm to completion and then the other -- that tests one period "
            "against another, not one arm against the other."
        ),
        observed_std_dev=round(std_dev, 2),
        min_detectable_effect=min_effect,
        target_power=power,
        alpha=alpha,
        uploads_per_day=round(rate, 2),
        estimated_days=round(days, 1),
        earliest_readout_utc=readout.replace(microsecond=0).isoformat(),
        created_utc=now.replace(microsecond=0).isoformat(),
        readout_command=f"hindsight verdict {exp_id}",
        feasibility=feasibility,
        warnings=warnings,
    )


def _hypothesis(fr: FeatureResult, arms: Sequence[Arm]) -> str:
    names = " vs ".join(f"'{a.value}'" for a in arms)
    return (
        f"Changing {fr.feature.label.lower()} ({names}) shifts a video's "
        f"cohort-relative view percentile. Null hypothesis: it makes no "
        f"difference."
    )


def save_plan(conn, plan: ExperimentPlan) -> None:
    """Persist a designed experiment so `verdict` can read it back later."""
    db.save_experiment(conn, {
        "experiment_id": plan.experiment_id,
        "channel_id": plan.channel_id,
        "feature": plan.feature,
        "hypothesis": plan.hypothesis,
        "arms_json": json.dumps([asdict(a) for a in plan.arms], ensure_ascii=False),
        "min_per_arm": plan.videos_per_arm,
        "created_utc": plan.created_utc,
        "status": "designed",
    })


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


@dataclass
class ArmOutcome:
    name: str
    value: str
    n: int
    mean_percentile: float
    median_views: float


@dataclass
class Verdict:
    experiment_id: str
    feature: str
    status: str                 # conclusive | inconclusive | insufficient-data
    headline: str
    arms: list[ArmOutcome]
    lift: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    detectable_effect: float | None = None
    recommendation: str = ""
    videos_considered: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["arms"] = [asdict(a) for a in self.arms]
        return d


def read_verdict(
    result: AnalysisResult,
    experiment: dict[str, Any],
    min_per_arm: int | None = None,
) -> Verdict:
    """
    Evaluate an experiment against videos published since it was designed.

    Arm membership is read from what was actually published rather than from
    the plan. If the creator's pipeline diverged from the manifest -- or they
    assigned arms by hand and drifted -- the verdict still reflects reality,
    because it classifies each video by the value it actually carries.
    """
    arms_spec = json.loads(experiment["arms_json"])
    created = datetime.fromisoformat(experiment["created_utc"])
    feature_key = experiment["feature"]

    fr = next(
        (f for f in result.feature_results if f.feature.key == feature_key), None
    )
    if fr is None:
        return Verdict(
            experiment_id=experiment["experiment_id"], feature=feature_key,
            status="insufficient-data", arms=[],
            headline=f"The '{feature_key}' lever no longer exists in the analysis.",
            recommendation="Re-run `hindsight analyze` after ingesting fresh data.",
        )

    # Only videos published after the design date count as part of the test.
    since = [
        s for s in result.scored
        if s.eligible and parse_published(s.video["published_utc"]) >= created
    ]

    groups: dict[str, list[float]] = {}
    views: dict[str, list[float]] = {}
    for sv in since:
        bucket = fr.feature.extract(sv.video)
        if bucket is None:
            continue
        groups.setdefault(bucket, []).append(sv.percentile)
        views.setdefault(bucket, []).append(sv.metric_value)

    outcomes: list[ArmOutcome] = []
    for spec in arms_spec:
        pcts = groups.get(spec["value"], [])
        vs = sorted(views.get(spec["value"], []))
        median = (vs[len(vs) // 2] if len(vs) % 2 else
                  (vs[len(vs) // 2 - 1] + vs[len(vs) // 2]) / 2) if vs else 0.0
        outcomes.append(ArmOutcome(
            name=spec["name"], value=spec["value"], n=len(pcts),
            mean_percentile=sum(pcts) / len(pcts) if pcts else 0.0,
            median_views=median,
        ))

    floor = min_per_arm or max(8, config.MIN_BUCKET_N // 2)
    populated = [o for o in outcomes if o.n >= floor]

    if len(populated) < 2:
        counts = ", ".join(f"{o.name}={o.n}" for o in outcomes)
        return Verdict(
            experiment_id=experiment["experiment_id"], feature=feature_key,
            status="insufficient-data", arms=outcomes, videos_considered=len(since),
            headline=(
                f"Not enough data yet: {len(since)} eligible videos published "
                f"since the experiment was designed ({counts})."
            ),
            recommendation=(
                f"Keep publishing on the plan. Each arm needs at least {floor} "
                f"videos that are also older than {config.MIN_AGE_DAYS} days "
                f"before this can be read."
            ),
        )

    a, b = populated[0], populated[1]
    test = stats.compare(groups[a.value], groups[b.value])
    std = float(np.std(
        groups[a.value] + groups[b.value], ddof=1
    )) if (a.n + b.n) > 1 else 28.9
    resolvable = stats.detectable_effect(min(a.n, b.n), std)

    if test.p_value <= config.ALPHA:
        winner, loser = (a, b) if a.mean_percentile > b.mean_percentile else (b, a)
        return Verdict(
            experiment_id=experiment["experiment_id"], feature=feature_key,
            status="conclusive", arms=outcomes, videos_considered=len(since),
            lift=test.lift, ci_low=test.ci_low, ci_high=test.ci_high,
            p_value=test.p_value, detectable_effect=resolvable,
            headline=(
                f"'{winner.value}' beats '{loser.value}' by "
                f"{abs(test.lift):.1f} percentile points "
                f"(95% CI {test.ci_low:+.1f} to {test.ci_high:+.1f}, "
                f"p={test.p_value:.4f})."
            ),
            recommendation=(
                f"Adopt '{winner.value}' as the default and design the next "
                f"experiment on a different lever. Median views per video in "
                f"the winning arm: {winner.median_views:.0f} vs "
                f"{loser.median_views:.0f}."
            ),
        )

    return Verdict(
        experiment_id=experiment["experiment_id"], feature=feature_key,
        status="inconclusive", arms=outcomes, videos_considered=len(since),
        lift=test.lift, ci_low=test.ci_low, ci_high=test.ci_high,
        p_value=test.p_value, detectable_effect=resolvable,
        headline=(
            f"No detectable difference between arms "
            f"({test.lift:+.1f} points, 95% CI {test.ci_low:+.1f} to "
            f"{test.ci_high:+.1f}, p={test.p_value:.3f})."
        ),
        recommendation=(
            f"With {min(a.n, b.n)} videos in the smaller arm this test could "
            f"only resolve effects of about {resolvable:.1f} percentile points "
            f"or larger. The honest reading is 'no large effect', not 'no "
            f"effect' -- either keep running to narrow the interval, or accept "
            f"that this lever is not worth more uploads and move to the next one."
        ),
    )
