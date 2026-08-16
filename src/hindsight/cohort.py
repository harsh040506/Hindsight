"""
Cohort-relative scoring: making a video from last July comparable to one from
last week.

THE PROBLEM

Raw view counts cannot be compared across a catalog. A video published eleven
months ago has had eleven months to accumulate views. A channel that had 80
subscribers last summer and 531 now pushes every recent upload to a larger
audience before the content matters at all. Sort a year of uploads by views
and you have mostly measured *when they were published*, then dressed it up as
a content insight.

The usual fixes are worse than the problem. Views-per-day punishes evergreen
content and rewards flash-in-the-pan spikes. Fitting a growth curve and taking
residuals means your findings inherit every assumption in the curve. Looking
only at the last 30 days throws away 90% of the evidence.

THE APPROACH

Score every video against its neighbours in publication order.

For the video at position i, its cohort is the COHORT_HALF_WIDTH videos
published immediately before it and the same number immediately after. Those
videos share, to a very close approximation, the same channel size, the same
season, the same algorithmic weather, and the same amount of time on the
platform. The score is the video's percentile rank within that cohort.

The result is a number that means "this video beat 78% of the videos published
around the same time as it" -- which is the comparison a creator actually
wants, and which is stable to compare across the whole catalog because every
video's score is expressed relative to its own local baseline.

WHY THIS IS ROBUST

Anything that moves slowly relative to the cohort window -- subscriber growth,
seasonal demand, a change in how the algorithm treats the channel, the natural
accumulation of views with age -- is shared by every member of a cohort and
therefore cancels out of the percentile. What survives is the part that varies
*within* a cohort, which is precisely the per-video choices: the title, the
length, the hour it went out.

The trade-off is that Hindsight cannot detect anything that changed slowly and
monotonically across the entire catalog, because that is exactly what the
method removes. This is stated in the report rather than hidden, and it is the
right trade: a slow channel-wide drift is not a lever you can pull on the next
upload anyway.

EDGE HANDLING

Videos near either end of the catalog cannot have a symmetric cohort. The
oldest video's neighbours are all newer than it, and on a growing channel
newer means more views, so its percentile is biased downward by an artifact of
where it sits in the list rather than by anything about the video. Those
videos are scored (the number is still shown) but flagged `cohort_full=False`
and excluded from the lift analysis by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Sequence

from . import config


@dataclass
class ScoredVideo:
    """A video with its cohort-relative standing on one metric."""

    video: dict[str, Any]
    metric_value: float
    percentile: float          # 0-100, position within its own cohort
    cohort_size: int
    cohort_median: float
    cohort_full: bool          # symmetric cohort on both sides
    eligible: bool             # passed every filter; used in the analysis
    exclusion_reason: str | None = None

    @property
    def video_id(self) -> str:
        return self.video["video_id"]

    @property
    def title(self) -> str:
        return self.video.get("title", "")


def parse_published(value: str) -> datetime:
    """Parse the API's RFC-3339 publish timestamp into an aware datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def views_metric(video: dict[str, Any]) -> float:
    """Raw view count -- how far the video was distributed."""
    return float(video.get("views") or 0)


def like_rate_metric(video: dict[str, Any]) -> float:
    """
    Likes per thousand views -- how strongly it landed with the people reached.

    Distribution and resonance are different questions and can point in
    opposite directions: a video can be pushed to a huge cold audience and
    land badly, or reach few people and delight all of them. Hindsight scores
    both so a "winning" title that buys reach at the cost of resonance is
    visible rather than celebrated.
    """
    views = float(video.get("views") or 0)
    if views <= 0:
        return 0.0
    return float(video.get("likes") or 0) / views * 1000.0


def _percentile_within(value: float, others: Sequence[float]) -> float:
    """
    Percentile rank of `value` among `others`, counting ties at half weight.

    Half-weighting ties is what makes the scale symmetric: a video in a cohort
    where every video scored identically lands at 50, not at 0 or 100. On a
    catalog with many low-view videos sharing exact counts, the naive
    "fraction strictly below" would push a whole cluster to 0 and invent a
    difference between videos that performed the same.
    """
    if not others:
        return 50.0
    below = sum(1 for o in others if o < value)
    tied = sum(1 for o in others if o == value)
    return (below + 0.5 * tied) / len(others) * 100.0


def score_catalog(
    videos: Sequence[dict[str, Any]],
    metric: Callable[[dict[str, Any]], float] = views_metric,
    half_width: int = config.COHORT_HALF_WIDTH,
    min_age_days: int = config.MIN_AGE_DAYS,
    now: datetime | None = None,
) -> list[ScoredVideo]:
    """
    Score every video against its publication-time cohort.

    `videos` must be sorted ascending by `published_utc`; ingestion returns
    them that way and this function re-sorts defensively rather than trusting
    the caller.

    Eligibility for the downstream analysis requires all of:
      - a symmetric, full-width cohort
      - at least `min_age_days` since publication
      - a non-zero duration (a zero means still-processing or a live stream)

    Ineligible videos keep their scores -- they appear in the report's catalog
    view -- but do not contribute evidence to any finding.
    """
    ordered = sorted(videos, key=lambda v: v["published_utc"])
    values = [metric(v) for v in ordered]
    n = len(ordered)
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=min_age_days)

    scored: list[ScoredVideo] = []
    for i, video in enumerate(ordered):
        lo, hi = max(0, i - half_width), min(n, i + half_width + 1)
        neighbours = values[lo:i] + values[i + 1:hi]

        full = (i - half_width >= 0) and (i + half_width < n)
        pct = _percentile_within(values[i], neighbours)

        srt = sorted(neighbours)
        if srt:
            mid = len(srt) // 2
            median = srt[mid] if len(srt) % 2 else (srt[mid - 1] + srt[mid]) / 2
        else:
            median = 0.0

        reason: str | None = None
        if not full:
            reason = "edge of catalog (no symmetric cohort)"
        elif parse_published(video["published_utc"]) > cutoff:
            reason = f"younger than {min_age_days} days (still accumulating views)"
        elif not video.get("duration_s"):
            reason = "zero duration (live broadcast or still processing)"

        scored.append(ScoredVideo(
            video=video,
            metric_value=values[i],
            percentile=pct,
            cohort_size=len(neighbours),
            cohort_median=median,
            cohort_full=full,
            eligible=reason is None,
            exclusion_reason=reason,
        ))

    return scored


def eligible_only(scored: Sequence[ScoredVideo]) -> list[ScoredVideo]:
    """Filter to the videos that may contribute evidence to a finding."""
    return [s for s in scored if s.eligible]


def summarize(scored: Sequence[ScoredVideo]) -> dict[str, Any]:
    """Headline counts for the report header and CLI output."""
    eligible = eligible_only(scored)
    excluded: dict[str, int] = {}
    for s in scored:
        if s.exclusion_reason:
            excluded[s.exclusion_reason] = excluded.get(s.exclusion_reason, 0) + 1

    values = sorted(s.metric_value for s in eligible)
    if values:
        mid = len(values) // 2
        median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    else:
        median = 0.0

    return {
        "total": len(scored),
        "eligible": len(eligible),
        "excluded": excluded,
        "median_metric": median,
        "mean_metric": sum(values) / len(values) if values else 0.0,
        "max_metric": values[-1] if values else 0.0,
        "min_metric": values[0] if values else 0.0,
        "total_metric": sum(values),
    }
