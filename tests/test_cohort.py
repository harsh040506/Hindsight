"""
Tests for cohort-relative scoring.

The centrepiece is `TestGrowthCancellation`. Hindsight's whole claim is that
scoring a video against its publication-time neighbours removes channel growth
and video age while preserving the effect of the creator's choices. That is a
falsifiable claim, so it is tested directly: build a synthetic catalog where a
feature has a known effect *and* views grow steeply over time, then confirm
that raw view counts give the wrong answer and cohort percentiles give the
right one.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hindsight import cohort


def make_video(idx: int, published: datetime, views: int, duration: int = 30,
               title: str = "t", **extra):
    return {
        "video_id": f"v{idx:04d}",
        "title": title,
        "description": "",
        "published_utc": published.isoformat().replace("+00:00", "Z"),
        "duration_s": duration,
        "privacy": "public",
        "tags": [],
        "views": views,
        "likes": 0,
        "comments": 0,
        **extra,
    }


BASE = datetime(2025, 1, 1, tzinfo=timezone.utc)


class TestPercentileWithin:
    def test_all_equal_scores_fifty(self):
        """Ties at half weight: a video identical to its cohort is average."""
        assert cohort._percentile_within(5.0, [5.0] * 10) == 50.0

    def test_highest_value_scores_one_hundred(self):
        assert cohort._percentile_within(100.0, [1.0, 2.0, 3.0]) == 100.0

    def test_lowest_value_scores_zero(self):
        assert cohort._percentile_within(0.0, [1.0, 2.0, 3.0]) == 0.0

    def test_median_value_scores_fifty(self):
        assert cohort._percentile_within(2.0, [1.0, 3.0]) == 50.0

    def test_partial_ties_split_the_difference(self):
        # Two below, two tied, two above -> (2 + 0.5*2) / 6 = 50%
        assert cohort._percentile_within(5.0, [1.0, 2.0, 5.0, 5.0, 8.0, 9.0]) == 50.0

    def test_empty_cohort_is_neutral(self):
        assert cohort._percentile_within(5.0, []) == 50.0


class TestScoreCatalog:
    def test_scores_every_video(self):
        vids = [make_video(i, BASE + timedelta(days=i), 100 + i) for i in range(60)]
        scored = cohort.score_catalog(vids, half_width=5, min_age_days=0,
                                      now=BASE + timedelta(days=400))
        assert len(scored) == 60
        assert all(0.0 <= s.percentile <= 100.0 for s in scored)

    def test_edges_lack_a_symmetric_cohort(self):
        vids = [make_video(i, BASE + timedelta(days=i), 100) for i in range(40)]
        scored = cohort.score_catalog(vids, half_width=5, min_age_days=0,
                                      now=BASE + timedelta(days=400))
        assert not scored[0].cohort_full
        assert not scored[-1].cohort_full
        assert scored[20].cohort_full

    def test_edge_videos_are_excluded_with_a_reason(self):
        vids = [make_video(i, BASE + timedelta(days=i), 100) for i in range(40)]
        scored = cohort.score_catalog(vids, half_width=5, min_age_days=0,
                                      now=BASE + timedelta(days=400))
        assert not scored[0].eligible
        assert "edge of catalog" in scored[0].exclusion_reason

    def test_young_videos_are_excluded(self):
        now = BASE + timedelta(days=100)
        vids = [make_video(i, BASE + timedelta(days=i), 100) for i in range(100)]
        scored = cohort.score_catalog(vids, half_width=5, min_age_days=14, now=now)
        recent = [s for s in scored if not s.cohort_full or s.exclusion_reason]
        assert any("younger than 14 days" in (s.exclusion_reason or "")
                   for s in recent)

    def test_zero_duration_is_excluded(self):
        vids = [make_video(i, BASE + timedelta(days=i), 100) for i in range(40)]
        vids[20]["duration_s"] = 0
        scored = cohort.score_catalog(vids, half_width=5, min_age_days=0,
                                      now=BASE + timedelta(days=400))
        assert not scored[20].eligible
        assert "zero duration" in scored[20].exclusion_reason

    def test_input_order_does_not_matter(self):
        vids = [make_video(i, BASE + timedelta(days=i), i * 7 % 500)
                for i in range(50)]
        forward = cohort.score_catalog(vids, half_width=5, min_age_days=0,
                                       now=BASE + timedelta(days=400))
        shuffled = cohort.score_catalog(list(reversed(vids)), half_width=5,
                                        min_age_days=0,
                                        now=BASE + timedelta(days=400))
        assert [s.video_id for s in forward] == [s.video_id for s in shuffled]
        assert [round(s.percentile, 6) for s in forward] == \
               [round(s.percentile, 6) for s in shuffled]


class TestGrowthCancellation:
    """
    The core claim, tested against a catalog built to break a naive method.

    Construction: 400 videos over 400 days. Baseline views grow 10x across the
    period, which is what a channel gaining subscribers looks like. Every third
    video carries a "boost" feature worth +40% views. Crucially, the boosted
    videos are spread evenly across time, so any correct method should find the
    boost and no method should confuse it with the growth.
    """

    @staticmethod
    def build():
        import random
        rng = random.Random(11)
        vids = []
        for i in range(400):
            baseline = 100 + i * 2.25          # 100 -> ~1000 over the period
            boosted = (i % 3 == 0)
            # Per-video noise, so percentiles spread the way real ones do
            # rather than collapsing onto a handful of exact values.
            jitter = 1.0 + rng.uniform(-0.15, 0.15)
            views = int(baseline * (1.4 if boosted else 1.0) * jitter)
            vids.append(make_video(
                i, BASE + timedelta(days=i), views, boosted=boosted
            ))
        return vids

    def test_raw_views_are_dominated_by_publish_date(self):
        """The failure mode this whole module exists to avoid."""
        vids = self.build()
        first_half = [v["views"] for v in vids[:200]]
        second_half = [v["views"] for v in vids[200:]]
        # Later videos look far better purely because the channel grew.
        assert (sum(second_half) / len(second_half)) > \
               2.0 * (sum(first_half) / len(first_half))

    def test_cohort_percentile_removes_the_growth_trend(self):
        vids = self.build()
        scored = cohort.score_catalog(vids, half_width=25, min_age_days=0,
                                      now=BASE + timedelta(days=800))
        eligible = cohort.eligible_only(scored)

        half = len(eligible) // 2
        early = [s.percentile for s in eligible[:half]]
        late = [s.percentile for s in eligible[half:]]

        # After normalisation the two eras must look the same, because nothing
        # about the *choices* differed between them.
        assert abs(sum(early) / len(early) - sum(late) / len(late)) < 5.0

    def test_cohort_percentile_still_detects_the_real_effect(self):
        vids = self.build()
        scored = cohort.score_catalog(vids, half_width=25, min_age_days=0,
                                      now=BASE + timedelta(days=800))
        eligible = cohort.eligible_only(scored)

        boosted = [s.percentile for s in eligible if s.video["boosted"]]
        plain = [s.percentile for s in eligible if not s.video["boosted"]]

        # Removing the trend must not remove the signal.
        assert sum(boosted) / len(boosted) > sum(plain) / len(plain) + 25.0

    def test_percentiles_span_the_full_range(self):
        vids = self.build()
        scored = cohort.score_catalog(vids, half_width=25, min_age_days=0,
                                      now=BASE + timedelta(days=800))
        pcts = [s.percentile for s in cohort.eligible_only(scored)]
        assert min(pcts) < 20.0 and max(pcts) > 80.0


class TestMetrics:
    def test_views_metric_reads_the_view_count(self):
        assert cohort.views_metric({"views": 42}) == 42.0

    def test_like_rate_is_per_thousand_views(self):
        assert cohort.like_rate_metric({"views": 1000, "likes": 25}) == 25.0

    def test_like_rate_of_unviewed_video_is_zero_not_an_error(self):
        assert cohort.like_rate_metric({"views": 0, "likes": 5}) == 0.0

    def test_like_rate_is_independent_of_reach(self):
        """A small video that lands well outscores a big one that does not."""
        small = cohort.like_rate_metric({"views": 100, "likes": 10})
        large = cohort.like_rate_metric({"views": 10000, "likes": 100})
        assert small > large


class TestSummarize:
    def test_counts_and_reasons(self):
        vids = [make_video(i, BASE + timedelta(days=i), 100 + i) for i in range(60)]
        scored = cohort.score_catalog(vids, half_width=5, min_age_days=0,
                                      now=BASE + timedelta(days=400))
        s = cohort.summarize(scored)
        assert s["total"] == 60
        assert s["eligible"] == 50          # 5 trimmed at each end
        assert "edge of catalog (no symmetric cohort)" in s["excluded"]

    def test_handles_an_empty_catalog(self):
        s = cohort.summarize([])
        assert s["total"] == 0 and s["eligible"] == 0
        assert s["median_metric"] == 0.0
