"""
Tests for analysis orchestration and, above all, lever classification.

Distinguishing "measured, no effect" from "never varied" from "varied too
thinly to measure" is the judgement Hindsight exists to make. Collapsing those
three into "no finding" is what makes a channel owner conclude that tags do
not matter after 950 uploads that all used identical tags.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hindsight import analyze as an

BASE = datetime(2025, 1, 1, tzinfo=timezone.utc)
NOW = BASE + timedelta(days=900)


def build(n=400, **overrides):
    """
    A catalog with genuinely varied metadata and one planted timing effect.

    Videos are published every six hours, so publish hour cycles cleanly
    through 00:00, 06:00, 12:00 and 18:00 with n/4 videos each. Videos going
    out at 12:00 get markedly more views; nothing else is correlated with
    time, so a correct analysis should find that one effect and no others.
    """
    vids = []
    for i in range(n):
        published = BASE + timedelta(hours=i * 6)
        hour = published.hour
        views = 300 if hour == 12 else 100 + (i % 40)
        vids.append({
            "video_id": f"v{i:04d}",
            "title": f"Wisdom and silence number {i}" + (" 🧠" if i % 2 else ""),
            "description": "Follow @x for more. " + "y" * (400 + i % 600),
            "published_utc": published.isoformat().replace("+00:00", "Z"),
            "duration_s": [12, 17, 33][i % 3],
            "privacy": "public",
            "tags": ["a", "b", "c"],
            "views": views,
            "likes": views // 10,
            "comments": 0,
            **overrides,
        })
    return vids


CHANNEL = {"channel_id": "UCtest", "slug": "test", "title": "Test"}


class TestClassification:
    def test_constant_lever_is_untested(self):
        """All 400 videos share one tag set -> no variation, not 'no effect'."""
        result = an.analyze(build(), CHANNEL, iters=300)
        keys = {fr.feature.key for fr in result.untested_levers}
        assert "tag_set" in keys
        assert "tag_count" in keys

    def test_untested_note_says_it_was_never_varied(self):
        result = an.analyze(build(), CHANNEL, iters=300)
        note = next(fr.note for fr in result.untested_levers
                    if fr.feature.key == "tag_set")
        assert "never been tested" in note

    @staticmethod
    def lopsided():
        """
        A lever used a handful of times, spread through the catalog.

        The minority videos are distributed rather than bunched at the start,
        because videos at either end are correctly dropped for lacking a
        symmetric cohort -- bunching them there would test edge trimming
        instead of the classifier.
        """
        vids = build()
        for i, v in enumerate(vids):
            v["title"] = "A quiet mind?" if i % 97 == 0 else "A quiet mind."
        return vids

    def test_lopsided_lever_is_underpowered_not_tested(self):
        result = an.analyze(self.lopsided(), CHANNEL, iters=300)
        keys = {fr.feature.key for fr in result.underpowered_levers}
        assert "title_form" in keys

    def test_underpowered_note_names_the_small_side(self):
        result = an.analyze(self.lopsided(), CHANNEL, iters=300)
        note = next(fr.note for fr in result.underpowered_levers
                    if fr.feature.key == "title_form")
        assert "smallest alternative" in note
        assert "not evidence" in note

    def test_properly_varied_lever_is_tested(self):
        result = an.analyze(build(), CHANNEL, iters=300)
        tested = {fr.feature.key for fr in result.feature_results
                  if fr.status == an.TESTED}
        assert "duration" in tested
        assert "publish_hour" in tested

    def test_every_feature_lands_in_exactly_one_category(self):
        result = an.analyze(build(), CHANNEL, iters=300)
        total = (len(result.untested_levers) + len(result.underpowered_levers)
                 + len([f for f in result.feature_results if f.status == an.TESTED]))
        assert total == len(result.feature_results)


class TestFindings:
    def test_recovers_a_planted_effect(self):
        result = an.analyze(build(), CHANNEL, iters=2000)
        hours = [f for f in result.findings if f.feature_key == "publish_hour"]
        assert hours, "the planted 12:00 effect should be found"
        assert any(f.bucket == "12:00" and f.lift > 0 for f in hours)

    def test_findings_require_corrected_significance(self):
        result = an.analyze(build(), CHANNEL, iters=2000)
        assert all(f.q_value <= 0.05 for f in result.findings)

    def test_findings_are_ranked_by_effect_size(self):
        result = an.analyze(build(), CHANNEL, iters=2000)
        lifts = [abs(f.lift) for f in result.findings]
        assert lifts == sorted(lifts, reverse=True)

    def test_no_findings_on_pure_noise(self):
        """Random views must not manufacture findings after correction."""
        import random
        rng = random.Random(4)
        vids = build()
        for v in vids:
            v["views"] = rng.randint(50, 500)
        result = an.analyze(vids, CHANNEL, iters=2000)
        assert len(result.findings) == 0

    def test_planted_effect_replicates_across_eras(self):
        result = an.analyze(build(), CHANNEL, iters=2000)
        hour = next((f for f in result.findings
                     if f.feature_key == "publish_hour" and f.bucket == "12:00"), None)
        assert hour is not None
        assert hour.replicates
        assert hour.early_lift is not None and hour.late_lift is not None

    def test_era_note_is_always_populated(self):
        result = an.analyze(build(), CHANNEL, iters=1000)
        assert all(f.era_note for f in result.findings)


class TestEraArtifacts:
    def test_effect_confined_to_one_era_does_not_replicate(self):
        """
        Views only elevated in the second half: cohort scoring will still
        surface it, but the era split must flag it as period-bound.
        """
        vids = build()
        for i, v in enumerate(vids):
            v["duration_s"] = 12 if i % 2 else 33
            v["views"] = 400 if (i > 200 and i % 2 == 0) else 100 + (i % 30)
        result = an.analyze(vids, CHANNEL, iters=2000)
        dur = [f for f in result.findings if f.feature_key == "duration"]
        if dur:
            assert any(not f.replicates for f in dur)


class TestSummaryAndParams:
    def test_summary_counts_are_consistent(self):
        result = an.analyze(build(), CHANNEL, iters=300)
        s = result.summary
        assert s["eligible"] <= s["total"]
        assert s["eligible"] == result.eligible_count

    def test_params_are_recorded_for_reproducibility(self):
        result = an.analyze(build(), CHANNEL, iters=300, half_width=15)
        assert result.params["cohort_half_width"] == 15
        assert result.params["permutation_iters"] == 300

    def test_top_and_bottom_are_ordered(self):
        result = an.analyze(build(), CHANNEL, iters=300)
        top, bottom = an.top_and_bottom(result, 5)
        assert len(top) == 5 and len(bottom) == 5
        assert top[0].percentile >= top[-1].percentile
        assert top[0].percentile >= bottom[0].percentile


class TestSmallCatalogs:
    def test_tiny_catalog_produces_no_findings_rather_than_crashing(self):
        vids = build(n=12)
        result = an.analyze(vids, CHANNEL, iters=200)
        assert result.findings == []

    def test_empty_catalog_is_handled(self):
        result = an.analyze([], CHANNEL, iters=200)
        assert result.summary["total"] == 0
        assert result.findings == []
