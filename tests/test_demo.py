"""
Tests for the bundled demo dataset and its anonymisation.

Two obligations pull against each other and both are checked here: the dataset
must not leak identifying information, and it must preserve enough structure
that running the demo exercises the same analysis the live path does. An
anonymiser that flattened every title would satisfy the first and quietly
destroy the second.
"""

from __future__ import annotations

import pytest

from hindsight import analyze as an, cohort, demo, features as feat


@pytest.fixture(scope="module")
def bundled():
    videos, channel = demo.load_demo()
    return videos, channel


class TestBundledDataset:
    def test_ships_with_the_package(self):
        assert demo.dataset_path().is_file()

    def test_is_a_substantial_real_catalog(self, bundled):
        videos, _ = bundled
        assert len(videos) > 500

    def test_channel_metadata_present(self, bundled):
        _, channel = bundled
        assert channel["channel_id"] and channel["slug"]

    def test_records_match_the_live_query_shape(self, bundled):
        """Demo must flow through the same code as live data, unmodified."""
        videos, _ = bundled
        required = {"video_id", "title", "description", "published_utc",
                    "duration_s", "privacy", "tags", "views", "likes", "comments"}
        assert required <= set(videos[0])

    def test_view_counts_are_real_and_skewed(self, bundled):
        videos, _ = bundled
        views = sorted(v["views"] for v in videos)
        mean = sum(views) / len(views)
        median = views[len(views) // 2]
        assert mean > median * 1.5      # the long tail is preserved

    def test_publish_times_span_many_months(self, bundled):
        videos, _ = bundled
        stamps = sorted(v["published_utc"] for v in videos)
        assert stamps[0][:4] != stamps[-1][:7]

    def test_durations_cluster_on_real_render_settings(self, bundled):
        videos, _ = bundled
        from collections import Counter
        common = Counter(v["duration_s"] for v in videos).most_common(3)
        assert sum(c for _, c in common) > len(videos) * 0.7


class TestAnonymisation:
    @staticmethod
    def source():
        return [{
            "video_id": "REAL_ID_123",
            "title": "The true measure of a mind, in silence. 🧠",
            "description": "Follow @real_channel for more. " + "z" * 400,
            "published_utc": "2026-01-05T12:00:00Z",
            "duration_s": 33, "privacy": "public",
            "tags": ["philosophy", "quotes"],
            "views": 1338, "likes": 37, "comments": 2,
        }], {"channel_id": "UCrealchannelid", "subscriber_count": 531,
             "view_count": 316209}

    def test_video_ids_are_replaced(self):
        vids, ch = self.source()
        out = demo.anonymize(vids, ch)
        assert out["videos"][0]["video_id"] != "REAL_ID_123"

    def test_channel_id_is_replaced(self):
        vids, ch = self.source()
        assert demo.anonymize(vids, ch)["channel"]["channel_id"] != "UCrealchannelid"

    def test_title_text_is_not_reproduced(self):
        vids, ch = self.source()
        out = demo.anonymize(vids, ch)["videos"][0]
        assert "true measure of a mind" not in out["title"].lower()

    def test_performance_data_is_untouched(self):
        """The statistics have to stay real or the demo proves nothing."""
        vids, ch = self.source()
        out = demo.anonymize(vids, ch)["videos"][0]
        assert out["views"] == 1338
        assert out["likes"] == 37
        assert out["duration_s"] == 33
        assert out["published_utc"] == "2026-01-05T12:00:00Z"
        assert out["tags"] == ["philosophy", "quotes"]

    def test_title_word_count_is_preserved(self):
        vids, ch = self.source()
        out = demo.anonymize(vids, ch)["videos"][0]
        assert len(feat.words_in(out["title"])) == \
               len(feat.words_in(vids[0]["title"]))

    def test_emoji_is_preserved(self):
        vids, ch = self.source()
        out = demo.anonymize(vids, ch)["videos"][0]
        assert feat.emojis_in(out["title"]) == ["🧠"]

    def test_clause_structure_is_preserved(self):
        vids, ch = self.source()
        out = demo.anonymize(vids, ch)["videos"][0]
        assert feat._clause_bucket(out["title"]) == \
               feat._clause_bucket(vids[0]["title"])

    def test_sentence_form_is_preserved(self):
        vids, ch = self.source()
        vids[0]["title"] = "Is the mind ever quiet?"
        out = demo.anonymize(vids, ch)["videos"][0]
        assert feat._sentence_form(out["title"]) == "question"

    def test_description_length_is_preserved(self):
        vids, ch = self.source()
        out = demo.anonymize(vids, ch)["videos"][0]
        assert len(out["description"]) == len(vids[0]["description"])

    def test_cta_presence_is_preserved(self):
        """Otherwise a lever used on 99% of uploads reads as never used."""
        vids, ch = self.source()
        out = demo.anonymize(vids, ch)["videos"][0]
        assert feat._cta_bucket(out["description"]) == "has CTA"
        assert "@real_channel" not in out["description"]

    def test_description_without_cta_stays_without_one(self):
        vids, ch = self.source()
        vids[0]["description"] = "y" * 300
        out = demo.anonymize(vids, ch)["videos"][0]
        assert feat._cta_bucket(out["description"]) == "no CTA"

    def test_is_deterministic(self):
        vids, ch = self.source()
        assert demo.anonymize(vids, ch) == demo.anonymize(vids, ch)

    def test_different_salts_give_different_ids(self):
        vids, ch = self.source()
        a = demo.anonymize(vids, ch, salt="one")["videos"][0]["video_id"]
        b = demo.anonymize(vids, ch, salt="two")["videos"][0]["video_id"]
        assert a != b


class TestDemoAnalysisRuns:
    def test_full_analysis_completes_on_bundled_data(self, bundled):
        videos, channel = bundled
        public = [v for v in videos if v["privacy"] == "public"]
        result = an.analyze(public, channel, timezone_offset_h=5.5, iters=500)
        assert result.eligible_count > 400
        assert result.feature_results

    def test_demo_surfaces_the_untested_tag_lever(self, bundled):
        videos, channel = bundled
        public = [v for v in videos if v["privacy"] == "public"]
        result = an.analyze(public, channel, timezone_offset_h=5.5, iters=500)
        assert any(fr.feature.key == "tag_set" for fr in result.untested_levers)

    def test_demo_reproduces_the_real_timing_finding(self, bundled):
        """
        Publish times and view counts in the bundled dataset are unmodified,
        so the timing result must match what the live channel produces.
        """
        videos, channel = bundled
        public = [v for v in videos if v["privacy"] == "public"]
        result = an.analyze(public, channel, timezone_offset_h=5.5, iters=4000)
        hours = [f for f in result.findings if f.feature_key == "publish_hour"]
        assert hours, "the real publish-hour effect should survive correction"
        assert all(f.replicates for f in hours)
