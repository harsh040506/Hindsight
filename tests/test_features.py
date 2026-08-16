"""
Tests for feature extraction and adaptive bucketing.

The bucketing behaviour matters more than any individual extractor: a tool
that hardcodes "short = under 60 seconds" produces one bucket and zero
findings on a channel that only posts 12-second clips. `adaptive_buckets` is
what lets the same code work on both, so its two regimes are tested directly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hindsight import features as feat


class TestTextHelpers:
    def test_detects_emoji(self):
        assert feat.has_emoji("Wisdom 🧠")
        assert not feat.has_emoji("Wisdom")

    def test_extracts_emoji_in_order(self):
        assert feat.emojis_in("a 🧠 b 🌌") == ["🧠", "🌌"]

    def test_words_are_lowercased_and_depunctuated(self):
        assert feat.words_in("The Mind, truly!") == ["the", "mind", "truly"]

    def test_content_words_drop_stopwords_and_short_tokens(self):
        assert feat.content_words("The mind is a quiet thing") == \
               ["mind", "quiet", "thing"]

    def test_handles_empty_and_none(self):
        assert feat.words_in("") == []
        assert feat.emojis_in(None) == []
        assert not feat.has_emoji(None)


class TestAdaptiveBuckets:
    def test_uses_exact_values_when_data_is_clustered(self):
        """An automated pipeline renders a few fixed lengths; use those."""
        values = [12.0] * 400 + [33.0] * 250 + [17.0] * 180
        label = feat.adaptive_buckets(values)
        assert label(12.0) == "12"
        assert label(33.0) == "33"

    def test_falls_back_to_quartiles_when_continuous(self):
        values = [float(i) for i in range(1000)]
        label = feat.adaptive_buckets(values)
        labels = {label(v) for v in values}
        assert len(labels) == 4
        assert any(l.startswith("<=") for l in labels)
        assert any(l.startswith(">") for l in labels)

    def test_rare_values_fall_outside_exact_buckets(self):
        values = [12.0] * 500 + [33.0] * 400 + [999.0]
        label = feat.adaptive_buckets(values, max_exact=2)
        assert label(999.0) is None

    def test_single_value_produces_no_split(self):
        label = feat.adaptive_buckets([5.0] * 100)
        assert label(5.0) is None or label(5.0) == "5"

    def test_empty_input_labels_nothing(self):
        assert feat.adaptive_buckets([])(1.0) is None

    def test_integers_render_without_decimals(self):
        label = feat.adaptive_buckets([10.0] * 50 + [20.0] * 50)
        assert label(10.0) == "10"


class TestExtractors:
    def test_sentence_form(self):
        assert feat._sentence_form("Why?") == "question"
        assert feat._sentence_form("Go!") == "exclamation"
        assert feat._sentence_form("It is.") == "statement"

    def test_clause_bucket_counts_commas(self):
        assert feat._clause_bucket("One thing") == "single clause"
        assert feat._clause_bucket("One, two") == "two clauses"
        assert feat._clause_bucket("One, two, three") == "three or more clauses"

    def test_cta_detection(self):
        assert feat._cta_bucket("Follow @someone for more") == "has CTA"
        assert feat._cta_bucket("Comment RESONATE below") == "has CTA"
        assert feat._cta_bucket("A quiet description.") == "no CTA"

    def test_first_word(self):
        assert feat._first_word("True wisdom begins") == "true"
        assert feat._first_word("") is None

    def test_suffix_handles_plain_and_range_labels(self):
        assert feat._suffix("12", "s") == "12s"
        assert feat._suffix("<= 920", " chars") == "<= 920 chars"
        assert feat._suffix("54-64", " chars") == "54-64 chars"
        assert feat._suffix(None, "s") is None


class TestTiming:
    @staticmethod
    def video(hour: int):
        dt = datetime(2026, 3, 4, hour, tzinfo=timezone.utc)  # a Wednesday
        return {"published_utc": dt.isoformat().replace("+00:00", "Z")}

    def test_hour_bucket_is_zero_padded(self):
        assert feat._hour_bucket(self.video(9), 0.0) == "09:00"

    def test_timezone_offset_shifts_the_hour(self):
        """The API reports UTC; advice must be in the creator's local time."""
        assert feat._hour_bucket(self.video(12), 5.5) == "17:00"

    def test_offset_can_roll_over_the_day(self):
        assert feat._dow(self.video(22), 5.5) == "Thursday"
        assert feat._dow(self.video(12), 0.0) == "Wednesday"

    def test_slots_partition_the_day(self):
        assert feat._slot(self.video(3), 0.0) == "night (00-06)"
        assert feat._slot(self.video(9), 0.0) == "morning (06-12)"
        assert feat._slot(self.video(15), 0.0) == "afternoon (12-18)"
        assert feat._slot(self.video(21), 0.0) == "evening (18-24)"

    def test_missing_timestamp_yields_none(self):
        assert feat._hour_bucket({}, 0.0) is None


class TestBuildFeatures:
    @staticmethod
    def corpus(n=200):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        return [{
            "video_id": f"v{i}",
            "title": f"True wisdom number {i} 🧠" if i % 2 else f"Your quiet mind {i}",
            "description": "Follow @x for more. " + "y" * (i % 300),
            "published_utc": (base + timedelta(hours=i * 3)).isoformat().replace("+00:00", "Z"),
            "duration_s": 12 if i % 3 else 33,
            "tags": ["a", "b"],
            "views": 100 + i,
            "likes": i,
        } for i in range(n)]

    def test_covers_all_three_lever_groups(self):
        fs = feat.build_features(self.corpus())
        assert {f.group for f in fs} == {"metadata", "title", "timing"}

    def test_feature_keys_are_unique(self):
        keys = [f.key for f in feat.build_features(self.corpus())]
        assert len(keys) == len(set(keys))

    def test_every_feature_has_an_actionable_lever(self):
        """A finding you cannot act on is a distraction, so all levers read
        as instructions to the creator."""
        for f in feat.build_features(self.corpus()):
            assert f.lever and f.lever[0].isupper()
            assert f.label

    def test_extractors_never_raise_on_sparse_records(self):
        fs = feat.build_features(self.corpus())
        sparse = {"video_id": "x", "published_utc": "", "title": "",
                  "description": "", "tags": [], "duration_s": 0}
        for f in fs:
            f.extract(sparse)  # must not raise

    def test_tag_set_supersedes_tag_count(self):
        fs = {f.key: f for f in feat.build_features(self.corpus())}
        assert "tag_count" in fs["tag_set"].supersedes

    def test_discovered_keywords_reflect_the_corpus(self):
        fs = feat.build_features(self.corpus(), min_keyword_docs=10)
        kw = {f.key for f in fs if f.key.startswith("kw_")}
        assert any(k in kw for k in ("kw_wisdom", "kw_true", "kw_quiet", "kw_mind"))
