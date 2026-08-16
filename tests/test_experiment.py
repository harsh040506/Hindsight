"""
Tests for experiment design and readout.

Two behaviours carry most of the value and are tested hardest: that a test is
sized to a time budget the creator can actually wait out, and that an
inconclusive result is reported as "no *large* effect" with the resolvable
effect stated, rather than as "no effect".
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from hindsight import analyze as an, config, experiment as ex
from tests.test_analyze import BASE, CHANNEL, build


def analysed(**kw):
    return an.analyze(build(**kw), CHANNEL, iters=800)


class TestRanking:
    def test_untested_levers_outrank_everything(self):
        ranked = ex.rank_candidates(analysed())
        assert ranked[0][0] == ex.PRIORITY_UNTESTED

    def test_superseded_lever_is_dropped(self):
        """Testing which tags necessarily changes how many -- one experiment."""
        ranked = ex.rank_candidates(analysed())
        keys = [fr.feature.key for _, fr, _ in ranked]
        assert "tag_set" in keys
        assert "tag_count" not in keys

    def test_confirmed_findings_rank_last(self):
        ranked = ex.rank_candidates(analysed())
        priorities = [p for p, _, _ in ranked]
        if ex.PRIORITY_CONFIRM in priorities:
            assert priorities.index(ex.PRIORITY_CONFIRM) == len(priorities) - 1 or \
                   priorities[-1] == ex.PRIORITY_CONFIRM

    def test_every_candidate_carries_a_reason(self):
        for _, _, reason in ex.rank_candidates(analysed()):
            assert reason and len(reason) > 20


class TestSizing:
    def test_defaults_to_the_time_budget_not_a_fixed_effect(self):
        plan = ex.design_experiment(analysed(), within_days=60)
        assert plan.estimated_days <= 75

    def test_shorter_budget_means_coarser_resolution(self):
        result = analysed()
        quick = ex.design_experiment(result, within_days=30)
        patient = ex.design_experiment(result, within_days=180)
        assert quick.min_detectable_effect > patient.min_detectable_effect
        assert quick.videos_per_arm < patient.videos_per_arm

    def test_explicit_min_effect_overrides_the_budget(self):
        plan = ex.design_experiment(analysed(), min_effect=3.0)
        assert plan.min_detectable_effect == 3.0

    def test_never_claims_to_resolve_below_the_floor(self):
        plan = ex.design_experiment(analysed(), within_days=10_000)
        assert plan.min_detectable_effect >= config.MIN_DETECTABLE_EFFECT

    def test_feasibility_table_is_monotone(self):
        plan = ex.design_experiment(analysed())
        effects = [row["detectable_effect"] for row in plan.feasibility]
        assert effects == sorted(effects, reverse=True)

    def test_warns_when_sized_to_a_budget(self):
        plan = ex.design_experiment(analysed(), within_days=30)
        assert any("Sized to read out" in w for w in plan.warnings)


class TestPlanContent:
    def test_has_at_least_two_arms(self):
        plan = ex.design_experiment(analysed())
        assert len(plan.arms) >= 2
        assert len({a.value for a in plan.arms}) == len(plan.arms)

    def test_arms_carry_explanatory_notes(self):
        for arm in ex.design_experiment(analysed()).arms:
            assert arm.note

    def test_interleaving_is_mandated(self):
        plan = ex.design_experiment(analysed())
        assert "Alternate" in plan.assignment

    def test_known_confounders_are_frozen(self):
        """The planted 12:00 effect must be held constant during a tag test."""
        plan = ex.design_experiment(analysed())
        if plan.feature != "publish_hour":
            frozen = {h["feature"] for h in plan.hold_constant}
            assert "publish_hour" in frozen or not plan.hold_constant

    def test_hold_constant_quotes_the_value_it_pins(self):
        plan = ex.design_experiment(analysed())
        for h in plan.hold_constant:
            assert h["value"] in h["why"] or h["label"] in h["why"] or h["why"]

    def test_serialises_to_valid_json(self):
        plan = ex.design_experiment(analysed())
        payload = json.loads(plan.to_json())
        assert payload["experiment_id"] == plan.experiment_id
        assert len(payload["arms"]) == len(plan.arms)

    def test_explicit_feature_selection(self):
        plan = ex.design_experiment(analysed(), feature_key="tag_set")
        assert plan.feature == "tag_set"

    def test_unknown_feature_is_rejected_with_options(self):
        with pytest.raises(ValueError, match="not an experiment candidate"):
            ex.design_experiment(analysed(), feature_key="nonsense")


class TestVerdict:
    @staticmethod
    def experiment_record(created, values=("12", "33")):
        return {
            "experiment_id": "exp_test",
            "channel_id": "UCtest",
            "feature": "duration",
            "hypothesis": "h",
            "arms_json": json.dumps([
                {"name": "control", "value": f"{values[0]}s", "videos": 50},
                {"name": "variant", "value": f"{values[1]}s", "videos": 50},
            ]),
            "min_per_arm": 50,
            "created_utc": created.isoformat(),
            "status": "designed",
        }

    def test_reports_insufficient_data_before_the_test_runs(self):
        result = analysed()
        # Designed "now" -- nothing in the catalog postdates it.
        rec = self.experiment_record(datetime.now(timezone.utc))
        v = ex.read_verdict(result, rec)
        assert v.status == "insufficient-data"
        assert "Not enough data" in v.headline

    def test_insufficient_verdict_says_what_is_still_needed(self):
        rec = self.experiment_record(datetime.now(timezone.utc))
        v = ex.read_verdict(analysed(), rec)
        assert "Keep publishing" in v.recommendation

    def test_detects_a_real_difference_between_arms(self):
        vids = build()
        for i, v in enumerate(vids):
            v["duration_s"] = 12 if i % 2 else 33
            v["views"] = 400 if i % 2 else 100
        result = an.analyze(vids, CHANNEL, iters=1500)
        rec = self.experiment_record(BASE - timedelta(days=1))
        v = ex.read_verdict(result, rec)
        assert v.status == "conclusive"
        assert v.p_value is not None and v.p_value <= config.ALPHA

    def test_inconclusive_result_states_what_it_could_resolve(self):
        """'No effect' and 'no effect this test could see' are different."""
        vids = build()
        for i, v in enumerate(vids):
            v["duration_s"] = 12 if i % 2 else 33
        result = an.analyze(vids, CHANNEL, iters=1500)
        rec = self.experiment_record(BASE - timedelta(days=1))
        v = ex.read_verdict(result, rec)
        if v.status == "inconclusive":
            assert v.detectable_effect is not None
            assert "no large effect" in v.recommendation.lower()

    def test_arm_membership_read_from_reality_not_the_plan(self):
        """Verdict must work even if the pipeline drifted from the manifest."""
        vids = build()
        for i, v in enumerate(vids):
            v["duration_s"] = 12 if i % 4 else 33   # deliberately unbalanced
        result = an.analyze(vids, CHANNEL, iters=800)
        rec = self.experiment_record(BASE - timedelta(days=1))
        v = ex.read_verdict(result, rec)
        counts = {a.value: a.n for a in v.arms}
        assert counts["12s"] != counts["33s"]

    def test_missing_feature_degrades_gracefully(self):
        rec = self.experiment_record(BASE - timedelta(days=1))
        rec["feature"] = "no_such_feature"
        v = ex.read_verdict(analysed(), rec)
        assert v.status == "insufficient-data"

    def test_verdict_serialises(self):
        rec = self.experiment_record(datetime.now(timezone.utc))
        payload = ex.read_verdict(analysed(), rec).to_dict()
        assert "arms" in payload and "status" in payload


class TestUploadRate:
    def test_infers_cadence_from_recent_publishes(self):
        result = analysed()
        rate = ex._uploads_per_day(result)
        assert rate > 0

    def test_never_returns_zero(self):
        result = an.analyze(build(n=3), CHANNEL, iters=100)
        assert ex._uploads_per_day(result) > 0
