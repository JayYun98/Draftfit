"""Execute resource decisions with synthetic observations; no GPU/speed claims."""
from copy import deepcopy
import json
import unittest

from draftfit.training.resource_policy import decide


def preset(name, batch, lr=1e-4, **extra):
    return dict(id=name, training=dict(batch_size=batch, accumulation_steps=2,
                                      learning_rate=lr), preflight_passed=True,
                pilot_steps=10, pilot_cost_usd=.05, pilot_seconds=30., **extra)


def config(**extra):
    return dict(presets=[preset("small", 1, 5e-5), preset("base", 2),
                         preset("large", 4, 2e-4), preset("largest", 8)], **extra)


def current(step=12, **extra):
    return dict(phase_id="phase-0", preset="base", step=step, **extra)


def admission(now=None, **extra):
    now = now or current()
    return dict(allowed=True, phase_id=now["phase_id"], step=now["step"],
                preset_ids=["small", "base", "large", "largest"],
                max_steps=10, max_cost_usd=.05, max_seconds=30.) | extra


def rows(*, steps=(10, 11, 12), phase="phase-0", name="base", **metrics):
    return [dict(phase_id=phase, preset=name, step=step, peak_vram_bytes=50.,
                 capacity_bytes=100., utilization=.5, throughput=20., loss=1.) | metrics
            for step in steps]


class ResourcePolicyTests(unittest.TestCase):
    def run_policy(self, observations=None, *, cfg=None, now=None, admit=None, state=None):
        return decide(config() if cfg is None else cfg,
                      rows() if observations is None else observations,
                      current=current() if now is None else now,
                      admission=admission(now) if admit is None else admit,
                      state=dict(transitions=0, last_transition_step=None) if state is None else state)

    def assert_measure_only(self, result, reason):
        self.assertEqual((result["action"], result["reason"]), ("measure_only", reason))
        self.assertIsNone(result["selected_preset"])
        self.assertIsNone(result["transition_state"])
        self.assertEqual((result["next_steps"], result["next_cost_usd"], result["next_seconds"]), (0, 0, 0))
        self.assertEqual(result["optimizer_mode"], "preserve")
        self.assertFalse(result["optimizer_migration_allowed"])

    def test_underutilization_selects_nearest_registered_weights_only_phase(self):
        result = self.run_policy()
        self.assertEqual((result["action"], result["next_preset"]), ("new_phase", "large"))
        self.assertEqual(result["selected_preset"], config()["presets"][2])
        self.assertEqual(result["selected_preset"]["training"]["learning_rate"], 2e-4)
        self.assertEqual(result["optimizer_mode"], "weights_only")
        self.assertTrue(result["new_phase_required"])
        self.assertFalse(result["optimizer_migration_allowed"])
        self.assertFalse(result["executed"])
        self.assertEqual(result["transition_state"], dict(transitions=1, last_transition_step=12))

    def test_pressure_selects_smaller_registered_batch_and_lr(self):
        result = self.run_policy(rows(peak_vram_bytes=95.))
        self.assertEqual((result["reason"], result["next_preset"]), ("memory_pressure", "small"))
        self.assertEqual(result["selected_preset"]["training"]["learning_rate"], 5e-5)

    def test_dead_band_and_single_pressure_spike_hold(self):
        for peaks in ((70, 70, 70), (50, 50, 95), (95, 89, 95), (50, 66, 50)):
            with self.subTest(peaks=peaks):
                data = rows()
                for row, peak in zip(data, peaks):
                    row["peak_vram_bytes"] = peak
                self.assert_measure_only(self.run_policy(data), "hysteresis_hold")

    def test_busy_gpu_does_not_grow_batch(self):
        self.assert_measure_only(self.run_policy(rows(utilization=.95)), "hysteresis_hold")

    def test_loss_or_throughput_regression_blocks_exploratory_growth(self):
        for metric, value in (("loss", 1.2), ("throughput", 15.)):
            with self.subTest(metric=metric):
                data = rows()
                data[1][metric] = value
                self.assert_measure_only(self.run_policy(data), "unstable_loss_or_throughput")

    def test_incomplete_and_duplicate_windows_do_not_trigger(self):
        for data in ([], rows()[:2], [rows()[0]] * 3):
            self.assert_measure_only(self.run_policy(data), "insufficient_window")

    def test_zero_utilization_peak_and_loss_are_valid_measurements(self):
        for metrics in (dict(utilization=0), dict(peak_vram_bytes=0), dict(loss=0),
                        dict(utilization=0, peak_vram_bytes=0, loss=0)):
            with self.subTest(metrics=metrics):
                result = self.run_policy(rows(**metrics))
                self.assertEqual((result["action"], result["next_preset"]), ("new_phase", "large"))
                self.assertEqual(result["rejected"], [])
                json.dumps(result, allow_nan=False)

    def test_zero_loss_baseline_has_no_division_and_rejects_later_growth(self):
        data = rows(loss=0)
        data[1]["loss"] = 1e-12
        self.assert_measure_only(self.run_policy(data), "unstable_loss_or_throughput")
        data = rows()
        data[-1]["loss"] = 0
        self.assertEqual(self.run_policy(data)["action"], "new_phase")

    def test_zero_throughput_is_valid_but_never_evidence_for_growth(self):
        for rates in ((0, 0, 0), (0, 20, 20), (20, 0, 20), (20, 20, 0)):
            with self.subTest(rates=rates):
                data = rows()
                for row, rate in zip(data, rates):
                    row["throughput"] = rate
                result = self.run_policy(data)
                self.assert_measure_only(result, "zero_throughput")
                self.assertEqual(result["rejected"], [])

    def test_memory_pressure_can_reduce_batch_with_zero_throughput(self):
        result = self.run_policy(rows(peak_vram_bytes=95, utilization=0, throughput=0, loss=0))
        self.assertEqual((result["reason"], result["next_preset"]), ("memory_pressure", "small"))
        self.assertEqual(result["rejected"], [])

    def test_zero_capacity_is_invalid(self):
        self.assert_measure_only(self.run_policy(rows(capacity_bytes=0)), "invalid_observations")

    def test_invalid_metrics_rejected_and_result_is_strict_json(self):
        for metric in ("peak_vram_bytes", "capacity_bytes", "utilization", "throughput", "loss"):
            for value in (float("nan"), float("inf"), -1, True, "1"):
                with self.subTest(metric=metric, value=value):
                    data = rows()
                    data[-1][metric] = value
                    result = self.run_policy(data)
                    self.assert_measure_only(result, "invalid_observations")
                    self.assertTrue(result["rejected"])
                    json.dumps(result, allow_nan=False)

    def test_stale_future_nonpositive_and_foreign_observations_rejected(self):
        for change in (dict(step=1), dict(step=13), dict(step=0), dict(step=-1),
                       dict(step=True), dict(phase_id="old"), dict(preset="other"),
                       dict(utilization=50), dict(peak_vram_bytes=101)):
            with self.subTest(change=change):
                data = rows()
                data[-1].update(change)
                self.assert_measure_only(self.run_policy(data), "invalid_observations")

    def test_conflicting_duplicates_cannot_select_by_input_order(self):
        data = rows() + rows()[-1:]
        data[-1]["throughput"] = 40
        for ordered in (data, list(reversed(data))):
            self.assert_measure_only(self.run_policy(ordered), "invalid_observations")

    def test_capacity_changes_do_not_mix_hardware(self):
        data = rows()
        data[-1]["capacity_bytes"] = 200
        self.assert_measure_only(self.run_policy(data), "capacity_changed")

    def test_missing_stale_denied_or_invalid_admission_is_measurement_only(self):
        for admit in ({}, admission(allowed=False), admission(step=11),
                      admission(phase_id="old"), admission(max_cost_usd=float("nan")),
                      admission(max_steps=True), admission(max_seconds=-1),
                      admission(preset_ids="large")):
            with self.subTest(admit=admit):
                self.assert_measure_only(self.run_policy(admit=admit), "admission_required")

    def test_each_external_limit_applies_without_clipping_preset(self):
        for change in (dict(max_steps=9), dict(max_cost_usd=.049), dict(max_seconds=29),
                       dict(max_cost_usd=0), dict(preset_ids=["small"]), dict(preset_ids=[])):
            with self.subTest(change=change):
                self.assert_measure_only(self.run_policy(admit=admission(**change)), "no_admitted_candidate")
        result = self.run_policy()
        self.assertEqual((result["next_steps"], result["next_cost_usd"], result["next_seconds"]), (10, .05, 30))

    def test_only_preflighted_registered_candidates_are_selected(self):
        cfg = config()
        cfg["presets"][2]["preflight_passed"] = False
        self.assertEqual(self.run_policy(cfg=cfg)["next_preset"], "largest")
        cfg["presets"] = cfg["presets"][:2]
        self.assert_measure_only(self.run_policy(cfg=cfg), "no_admitted_candidate")
        self.assert_measure_only(self.run_policy(cfg={"presets": []}), "current_preset_not_registered")

    def test_oom_bypasses_window_and_cooldown_but_only_moves_down(self):
        result = self.run_policy([], now=current(oom=True),
                                 state=dict(transitions=1, last_transition_step=11))
        self.assertEqual((result["reason"], result["next_preset"]), ("oom", "small"))
        now = current(oom=True)
        now["preset"] = "small"
        self.assert_measure_only(self.run_policy([], now=now), "no_admitted_candidate")

    def test_oom_cannot_bypass_limits_or_admission(self):
        now = current(oom=True)
        self.assert_measure_only(self.run_policy([], now=now,
                                 state=dict(transitions=3, last_transition_step=11)), "transition_limit")
        self.assert_measure_only(self.run_policy([], now=now, admit={}), "admission_required")
        self.assert_measure_only(self.run_policy([], now=now, admit=admission(max_seconds=0)), "no_admitted_candidate")

    def test_transition_limit_zero_disables_tuning(self):
        self.assert_measure_only(self.run_policy(cfg=config(max_transitions=0)), "transition_limit")

    def test_json_restart_preserves_cooldown_and_transition_count(self):
        first = self.run_policy(cfg=config(max_transitions=2))
        state = json.loads(json.dumps(first["transition_state"]))
        now = dict(phase_id="phase-1", preset="large", step=15)
        recent = rows(steps=(13, 14, 15), phase="phase-1", name="large", peak_vram_bytes=95)
        self.assert_measure_only(self.run_policy(recent, now=now, state=state), "cooldown")
        # Exactly the cooldown boundary admits a new phase, using fresh metrics.
        now["step"] = 32
        recent = rows(steps=(30, 31, 32), phase="phase-1", name="large", peak_vram_bytes=95)
        second = self.run_policy(recent, cfg=config(max_transitions=2), now=now, state=state)
        self.assertEqual(second["next_preset"], "base")
        self.assertEqual(second["transition_state"]["transitions"], 2)
        restored = json.loads(json.dumps(second["transition_state"]))
        now = dict(phase_id="phase-2", preset="base", step=60)
        self.assert_measure_only(self.run_policy(rows(steps=(58, 59, 60), phase="phase-2"),
                                 cfg=config(max_transitions=2), now=now, state=restored), "transition_limit")

    def test_previous_phase_observations_cannot_survive_restart(self):
        state = dict(transitions=1, last_transition_step=11)
        self.assert_measure_only(self.run_policy(state=state), "invalid_observations")

    def test_inconsistent_or_future_state_fails_closed(self):
        for state in (dict(transitions=0, last_transition_step=1),
                      dict(transitions=1, last_transition_step=None),
                      dict(transitions=1, last_transition_step=13),
                      dict(transitions=-1, last_transition_step=None)):
            with self.subTest(state=state), self.assertRaises(ValueError):
                self.run_policy(state=state)

    def test_deterministic_pure_json_replay_and_detached_result(self):
        cfg, data, now, admit, state = config(), rows(), current(), admission(), dict(transitions=0, last_transition_step=None)
        before = deepcopy((cfg, data, now, admit, state))
        first = decide(cfg, data, current=now, admission=admit, state=state)
        replay = self.run_policy(list(reversed(data)) + data[:1])
        self.assertEqual(first, replay)
        self.assertEqual(first, json.loads(json.dumps(first, allow_nan=False)))
        self.assertEqual((cfg, data, now, admit, state), before)
        first["selected_preset"]["training"]["learning_rate"] = 999
        self.assertEqual(cfg, before[0])

    def test_rejects_exact_resume_presets_and_invalid_configuration(self):
        for change in (dict(optimizer_mode="exact_resume"), dict(pilot_cost_usd=float("inf")),
                       dict(pilot_steps=0), dict(pilot_seconds=0)):
            cfg = config()
            cfg["presets"][2].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_policy(cfg=cfg)
        for change in (dict(window_size=1), dict(low_memory_ratio=.95),
                       dict(max_transitions=-1), dict(max_loss_growth=float("nan"))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_policy(cfg=config(**change))


if __name__ == "__main__":
    unittest.main()
