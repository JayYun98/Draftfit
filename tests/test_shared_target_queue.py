"""Actual CLI queue training, partial-consumption replay and export reload."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import torch

from tests.test_joint_online_cpu import JointOnlineCPU, online_fixture


class SharedTargetQueueCPU(JointOnlineCPU):
    def test_epoch_boundary_keeps_inflight_credit(self):
        from unittest.mock import patch

        from draftfit.family import build_family_plan
        from draftfit.training.feature_queue import tensor_bytes
        from draftfit.training.joint_family import _backward_window, run_joint_family
        from draftfit.training.joint_online import JointOnlineSource

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            online_fixture(root, False)
            rows = (root / "train.jsonl").read_text().splitlines()[:3]
            (root / "train.jsonl").write_text("\n".join(rows) + "\n")
            base = json.loads((root / "base.json").read_text())
            base["training"].update(
                max_steps=3,
                total_steps=3,
                num_epochs=2,
                joint_feature_queue_bytes=4096,
                joint_feature_queue_batches=2,
            )
            (root / "base.json").write_text(json.dumps(base))
            _, stages = build_family_plan(str(root / "family.json"))
            for stage in stages:
                path = Path(stage.config.model.draft_model_config)
                path.parent.mkdir(parents=True, exist_ok=True)
                stage.draft_config.to_json_file(str(path))
            source = JointOnlineSource(stages[0], True)

            def checked(*args):
                live = tensor_bytes([b.tensors for b in args[-1]])
                self.assertLessEqual(live, source.queue_metrics()["resident_bytes"])
                self.assertLessEqual(source.queue_metrics()["resident_bytes"], 4096)
                return _backward_window(*args)

            with (
                patch(
                    "draftfit.training.joint_online.JointOnlineSource",
                    return_value=source,
                ),
                patch(
                    "draftfit.training.joint_family._backward_window",
                    side_effect=checked,
                ),
            ):
                run_joint_family(root / "outputs", stages)
            self.assertEqual(source.queue_metrics()["consumed"], 6)
            self.assertEqual(source.queue_metrics()["cancelled"], 0)
            self.assertEqual(source.queue_metrics()["resident_bytes"], 0)

    def test_control_stops_and_preserves_resume_patience(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            online_fixture(root, False)
            budget = root / "budget.json"
            budget.write_text(json.dumps(dict(spent_usd=0, timestamp_unix=time.time())))
            base = json.loads((root / "base.json").read_text())
            base["training"].update(
                max_steps=6,
                total_steps=6,
                num_epochs=3,
                joint_feature_queue_bytes=32768,
                joint_feature_queue_batches=4,
                joint_verification=dict(
                    interval=5, max_prompts=2, max_new_tokens=8, draft_tokens=2
                ),
                joint_control=dict(
                    budget_path=str(budget),
                    allow_reference_timings=True,
                    policy=dict(
                        schema="draftfit-joint-policy-v1",
                        revision="smoke",
                        domains=["unlabeled"],
                        patience=1,
                        min_delta=100,
                        min_requests=2,
                        next_steps=1,
                        next_cost_usd=0.01,
                        require_serving=True,
                        limits=dict(
                            max_usd=1,
                            reserve_usd=0.2,
                            max_steps=6,
                            max_pilot_usd=0,
                            max_pilot_steps=0,
                            max_exploration_usd=0,
                            max_pilots=0,
                            max_unvalidated_steps=0,
                        ),
                    ),
                ),
            )
            (root / "base.json").write_text(json.dumps(base))
            self.cli(root, "family", "--manifest", "family.json", "--joint")
            outcome = json.loads((root / "outputs/results.json").read_text())
            self.assertTrue(
                all(
                    r["global_step"] == 2 and r["reason"] == "policy_boundary"
                    for r in outcome
                )
            )
            saved = json.loads((root / "outputs/control-step2.json").read_text())
            self.assertTrue(saved["stop"])
            self.assertTrue(
                all(d["reason"] == "loss_plateau" for d in saved["decisions"].values())
            )
            manifest = json.loads((root / "family.json").read_text())
            manifest["output_dir"] = "resumed"
            (root / "resumed.json").write_text(json.dumps(manifest))
            self.cli(
                root,
                "family",
                "--manifest",
                "resumed.json",
                "--joint",
                "--resume-from",
                str(root / "outputs/joint-step2.pt"),
            )
            follow = json.loads((root / "resumed/control-step3.json").read_text())
            self.assertTrue(all(len(rows) == 3 for rows in follow["history"].values()))

    def test_failed_backward_releases_queue_without_acknowledging(self):
        from unittest.mock import patch

        from draftfit.family import build_family_plan
        from draftfit.training.joint_family import run_joint_family
        from draftfit.training.joint_online import JointOnlineSource

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            online_fixture(root, False)
            base = json.loads((root / "base.json").read_text())
            base["training"].update(
                joint_feature_queue_bytes=32768, joint_feature_queue_batches=4
            )
            (root / "base.json").write_text(json.dumps(base))
            _, stages = build_family_plan(str(root / "family.json"))
            for stage in stages:
                path = Path(stage.config.model.draft_model_config)
                path.parent.mkdir(parents=True, exist_ok=True)
                stage.draft_config.to_json_file(str(path))
            source = JointOnlineSource(stages[0], True)
            with (
                patch(
                    "draftfit.training.joint_online.JointOnlineSource",
                    return_value=source,
                ),
                patch(
                    "draftfit.training.joint_family._backward_window",
                    side_effect=RuntimeError("backward failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "backward failure"):
                    run_joint_family(root / "outputs", stages)
            self.assertEqual(source.queue_metrics()["resident_bytes"], 0)
            self.assertEqual(source.queue_metrics()["consumed"], 0)
            self.assertEqual(source.consumed_samples, 0)

    def test_scratch_all_families(self):
        # Parent's original synchronous gate remains in its own test module.
        self.exercise_queue()

    def test_existing_all_families(self):
        self.exercise_queue(seeded=True)

    def test_gemma_all_families(self):
        self.exercise_queue(gemma=True)

    def exercise_queue(self, seeded=False, gemma=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            online_fixture(root, seeded, gemma)
            base = json.loads((root / "base.json").read_text())
            base["training"].update(
                joint_feature_queue_bytes=32768, joint_feature_queue_batches=4
            )
            (root / "base.json").write_text(json.dumps(base))
            self.cli(root, "family", "--manifest", "family.json", "--joint")
            full = torch.load(root / "outputs/joint-step2.pt", weights_only=True)
            self.assertEqual(full["consumed_samples"], 4)
            curves = [
                json.loads(x)
                for x in (root / "outputs/learning-curves.jsonl")
                .read_text()
                .splitlines()
            ]
            closed = [r for r in curves if r["kind"] == "queue_closed"][-1]
            self.assertEqual(closed["resident_bytes"], 0)
            self.assertLessEqual(closed["peak_bytes"], 32768)
            self.assertEqual(closed["consumed"], 4)
            # Stop with prefetched but not consumed samples, then replay from
            # optimizer/shuffle cursor. Producer progress must not skip data.
            manifest = json.loads((root / "family.json").read_text())
            manifest["output_dir"] = "partial"
            (root / "partial.json").write_text(json.dumps(manifest))
            self.cli(
                root,
                "family",
                "--manifest",
                "partial.json",
                "--joint",
                "--stop-at-step",
                "1",
            )
            manifest["output_dir"] = "resumed"
            (root / "resumed.json").write_text(json.dumps(manifest))
            self.cli(
                root,
                "family",
                "--manifest",
                "resumed.json",
                "--joint",
                "--resume-from",
                str(root / "partial/joint-step1.pt"),
            )
            resumed = torch.load(root / "resumed/joint-step2.pt", weights_only=True)
            for arm in full["drafts"]:
                for key, tensor in full["drafts"][arm].items():
                    self.assertTrue(
                        torch.equal(tensor, resumed["drafts"][arm][key]), f"{arm}:{key}"
                    )
            from draftfit.export.to_hf import export_to_hf
            from draftfit.modeling.auto import AutoDraftModel

            for arm in full["drafts"]:
                out = root / "exports" / arm
                export_to_hf(
                    str(root / "outputs" / arm / f"{arm}-latest"),
                    str(root / "outputs" / arm / "config.json"),
                    str(out),
                    embedding_source=str(root / "target"),
                )
                loaded = AutoDraftModel.from_pretrained(str(out))
                for key, tensor in full["drafts"][arm].items():
                    self.assertTrue(torch.equal(tensor, loaded.state_dict()[key]))


@unittest.skipUnless(
    torch.cuda.is_available() and os.environ.get("DRAFTFIT_RUN_QUEUE_CUDA") == "1",
    "explicit CUDA queue gate",
)
class SharedTargetQueueCUDA(SharedTargetQueueCPU):
    execution_device = "cuda"
