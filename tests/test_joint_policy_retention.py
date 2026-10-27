"""Real CPU joint updates with synthetic policy rankings; no serving claims."""

import hashlib
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import torch

from draftfit.optimizer import BF16Optimizer
from draftfit.training import joint_family
from tests.test_joint_deadline import stages_for
from tests.test_joint_online_cpu import online_fixture
from tests.test_joint_policy import policy


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class JointPolicyRetentionTest(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        env = mock.patch.dict(
            os.environ, {"SPECFORGE_DEVICE": "cpu", "HF_HUB_OFFLINE": "1"}
        )
        env.start()
        self.addCleanup(env.stop)
        directory = tempfile.TemporaryDirectory(prefix="joint-policy-retention-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        online_fixture(self.root, False)
        budget = self.root / "budget.json"
        budget.write_text(json.dumps(dict(timestamp_unix=time.time(), spent_usd=0)))
        path = self.root / "base.json"
        base = json.loads(path.read_text())
        base["training"].update(
            max_checkpoints=1,
            num_epochs=6,
            max_steps=11,
            total_steps=11,
            joint_verification=dict(interval=1, max_prompts=1, max_new_tokens=10),
            joint_control=dict(
                budget_path=str(budget),
                allow_reference_timings=True,
                policy=policy(
                    domains=["general"], patience=100, min_requests=1, next_steps=1
                ),
            ),
        )
        path.write_text(json.dumps(base))
        self.output = self.root / "run"

    def generation(self, stages, bundles, source, optimizer, step, settings):
        # Each arm and each metric selects a different artifact: steps 1..9.
        # Step 10 is deliberately not best; it should expire unless resumed.
        tokens = list(range(10))
        baseline = dict(
            trial=0, row_index=0, token_ids=tokens, generated_tokens=10, seconds=100.0
        )
        candidates = []
        for index, stage in enumerate(stages):
            accepted = 9 if step == 3 * index + 2 else 1
            seconds = 1.0 if step == 3 * index + 3 else 10.0 + step
            row = dict(
                baseline, accepted_tokens=accepted, proposed_tokens=10, seconds=seconds
            )
            candidates.append(
                dict(
                    name=stage.name,
                    tokenparity=True,
                    runs=[row],
                    accepted_tokens=accepted,
                    proposed_tokens=10,
                    acceptance_rate=accepted / 10,
                    tok_s=10 / seconds,
                )
            )
        return dict(
            schema="draftfit-joint-generation-v1",
            step=step,
            production_engine_speed=False,
            timing_scope="synthetic-retention-test",
            domains=["general"],
            tokenparity=True,
            total_seconds=1.0,
            bindings=dict(config={"max_prompts": 1}, prompts=[{"domain": "general"}]),
            baseline=dict(runs=[baseline], tok_s=0.1),
            candidates=candidates,
        )

    def assert_best_exists(self, control):
        selected = set()
        for decision in control["decisions"].values():
            for metric in ("loss", "acceptance", "speed"):
                artifact = decision["best"][metric]["artifact"]
                path = self.output / f"joint-step{artifact['step']}.pt"
                self.assertTrue(
                    path.is_file(),
                    f"policy-selected {metric} best deleted: {path.name}",
                )
                self.assertEqual(sha256(path), artifact["artifact_sha256"])
                selected.add(artifact["step"])
        return selected

    def run_training(self, *, stop=11, resume=None, fail_eval=None):
        stages = stages_for(self.root, self.output)
        current = torch.load(resume, weights_only=True)["step"] if resume else 0
        original_step = BF16Optimizer.step
        original_eval = joint_family._evaluate

        def update(optimizer):
            nonlocal current
            result = original_step(optimizer)
            current += 1
            return result

        def evaluate(*args, **kwargs):
            # This runs after checkpoint/pruning, before the new policy result.
            previous = self.output / f"control-step{current - 1}.json"
            if previous.exists():
                self.assert_best_exists(json.loads(previous.read_text()))
            if current == fail_eval:
                raise RuntimeError("injected evaluation failure")
            metrics = original_eval(*args, **kwargs)
            for index, stage in enumerate(stages):
                metrics[stage.name]["eval/avg_loss"] = (
                    0.1 if current == 3 * index + 1 else 10.0 + current
                )
            return metrics

        with (
            mock.patch.object(BF16Optimizer, "step", update),
            mock.patch.object(joint_family, "_evaluate", side_effect=evaluate),
            mock.patch(
                "draftfit.training.joint_verification.evaluate_generation",
                side_effect=self.generation,
            ),
        ):
            return joint_family.run_joint_family(
                self.output, stages, resume_from=resume, stop_at_step=stop
            )

    def saved_steps(self):
        return {
            int(path.stem.removeprefix("joint-step"))
            for path in self.output.glob("joint-step*.pt")
        }

    def test_all_arm_metric_bests_survive_before_next_evaluation(self):
        self.run_training()
        control = json.loads((self.output / "control-step11.json").read_text())
        self.assertEqual(self.assert_best_exists(control), set(range(1, 10)))
        self.assertEqual(self.saved_steps(), set(range(1, 10)) | {11})

    def test_authenticated_control_resume_preserves_best_and_nonbest_source(self):
        self.run_training(stop=10)
        source = self.output / "joint-step10.pt"
        before = sha256(source)
        control = json.loads((self.output / "control-step10.json").read_text())
        self.assertEqual(control["artifact_sha256"], before)
        self.run_training(resume=source)
        self.assertEqual(sha256(source), before)
        self.assertEqual(self.saved_steps(), set(range(1, 12)))
        self.assertEqual(self.assert_best_exists(control), set(range(1, 10)))

    def test_eval_failure_resume_keeps_previous_policy_selection(self):
        with self.assertRaisesRegex(RuntimeError, "injected evaluation failure"):
            self.run_training(fail_eval=10)
        source = self.output / "joint-step10.pt"
        self.assertFalse((self.output / "control-step10.json").exists())
        control = json.loads((self.output / "control-step9.json").read_text())
        self.assertEqual(self.assert_best_exists(control), set(range(1, 10)))
        self.run_training(resume=source)
        self.assertEqual(self.saved_steps(), set(range(1, 12)))

    def test_mismatched_control_hash_rejected_before_pruning(self):
        self.run_training(stop=2)
        source = self.output / "joint-step2.pt"
        receipt = self.output / "control-step2.json"
        control = json.loads(receipt.read_text())
        control["artifact_sha256"] = "f" * 64
        receipt.write_text(json.dumps(control))
        before = {p.name: sha256(p) for p in self.output.glob("joint-step*.pt")}
        with self.assertRaisesRegex(
            ValueError, "control history checkpoint hash mismatch"
        ):
            self.run_training(resume=source, stop=3)
        self.assertEqual(
            before, {p.name: sha256(p) for p in self.output.glob("joint-step*.pt")}
        )


if __name__ == "__main__":
    unittest.main()
