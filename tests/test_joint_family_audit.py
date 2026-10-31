"""Regressions from the joint-family contract audit."""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from draftfit.family import FamilyStage
from draftfit.training.joint_family import run_joint_family, validate_joint_family
from tests import test_joint_family_core as core


class JointAuditTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.env = mock.patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        torch.set_num_threads(self.threads)

    def test_joint_accumulation_weights_each_arms_denominators(self):
        from draftfit.optimizer import BF16Optimizer
        from draftfit.training.strategies.base import DFlashTrainStrategy, StepOutput

        with tempfile.TemporaryDirectory() as directory:
            output, stages = core.JointFamilyCoreTest().stages(Path(directory))
            first = stages[1]
            cfg = first.config.model_copy(deep=True)
            cfg.output_dir = str(output / "second")
            cfg.run_id = "second"
            stages = [
                first,
                FamilyStage("second", cfg, copy.deepcopy(first.draft_config), None),
            ]
            counters = {}

            def forward(strategy, batch, ctx=None):
                param = strategy.dflash_model.draft_model.layers[
                    0
                ].self_attn.q_proj.weight
                i = counters.get(id(strategy), 0)
                if strategy.dflash_model.training:
                    counters[id(strategy)] = i + 1
                # Same mean-micro loss has gradients 1 and 3; correct window
                # normalization is (1*1 + 3*3)/(1+3) = 2.5, NOT (1+3)/2.
                den = param.new_tensor(1.0 if i % 2 == 0 else 3.0, dtype=torch.float32)
                num = param[0, 0].float() * den * den
                return StepOutput(
                    num / den,
                    {"accuracy": param.new_tensor(0.0)},
                    loss_terms=(num, den),
                )

            observed = []
            step = BF16Optimizer.step

            def capture(opt):
                observed.append(
                    float(opt.model[0].layers[0].self_attn.q_proj.weight.grad[0, 0])
                )
                return step(opt)

            with (
                mock.patch.object(DFlashTrainStrategy, "forward_loss", forward),
                mock.patch.object(BF16Optimizer, "step", capture),
            ):
                run_joint_family(output, stages)
            self.assertEqual(observed, [2.5, 2.5])

    def test_preflight_rejects_heldout_symlink_to_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, stages = core.JointFamilyCoreTest().stages(root)
            (root / "alias").symlink_to(root / "train", target_is_directory=True)
            for stage in stages:
                stage.config.data.eval_hidden_states_path = str(root / "alias")
            with self.assertRaisesRegex(ValueError, "overlap|differ"):
                validate_joint_family(stages)

    def test_fingerprint_tracks_changed_frozen_target(self):
        from draftfit.training.joint_family import _semantic_fingerprint

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, stages = core.JointFamilyCoreTest().stages(root)
            before = _semantic_fingerprint(stages)
            target = root / "target" / "model.safetensors"
            stat = target.stat()
            os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
            self.assertNotEqual(before, _semantic_fingerprint(stages))

    def test_shared_and_private_gradients_keep_each_objective_normalization(self):
        from types import SimpleNamespace

        from draftfit.training.joint_family import _backward_window
        from draftfit.training.strategies.base import StepOutput

        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                shared = torch.nn.Parameter(torch.tensor(1.0, dtype=dtype))
                models = [
                    torch.nn.ParameterDict(
                        {
                            "shared": shared,
                            "private": torch.nn.Parameter(
                                torch.tensor(1.0, dtype=dtype)
                            ),
                        }
                    )
                    for _ in range(2)
                ]

                class Strategy:
                    def __init__(self, model, terms):
                        self.model, self.terms = model, terms

                    def forward_loss(self, batch):
                        mean = (self.model["shared"] + self.model["private"]) * batch
                        den = mean.new_tensor(batch)
                        return StepOutput(
                            mean,
                            {},
                            loss_terms=(mean * den, den) if self.terms else None,
                        )

                strategies = [Strategy(models[0], False), Strategy(models[1], True)]
                metrics = _backward_window(
                    [SimpleNamespace(name="s"), SimpleNamespace(name="d")],
                    strategies,
                    [SimpleNamespace(draft_model=m) for m in models],
                    [1.0, 3.0],
                )
                self.assertEqual(float(shared.grad), 2.25)
                self.assertEqual(float(models[0]["private"].grad), 1.0)
                self.assertEqual(float(models[1]["private"].grad), 1.25)
                self.assertEqual(metrics["s"][0], 4.0)
                self.assertEqual(metrics["d"][0], 5.0)

    def test_accumulation_crosses_epoch_and_resume_seeks_without_rereads(self):
        from draftfit.runtime.data_plane import feature_store

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, stages = core.JointFamilyCoreTest().stages(root)
            (root / "train" / "03.ckpt").unlink()
            for stage in stages:
                stage.config.training.num_epochs = 2
                stage.config.training.max_steps = 3
                stage.config.training.total_steps = 3
            reads = []
            original = feature_store.load_feature_file

            def read(path):
                reads.append(Path(path))
                return original(path)

            with mock.patch.object(
                feature_store, "load_feature_file", side_effect=read
            ):
                run_joint_family(output, stages)
            training_reads = [path for path in reads if path.parent == root / "train"]
            self.assertEqual(len(training_reads), 6)
            saved = torch.load(output / "joint-step2.pt", weights_only=True)
            self.assertEqual((saved["epoch"], saved["next_batch"]), (1, 1))
            resumed = []
            for stage in stages:
                cfg = stage.config.model_copy(deep=True)
                cfg.output_dir = str(root / "resume" / stage.name)
                resumed.append(FamilyStage(stage.name, cfg, stage.draft_config, None))
            reads.clear()
            with mock.patch.object(
                feature_store, "load_feature_file", side_effect=read
            ):
                run_joint_family(
                    root / "resume", resumed, resume_from=output / "joint-step2.pt"
                )
            # Resume evaluates first, consumes only the remaining epoch-1
            # samples in the uninterrupted order, then evaluates at step 3.
            # Held-out reads must not hide a duplicate/skipped training read.
            heldout = sorted((root / "holdout").glob("*.ckpt"))
            self.assertEqual(reads, heldout + training_reads[-2:] + heldout)
            expected = torch.load(output / "joint-step3.pt", weights_only=True)
            actual = torch.load(root / "resume" / "joint-step3.pt", weights_only=True)
            for name, state in expected["drafts"].items():
                for key, value in state.items():
                    torch.testing.assert_close(
                        actual["drafts"][name][key], value, rtol=0, atol=0
                    )

    def test_overlapping_feature_files_are_rejected(self):
        from draftfit.training.joint_family import _semantic_fingerprint

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, stages = core.JointFamilyCoreTest().stages(root)
            os.link(root / "train" / "00.ckpt", root / "holdout" / "shared.ckpt")
            with self.assertRaisesRegex(ValueError, "overlap"):
                _semantic_fingerprint(stages)

    def test_capture_metadata_changes_invalidate_resume_fingerprint(self):
        from draftfit.inference.capture_manifest import (
            build_capture_manifest,
            write_capture_manifest,
        )
        from draftfit.training.joint_family import _semantic_fingerprint

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, stages = core.JointFamilyCoreTest().stages(root)
            before = _semantic_fingerprint(stages)
            changed = build_capture_manifest(
                strategy="dspark",
                capture_method="target_hidden",
                capture_layers=(1, 5),
                feature_names=(
                    "input_ids",
                    "loss_mask",
                    "hidden_states",
                    "target_last_hidden_states",
                ),
                target_model=str(root / "target"),
                target_revision=stages[0].config.model.target_revision,
                target_feature="target_last_hidden_states",
                target_repr="hidden_state",
                tokenizer_version="changed-tokenizer",
            )
            write_capture_manifest(
                str(root / "train" / "capture_manifest.json"), changed
            )
            self.assertNotEqual(before, _semantic_fingerprint(stages))

    def test_long_bf16_window_does_not_lose_small_gradients(self):
        from types import SimpleNamespace

        from draftfit.training.joint_family import _backward_window
        from draftfit.training.strategies.base import StepOutput

        model = torch.nn.ParameterDict(
            {"weight": torch.nn.Parameter(torch.tensor(1.0, dtype=torch.bfloat16))}
        )

        class Strategy:
            def forward_loss(self, batch):
                loss = model["weight"].float() * batch
                return StepOutput(loss, {}, loss_terms=(loss, loss.new_tensor(1.0)))

        _backward_window(
            [SimpleNamespace(name="a")],
            [Strategy()],
            [SimpleNamespace(draft_model=model)],
            [1.0] * 512,
        )
        self.assertEqual(float(model["weight"].grad), 1.0)

    def test_resolved_target_revision_is_part_of_joint_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, stages = core.JointFamilyCoreTest().stages(root)
            run_joint_family(output, stages)
            path = output / "joint-step1.pt"
            state = torch.load(path, weights_only=True)
            self.assertEqual(
                state["target_revision"], stages[0].config.model.target_revision
            )
            state["target_revision"] = "different-resolved-target"
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "resolved target revision"):
                run_joint_family(root / "resume", stages, resume_from=path)


if __name__ == "__main__":
    unittest.main()
