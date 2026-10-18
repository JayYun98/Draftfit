"""Direct CPU checks for the joint trainer, independent of family CLI wiring."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from draftfit.config import Config
from draftfit.family import FamilyStage
from draftfit.training.joint_family import run_joint_family, validate_joint_family


class JointFamilyCoreTest(unittest.TestCase):
    def test_rank0_logger_without_process_group(self):
        from draftfit.utils import print_on_rank0

        with mock.patch("draftfit.utils.dist.is_initialized", return_value=False):
            with mock.patch("draftfit.utils.logger.info") as logged:
                print_on_rank0("local resume")
        logged.assert_called_once_with("local resume")

    def stages(self, root, *, seeded=False):
        from draftfit.training.model_loading import (
            convert_family_draft_config, load_draft_config_source,
        )
        from tests.test_family_joint_cpu import fixture

        fixture(root, seeded)
        base = json.loads((root / "base.json").read_text())
        source = load_draft_config_source(str(root / "draft.json"))
        output = root / "direct"
        stages = []
        for name, strategy in (("dspark", "dspark"), ("ce", "dflash"),
                               ("selector", "dflash2")):
            draft = convert_family_draft_config(source, strategy)
            directory = output / name
            directory.mkdir(parents=True)
            draft.to_json_file(str(directory / "draft.json"))
            raw = json.loads(json.dumps(base))
            raw["output_dir"] = str(directory)
            raw["run_id"] = name
            raw["training"]["strategy"] = strategy
            raw["model"]["draft_model_config"] = str(directory / "draft.json")
            stages.append(FamilyStage(name, Config.model_validate(raw), draft, None))
        return output, stages

    def test_real_joint_step_reads_each_feature_once_and_ties_weights(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(prefix="joint-core-") as directory:
                root = Path(directory)
                output, stages = self.stages(root)
                original = __import__("draftfit.runtime.data_plane.feature_store",
                                      fromlist=["load_feature_file"]).load_feature_file
                reads = []

                def counted(path):
                    reads.append(str(path))
                    return original(path)

                with mock.patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"}):
                    with mock.patch("draftfit.runtime.data_plane.feature_store.load_feature_file",
                                    side_effect=counted):
                        result = run_joint_family(output, stages)
                self.assertEqual([row["status"] for row in result["runs"]], ["complete"] * 3)
                # Four training reads; two held-out reads at step0, step1, step2.
                # Each pass still reads once regardless of the three arms.
                self.assertEqual(sum(Path(p).parent.name == "train" for p in reads), 4)
                self.assertEqual(sum(Path(p).parent.name == "holdout" for p in reads), 6)
                states = [torch.load(output / stage.name / f"{stage.name}-latest" /
                                     "training_state.pt", weights_only=True)["draft_state_dict"]
                          for stage in stages]
                key = "layers.0.self_attn.q_proj.weight"
                self.assertTrue(torch.equal(states[0][key], states[1][key]))
                self.assertTrue(torch.equal(states[0][key], states[2][key]))
                self.assertEqual(result["shared_backbone_parameters"] > 0, True)
                self.assertTrue((output / "joint-step1.pt").is_file())
                self.assertTrue((output / "joint-step2.pt").is_file())
        finally:
            torch.set_num_threads(old_threads)

    def test_preflight_rejects_conflicting_freeze_before_output(self):
        with tempfile.TemporaryDirectory(prefix="joint-core-") as directory:
            root = Path(directory)
            output, stages = self.stages(root)
            stages[0].config.training.dspark_head_only = True
            with self.assertRaisesRegex(ValueError, "head-only"):
                validate_joint_family(stages)
            self.assertFalse((output / "joint-step1.pt").exists())

    def test_preflight_rejects_different_rope_and_target_geometry(self):
        with tempfile.TemporaryDirectory(prefix="joint-geometry-") as directory:
            _, stages = self.stages(Path(directory))
            stages[1].draft_config.rope_parameters = {
                **stages[1].draft_config.rope_parameters, "rope_theta": 20000.0,
            }
            with self.assertRaisesRegex(ValueError, "rope_parameters"):
                validate_joint_family(stages)
            stages[1].draft_config.rope_parameters = stages[0].draft_config.rope_parameters
            stages[1].draft_config.target_hidden_size += 1
            with self.assertRaisesRegex(ValueError, "target_hidden_size"):
                validate_joint_family(stages)

    def test_build_ties_target_modules_before_next_arm(self):
        from draftfit.training.assembly import build_model_bundle

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(prefix="joint-target-") as directory:
                root = Path(directory)
                output, stages = self.stages(root)
                seen = []

                def observed(cfg, *, algorithm):
                    if seen:
                        first, prior = seen[0], seen[-1]
                        self.assertIs(first.model.lm_head, prior.model.lm_head)
                        self.assertIs(first.model.embed_tokens, prior.model.embed_tokens)
                        self.assertIs(first.target_head, prior.target_head)
                        self.assertIs(first.draft_model.layers[0].self_attn.q_proj.weight,
                                      prior.draft_model.layers[0].self_attn.q_proj.weight)
                    bundle = build_model_bundle(cfg, algorithm=algorithm)
                    seen.append(bundle)
                    return bundle

                with mock.patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"}):
                    with mock.patch("draftfit.training.assembly.build_model_bundle",
                                    side_effect=observed):
                        run_joint_family(output, stages)
                self.assertEqual(len(seen), 3)
                self.assertIs(seen[0].target_head, seen[-1].target_head)
        finally:
            torch.set_num_threads(old_threads)

    def test_real_family_gradients_average_on_one_shared_parameter(self):
        from torch import nn
        from draftfit.algorithms.builtin import builtin_algorithm_registry
        from draftfit.optimizer import BF16Optimizer
        from draftfit.training.assembly import build_model_bundle
        from draftfit.training.evaluation import Evaluator
        from draftfit.training.joint_family import (
            _evaluate, _make_loader, _restore_rng, _rng_state, _tie_backbones,
        )

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(prefix="joint-gradient-") as directory:
                root = Path(directory)
                _, stages = self.stages(root)
                registry = builtin_algorithm_registry()
                with mock.patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"}):
                    torch.manual_seed(7)
                    bundles = [build_model_bundle(stage.config,
                               algorithm=registry.resolve(stage.config.training.strategy))
                               for stage in stages]
                    _tie_backbones(bundles, scratch=True)
                    loader, _ = _make_loader(stages[0].config.data.hidden_states_path,
                                             stages[0], needs_dspark=True, epoch=0,
                                             shuffle=False, drop_last=True)
                    batch = next(iter(loader))
                    strategies = [registry.resolve(stage.config.training.strategy)
                                  .providers.step.build(bundle.model)
                                  for stage, bundle in zip(stages, bundles)]
                    short_path = root / "holdout" / "01.ckpt"
                    short = torch.load(short_path, weights_only=True)
                    short["loss_mask"] = torch.tensor([0, 1, 1, 0, 0, 0, 0, 0])
                    torch.save(short, short_path)
                    before_eval = _rng_state()
                    metrics = _evaluate(stages, strategies, bundles)
                    self.assertTrue(all("eval/avg_loss" in row for row in metrics.values()))
                    self.assertIn("eval/teacher_agreement", metrics["dspark"])
                    self.assertIn("eval/selector_loss", metrics["selector"])
                    self.assertTrue(torch.equal(before_eval["torch"], _rng_state()["torch"]))
                    self.assertEqual(before_eval["python"], _rng_state()["python"])
                    _restore_rng(before_eval)
                    reference_loader, _ = _make_loader(stages[0].config.data.eval_hidden_states_path,
                                                       stages[0], needs_dspark=True, epoch=0,
                                                       shuffle=False, drop_last=False)
                    bundles[0].model.eval()
                    reference = Evaluator().run(strategies[0].forward_loss, reference_loader)
                    bundles[0].model.train()
                    self.assertEqual(set(metrics["dspark"]), set(reference))
                    for name in reference:
                        self.assertAlmostEqual(metrics["dspark"][name], reference[name], places=6)
                    shared = bundles[0].draft_model.layers[0].self_attn.q_proj.weight
                    for bundle in bundles[1:]:
                        self.assertIs(shared,
                                      bundle.draft_model.layers[0].self_attn.q_proj.weight)
                    fixed_rng = _rng_state()
                    separate = []
                    for strategy in strategies:
                        _restore_rng(fixed_rng)
                        out = strategy.forward_loss(batch)
                        separate.append(torch.autograd.grad(out.loss, shared)[0].detach().clone())
                    shared.grad = None
                    for strategy in strategies:
                        _restore_rng(fixed_rng)
                        (strategy.forward_loss(batch).loss / len(strategies)).backward()
                    expected = torch.stack(separate).mean(dim=0)
                    torch.testing.assert_close(shared.grad, expected, rtol=2e-2, atol=2e-3)
                    self.assertTrue(all(grad.abs().sum() > 0 for grad in separate))
                    drafts = nn.ModuleList([bundle.draft_model for bundle in bundles])
                    optimizer = BF16Optimizer(drafts, lr=1e-4, total_steps=2)
                    self.assertEqual(len(optimizer.fp32_params),
                                     len({id(param) for param in drafts.parameters()}))
                    active = sum(param.grad is not None for param in optimizer.model_params)
                    optimizer.step()
                    self.assertEqual(len(optimizer.optimizer.state), active)
        finally:
            torch.set_num_threads(old_threads)

    def test_joint_checkpoint_retention(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(prefix="joint-retain-") as directory:
                root = Path(directory)
                output, stages = self.stages(root)
                for stage in stages:
                    stage.config.training.max_checkpoints = 1
                with mock.patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"}):
                    run_joint_family(output, stages)
                self.assertFalse((output / "joint-step1.pt").exists())
                self.assertTrue((output / "joint-step2.pt").is_file())
                for stage in stages:
                    self.assertTrue((output / stage.name / f"{stage.name}-latest" /
                                     "training_state.pt").is_file())
                resumed_root = root / "completed-resume"
                resumed = []
                for stage in stages:
                    target = resumed_root / stage.name
                    target.mkdir(parents=True)
                    stage.draft_config.to_json_file(str(target / "draft.json"))
                    payload = stage.config.model_dump(mode="json")
                    payload["output_dir"] = str(target)
                    payload["model"]["draft_model_config"] = str(target / "draft.json")
                    resumed.append(FamilyStage(stage.name, Config.model_validate(payload),
                                               stage.draft_config, None))
                with mock.patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"}):
                    run_joint_family(resumed_root, resumed,
                                     resume_from=output / "joint-step2.pt")
                self.assertTrue((resumed_root / "joint-step2.pt").is_file())
                for stage in resumed:
                    self.assertTrue((resumed_root / stage.name / f"{stage.name}-latest" /
                                     "training_state.pt").is_file())
                    self.assertIn("eval/avg_loss", (resumed_root / stage.name /
                                                    "metrics.jsonl").read_text())
                damaged = torch.load(output / "joint-step2.pt", weights_only=True)
                key = "layers.0.self_attn.q_proj.weight"
                damaged["drafts"]["selector"][key] = damaged["drafts"]["selector"][key] + 1
                bad_file = root / "damaged.pt"
                torch.save(damaged, bad_file)
                with mock.patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"}):
                    with self.assertRaisesRegex(ValueError, "unequal shared backbone"):
                        run_joint_family(resumed_root, resumed, resume_from=bad_file)
                original = torch.load(output / "joint-step2.pt", weights_only=True)
                original["next_batch"] = 1
                torch.save(original, bad_file)
                with mock.patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"}):
                    with self.assertRaisesRegex(ValueError, "cursor is invalid"):
                        run_joint_family(resumed_root, resumed, resume_from=bad_file)
                original["next_batch"] = 0
                del original["optimizer"]["fp32_params"]
                torch.save(original, bad_file)
                with mock.patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"}):
                    with self.assertRaisesRegex(ValueError, "requires fp32_params"):
                        run_joint_family(resumed_root, resumed, resume_from=bad_file)
        finally:
            torch.set_num_threads(old_threads)

    def test_schedule_horizon_does_not_skip_final_eval_on_data_exhaustion(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(prefix="joint-horizon-") as directory:
                output, stages = self.stages(Path(directory))
                for stage in stages:
                    stage.config.training.max_steps = None
                    stage.config.training.total_steps = 10
                    stage.config.training.eval_interval = 50
                    stage.config.training.save_interval = 0
                with mock.patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"}):
                    result = run_joint_family(output, stages)
                for row in result["runs"]:
                    self.assertEqual(row["global_step"], 2)
                    self.assertIn("eval/avg_loss", row["metrics"])
                self.assertTrue((output / "joint-step2.pt").is_file())
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
