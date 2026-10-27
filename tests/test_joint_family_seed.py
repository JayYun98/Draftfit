"""Weights-only continuation from each arm's actual trained HF export."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from draftfit.family import FamilyStage
from draftfit.training.joint_family import (
    _backbone_key,
    run_joint_family,
    validate_joint_family,
)
from tests import test_joint_family_core as core


class JointFamilySeedTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.env = mock.patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        torch.set_num_threads(self.threads)

    def trained_exports(self, root):
        from draftfit.export.to_hf import export_to_hf
        from draftfit.modeling.auto import AutoDraftModel
        from draftfit.training.assembly import build_model_bundle

        output, stages = core.JointFamilyCoreTest().stages(root)
        dspark = stages[0]
        dspark.draft_config.dflash_config["enable_confidence_head"] = True
        dspark.draft_config.to_json_file(dspark.config.model.draft_model_config)
        initial = {}

        def observe(cfg, *, algorithm):
            bundle = build_model_bundle(cfg, algorithm=algorithm)
            initial[cfg.run_id] = {
                k: v.detach().clone()
                for k, v in bundle.draft_model.state_dict().items()
            }
            return bundle

        with mock.patch(
            "draftfit.training.assembly.build_model_bundle", side_effect=observe
        ):
            run_joint_family(output, stages)
        expected, continued = {}, []
        for stage in stages:
            export = root / "exports" / stage.name
            export_to_hf(
                str(output / stage.name / f"{stage.name}-latest"),
                stage.config.model.draft_model_config,
                str(export),
            )
            self.assertFalse((export / "training_state.pt").exists())
            loaded = AutoDraftModel.from_pretrained(export)
            # These are same-family seeds with per-arm custom configs, even
            # though all configs were originally converted from DSpark.
            self.assertEqual(
                loaded.config.architectures, stage.draft_config.architectures
            )
            self.assertIs(loaded.config.dflash_config["sample_from_anchor"], False)
            self.assertIs(stage.draft_config.dflash_config["sample_from_anchor"], False)
            expected[stage.name] = loaded.state_dict()
            cfg = stage.config.model_copy(deep=True)
            cfg.model.draft_checkpoint_path = str(export)
            cfg.output_dir = str(root / "continued" / stage.name)
            cfg.training.max_steps = 1
            cfg.training.total_steps = 1
            self.assertFalse(cfg.training.retrain_anchor_alignment)
            self.assertIsNone(cfg.training.resume_from)
            continued.append(
                FamilyStage(stage.name, cfg, copy.deepcopy(stage.draft_config), None)
            )

        # Exercise every private tensor group, including the optional confidence head.
        groups = {
            "dspark": ("markov_head.", "confidence_head."),
            "selector": ("candidate_selector.", ".attention_conv.", ".mlp_conv."),
        }
        for arm, prefixes in groups.items():
            for prefix in prefixes:
                keys = [k for k in expected[arm] if prefix in k]
                self.assertTrue(keys, (arm, prefix))
                self.assertTrue(
                    any(
                        not torch.equal(expected[arm][k], initial[arm][k]) for k in keys
                    ),
                    (arm, prefix),
                )
        return continued, expected

    def test_distinct_trained_exports_preserve_all_heads_and_continue(self):
        from draftfit.optimizer import BF16Optimizer

        with tempfile.TemporaryDirectory(prefix="joint-seeds-") as directory:
            root = Path(directory)
            stages, expected = self.trained_exports(root)
            validate_joint_family(stages)
            initialize = BF16Optimizer.__init__
            observed = []

            def observe(optimizer, models, **kwargs):
                initialize(optimizer, models, **kwargs)
                self.assertEqual(len(optimizer.optimizer.state), 0)
                self.assertEqual(optimizer.scheduler.last_epoch, 0)
                observed.append(True)
                owner = dict(models[0].named_parameters())
                for stage, model in zip(stages, models):
                    self.assertEqual(
                        model.config.architectures, stage.draft_config.architectures
                    )
                    # Exact loaded values before the first new optimizer update.
                    self.assertEqual(set(model.state_dict()), set(expected[stage.name]))
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(
                            value, expected[stage.name][key], rtol=0, atol=0
                        )
                    for key, param in model.named_parameters():
                        if _backbone_key(key):
                            self.assertIs(param, owner[key])
                        elif key in owner and model is not models[0]:
                            self.assertIsNot(param, owner[key])

            with mock.patch.object(BF16Optimizer, "__init__", observe):
                result = run_joint_family(root / "continued", stages)
            self.assertEqual(observed, [True])
            self.assertEqual([r["global_step"] for r in result["runs"]], [1, 1, 1])
            self.assertEqual([r["status"] for r in result["runs"]], ["complete"] * 3)
            monitor = json.loads((root / "continued" / "monitor.json").read_text())
            self.assertEqual(monitor["origin"], "existing")
            self.assertEqual(monitor["start_step"], 0)
            self.assertIsNone(monitor["resume_from"])
            initial = torch.load(
                root / "continued" / "joint-step0.pt", weights_only=True
            )
            self.assertEqual(initial["step"], 0)
            for arm, tensors in expected.items():
                for key, value in tensors.items():
                    torch.testing.assert_close(
                        initial["drafts"][arm][key], value, rtol=0, atol=0
                    )
            state = torch.load(root / "continued" / "joint-step1.pt", weights_only=True)
            self.assertEqual(state["step"], 1)
            shared_key = "layers.0.self_attn.q_proj.weight"
            self.assertFalse(
                torch.equal(
                    state["drafts"]["dspark"][shared_key],
                    expected["dspark"][shared_key],
                )
            )
            for arm in ("ce", "selector"):
                self.assertTrue(
                    torch.equal(
                        state["drafts"][arm][shared_key],
                        state["drafts"]["dspark"][shared_key],
                    )
                )

    def test_unequal_loaded_shared_weight_rejected_before_optimizer(self):
        from draftfit.modeling.auto import AutoDraftModel

        with tempfile.TemporaryDirectory(prefix="joint-seeds-") as directory:
            root = Path(directory)
            stages, _ = self.trained_exports(root)
            # Damage the last arm so the earlier successful tie cannot hide it.
            source = stages[-1].config.model.draft_checkpoint_path
            model = AutoDraftModel.from_pretrained(source)
            with torch.no_grad():
                model.layers[0].self_attn.q_proj.weight[0, 0] += 1
            model.save_pretrained(source)
            validate_joint_family(stages)
            with mock.patch("draftfit.optimizer.BF16Optimizer") as optimizer:
                with self.assertRaisesRegex(
                    ValueError, "unequal shared weight layers.0.self_attn.q_proj.weight"
                ):
                    run_joint_family(root / "continued", stages)
            optimizer.assert_not_called()
            self.assertFalse((root / "continued").exists())

    def test_scratch_seed_mixes_rejected_in_either_order(self):
        with tempfile.TemporaryDirectory(prefix="joint-seeds-") as directory:
            _, stages = core.JointFamilyCoreTest().stages(Path(directory))
            for seeded in ((True, False, False), (False, True, True)):
                with self.subTest(seeded=seeded):
                    for stage, present in zip(stages, seeded):
                        stage.config.model.draft_checkpoint_path = (
                            "seed" if present else None
                        )
                    with self.assertRaisesRegex(
                        ValueError, "all arms.*seed.*all scratch"
                    ):
                        validate_joint_family(stages)

    def test_distinct_seeds_still_validate_geometry_and_taps(self):
        with tempfile.TemporaryDirectory(prefix="joint-seeds-") as directory:
            _, stages = core.JointFamilyCoreTest().stages(Path(directory))
            for stage in stages:
                stage.config.model.draft_checkpoint_path = f"seed-{stage.name}"
            for field, value, error in (
                ("hidden_size", 48, "incompatible backbone hidden_size"),
                ("target_layer_ids", [2, 5], "incompatible draft target_layer_ids"),
            ):
                with self.subTest(field=field):
                    draft = stages[-1].draft_config
                    saved = copy.deepcopy(draft)
                    if field == "target_layer_ids":
                        draft.dflash_config[field] = value
                    else:
                        setattr(draft, field, value)
                    with self.assertRaisesRegex(ValueError, error):
                        validate_joint_family(stages)
                    stage = stages[-1]
                    stages[-1] = FamilyStage(stage.name, stage.config, saved, None)


if __name__ == "__main__":
    unittest.main()
