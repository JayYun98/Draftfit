"""Real target/current/export comparisons; CUDA is explicit opt-in only."""

import hashlib
import json
import os
import random
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from draftfit.config.schema import JointVerificationConfig
from draftfit.training import joint_verification as verification
from tests import test_joint_verification as fixtures


class ExistingDraftVerificationTest(unittest.TestCase):
    execution_device = "cpu"

    def setUp(self):
        self.f = fixtures.JointVerificationTest()
        self.f.execution_device = self.execution_device
        self.addCleanup(self.f.doCleanups)
        self.f.setUp()
        self.paths = {}
        for index, (stage, bundle) in enumerate(zip(self.f.stages, self.f.bundles)):
            path = Path(self.f.directory.name) / "old" / stage.name
            # Actual older weights, independently serialized, including sharded HF.
            state = {
                k: v.detach().cpu().clone() + (0.01 if v.is_floating_point() else 0)
                for k, v in bundle.draft_model.state_dict().items()
            }
            bundle.draft_model.save_pretrained(
                path, state_dict=state, max_shard_size="10KB" if index == 2 else "5GB"
            )
            (path / "README.md").write_text(
                "CPU test reference; immutable during comparison.\n"
            )
            self.paths[stage.name] = str(path.resolve())
        self.f.config["existing_drafts"] = self.paths

    def evaluate(self):
        return verification.evaluate_generation(
            self.f.stages,
            self.f.bundles,
            self.f.source,
            self.f.optimizer,
            1,
            self.f.config,
        )

    def test_real_three_family_comparison_bindings_and_state(self):
        self.f.config["trials"] = 2
        self.f.bundles[0].draft_model.layers[0].eval()
        before = self.f.snapshot()
        current_ids = {
            id(p) for b in self.f.bundles for p in b.draft_model.parameters()
        }
        generate = verification._generate
        calls, old_ids = [], set()

        def observe(teacher, bundle, *args, **kwargs):
            self.assertIs(teacher, before["teacher"])
            old = all(bundle.draft_model is not b.draft_model for b in self.f.bundles)
            calls.append(
                "old" if old else "current" if kwargs["speculative"] else "target"
            )
            if old:
                old_ids.add(id(bundle.draft_model))
                self.assertIs(bundle.model.lm_head, self.f.bundles[0].model.lm_head)
                self.assertIs(
                    bundle.model.embed_tokens, self.f.bundles[0].model.embed_tokens
                )
                self.assertFalse(bundle.model.training)
                self.assertFalse(any(m.training for m in bundle.draft_model.modules()))
                self.assertTrue(
                    current_ids.isdisjoint(
                        id(p) for p in bundle.draft_model.parameters()
                    )
                )
                self.assertTrue(all(p.grad is None for p, *_ in before["parameters"]))
                self.assertTrue(
                    all(p.device.type == "cpu" for p in self.f.optimizer.fp32_params)
                )
                current = next(
                    b
                    for b in self.f.bundles
                    if type(b.draft_model) is type(bundle.draft_model)
                )
                self.assertTrue(
                    any(
                        not torch.equal(v, current.draft_model.state_dict()[k])
                        for k, v in bundle.draft_model.state_dict().items()
                    )
                )
            return generate(teacher, bundle, *args, **kwargs)

        with mock.patch.object(verification, "_generate", side_effect=observe):
            result = self.evaluate()
        self.f.assert_restored(before)
        self.assertEqual(calls, ["target"] * 4 + ["current"] * 12 + ["old"] * 12)
        self.assertGreaterEqual(
            len(old_ids), 1
        )  # Python can reuse released object IDs.
        self.assertEqual(len(result["candidates"]), 3)
        self.assertEqual(len(result["existing_drafts"]), 3)
        self.assertTrue(result["tokenparity"])
        self.assertEqual(result["parity_failures"], [])
        self.assertFalse(result["production_engine_speed"])
        for group in (result["candidates"], result["existing_drafts"]):
            for candidate in group:
                for run in candidate["runs"]:
                    pair = run["baseline_pair"]
                    baseline = result["baseline"]["runs"][pair["run_index"]]
                    self.assertEqual(run["request_id"], baseline["request_id"])
                    self.assertEqual(pair["request_id"], baseline["request_id"])
                    self.assertEqual(
                        pair["output_sha256"],
                        verification._json_sha(baseline["output_token_ids"]),
                    )
                    self.assertEqual(
                        run["output_token_ids"], baseline["output_token_ids"]
                    )
                    self.assertEqual(run["trial"], baseline["trial"])
        self.assertEqual(len({r["request_id"] for r in result["baseline"]["runs"]}), 4)
        for old in result["existing_drafts"]:
            self.assertFalse(old["eligible_for_selection"])
            source = old["bindings"]["source"]
            root = Path(self.paths[old["name"]])
            expected = {
                p.relative_to(root)
                .as_posix(): hashlib.sha256(p.read_bytes())
                .hexdigest()
                for p in root.rglob("*")
                if p.is_file()
            }
            self.assertEqual(source["files"], expected)
            self.assertEqual(source["sha256"], verification._json_sha(expected))
        json.dumps(result, allow_nan=False)

    def test_next_actual_update_is_exact_with_old_comparison(self):
        self.f.test_next_real_training_update_is_identical_with_and_without_verification()

    def test_old_forward_failure_restores_modes_rng_optimizer_and_identities(self):
        self.f.bundles[1].model.eval()
        self.f.bundles[0].draft_model.layers[0].train()
        before = self.f.snapshot()
        generate = verification._generate
        old_drafts = []

        def fail(teacher, bundle, *args, **kwargs):
            if all(bundle.draft_model is not b.draft_model for b in self.f.bundles):
                old_drafts.append(bundle.draft_model)
                random.random()
                np.random.random()
                torch.rand(4)
                raise RuntimeError("old draft forward failed")
            return generate(teacher, bundle, *args, **kwargs)

        with mock.patch.object(verification, "_generate", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "old draft forward failed"):
                self.evaluate()
        self.assertTrue(old_drafts)
        self.assertTrue(all(p.device.type == "cpu" for p in old_drafts[0].parameters()))
        self.f.assert_restored(before)

    def test_old_parity_failure_is_aggregate_failure_without_eligible_old_arm(self):
        generate = verification._generate

        def corrupt(teacher, bundle, *args, **kwargs):
            run = generate(teacher, bundle, *args, **kwargs)
            if all(bundle.draft_model is not b.draft_model for b in self.f.bundles):
                run["output_token_ids"][-1] = (
                    run["output_token_ids"][-1] + 1
                ) % bundle.target_vocab_size
            return run

        with mock.patch.object(verification, "_generate", side_effect=corrupt):
            result = self.evaluate()
        self.assertFalse(result["tokenparity"])
        self.assertTrue(all(c["tokenparity"] for c in result["candidates"]))
        self.assertEqual(
            {f["kind"] for f in result["parity_failures"]}, {"existing_draft"}
        )
        self.assertEqual({c["name"] for c in result["candidates"]}, set(self.paths))

    def test_metadata_change_during_comparison_fails_and_restores_state(self):
        before = self.f.snapshot()
        generate = verification._generate

        def change(teacher, bundle, *args, **kwargs):
            run = generate(teacher, bundle, *args, **kwargs)
            if all(bundle.draft_model is not b.draft_model for b in self.f.bundles):
                (Path(next(iter(self.paths.values()))) / "README.md").write_text(
                    "changed"
                )
            return run

        with mock.patch.object(verification, "_generate", side_effect=change):
            with self.assertRaisesRegex(ValueError, "export changed"):
                self.evaluate()
        self.f.assert_restored(before)

    def test_preflight_rejects_unknown_arm_architecture_and_incomplete_weights(self):
        before = self.f.snapshot()
        bad = dict(
            self.f.config, existing_drafts={"typo": next(iter(self.paths.values()))}
        )
        with self.assertRaisesRegex(ValueError, "unknown arms"):
            verification.preflight_existing_drafts(self.f.stages, self.f.bundles, bad)
        names = list(self.paths)
        bad = dict(self.f.config, existing_drafts={names[0]: self.paths[names[1]]})
        with self.assertRaisesRegex(ValueError, "architecture differs"):
            verification.preflight_existing_drafts(self.f.stages, self.f.bundles, bad)
        weights = Path(self.paths[names[0]]) / "model.safetensors"
        state = load_file(weights)
        state.pop(next(iter(state)))
        save_file(state, weights, metadata={"format": "pt"})
        with self.assertRaisesRegex(ValueError, "missing_keys"):
            verification.preflight_existing_drafts(
                self.f.stages, self.f.bundles, self.f.config
            )
        self.f.assert_restored(before)

    def test_preflight_rejects_bad_shape_and_semantic_config(self):
        path = Path(next(iter(self.paths.values())))
        config_path = path / "config.json"
        original = config_path.read_text()
        config = json.loads(original)
        config["dflash_config"]["mask_token_id"] = 0
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "incompatible dflash_config"):
            verification.preflight_existing_drafts(
                self.f.stages, self.f.bundles, self.f.config
            )
        config_path.write_text(original)
        state = load_file(path / "model.safetensors")
        key = next(k for k, v in state.items() if v.ndim == 2)
        state[key] = state[key][:1].clone()
        save_file(state, path / "model.safetensors", metadata={"format": "pt"})
        with self.assertRaises((ValueError, RuntimeError)):
            verification.preflight_existing_drafts(
                self.f.stages, self.f.bundles, self.f.config
            )

    def test_schema_and_mapping_settings_reject_before_training(self):
        parsed = JointVerificationConfig(existing_drafts=self.paths)
        self.assertEqual(verification._settings(parsed)["existing_drafts"], self.paths)
        self.assertEqual(
            verification._settings(JointVerificationConfig())["existing_drafts"], {}
        )
        for value in (
            None,
            [],
            {"": next(iter(self.paths.values()))},
            {"x": "not-a-local-export"},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                verification._settings(dict(existing_drafts=value))
        for value in (dict(existing_drafts=self.paths, offload_optimizer=False),):
            with self.assertRaisesRegex(ValueError, "offload_optimizer"):
                JointVerificationConfig(**value)
            with self.assertRaisesRegex(ValueError, "offload_optimizer"):
                verification._settings(value)
        path = Path(next(iter(self.paths.values()))) / "config.json"
        payload = json.loads(path.read_text())
        payload["architectures"] = ["UnsupportedDraft"]
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(
            ValueError, "unsupported existing draft architecture"
        ):
            JointVerificationConfig(existing_drafts=self.paths)

    def test_shard_paths_cannot_escape_hashed_inventory(self):
        path = Path(list(self.paths.values())[2]) / "model.safetensors.index.json"
        payload = json.loads(path.read_text())
        payload["weight_map"][
            next(iter(payload["weight_map"]))
        ] = "../external.safetensors"
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "bound files inside the export"):
            verification.preflight_existing_drafts(
                self.f.stages, self.f.bundles, self.f.config
            )

    def test_old_timeout_restores_state_and_releases_reference(self):
        before = self.f.snapshot()
        generate = verification._generate

        def expire(teacher, bundle, *args, **kwargs):
            if all(bundle.draft_model is not b.draft_model for b in self.f.bundles):
                expired = dict(args[1], deadline_unix=0.0)
                return generate(teacher, bundle, args[0], expired, **kwargs)
            return generate(teacher, bundle, *args, **kwargs)

        with mock.patch.object(verification, "_generate", side_effect=expire):
            with self.assertRaisesRegex(TimeoutError, "deadline"):
                self.evaluate()
        self.f.assert_restored(before)


@unittest.skipUnless(
    torch.cuda.is_available() and os.environ.get("DRAFTFIT_RUN_QUEUE_CUDA") == "1",
    "explicit CUDA existing-draft comparison gate",
)
class ExistingDraftVerificationCUDA(ExistingDraftVerificationTest):
    execution_device = "cuda"


if __name__ == "__main__":
    unittest.main()
