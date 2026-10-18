"""Actual CLI joint family updates from scratch and a seed, exports and resume."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


def fixture(root, seeded):
    from safetensors.torch import save_file
    from transformers import Qwen3Config
    from draftfit.inference.capture_manifest import build_capture_manifest, write_capture_manifest
    from draftfit.modeling.auto import AutoDraftModel
    from draftfit.training.model_loading import convert_family_draft_config
    from tests.test_dflash2_integration import tiny_config

    torch.manual_seed(19)
    target = root / "target"
    target.mkdir()
    Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
                head_dim=8, tie_word_embeddings=False).save_pretrained(target)
    save_file({"model.embed_tokens.weight": torch.randn(64, 32),
               "lm_head.weight": torch.randn(64, 32)}, str(target / "model.safetensors"))
    source = convert_family_draft_config(tiny_config(), "dspark")
    source.dflash_config = {**source.dflash_config, "sample_from_anchor": bool(seeded),
                           "markov_rank": 4, "enable_confidence_head": False}
    source.to_json_file(str(root / "draft.json"))
    if seeded:
        AutoDraftModel.from_config(source).save_pretrained(root / "seed")
    capture = build_capture_manifest(
        strategy="dspark", capture_method="target_hidden", capture_layers=(1, 5),
        feature_names=("input_ids", "loss_mask", "hidden_states", "target_last_hidden_states"),
        target_model=str(target), target_revision="0123456789abcdef0123456789abcdef01234567",
        target_feature="target_last_hidden_states", target_repr="hidden_state")
    for split, count in (("train", 4), ("holdout", 2)):
        folder = root / split
        folder.mkdir()
        write_capture_manifest(str(folder / "capture_manifest.json"), capture)
        for index in range(count):
            torch.save({"input_ids": torch.arange(1, 9),
                        "loss_mask": torch.tensor([0, 1, 1, 1, 1, 1, 1, 0]),
                        "hidden_states": torch.randn(8, 64, dtype=torch.bfloat16),
                        "target_last_hidden_states": torch.randn(8, 32, dtype=torch.bfloat16)},
                       folder / f"{index:02d}.ckpt")
    base = {
        "model": {"target_model_path": str(target),
                  "target_revision": "0123456789abcdef0123456789abcdef01234567",
                  "draft_model_config": str(root / "draft.json"),
                  "mask_token_id": 63, "torch_dtype": "bfloat16"},
        "data": {"hidden_states_path": str(root / "train"),
                 "eval_hidden_states_path": str(root / "holdout"), "max_length": 8},
        "training": {"strategy": "dspark", "num_epochs": 1, "max_steps": 2,
                     "batch_size": 1, "num_anchors": 1, "eval_interval": 1,
                     "save_interval": 1, "log_interval": 1, "accumulation_steps": 2,
                     "fsdp_sharding": "NO_SHARD", "attention_backend": "sdpa"}}
    if seeded:
        base["model"]["draft_checkpoint_path"] = str(root / "seed")
    (root / "base.json").write_text(json.dumps(base))
    manifest = {"base_config": "base.json", "output_dir": "outputs", "runs": [
        {"name": "dspark", "strategy": "dspark"},
        {"name": "ce", "strategy": "dflash"},
        {"name": "selector", "strategy": "dflash2"}]}
    if seeded:
        for arm in manifest["runs"][1:]:
            arm["overrides"] = ["training.retrain_anchor_alignment=true"]
    (root / "family.json").write_text(json.dumps(manifest))
    return manifest


class FamilyJointCPU(unittest.TestCase):
    execution_device = "cpu"
    cli_module = "draftfit.cli"
    cli_prefix = ()

    def cli(self, root, *args):
        env = os.environ.copy()
        env.update(SPECFORGE_DEVICE=self.execution_device, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(key, None)
        result = subprocess.run([sys.executable, "-m", self.cli_module, *self.cli_prefix, *args],
                                cwd=root, env=env, text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def exercise(self, seeded):
        from draftfit.modeling.auto import AutoDraftModel
        with tempfile.TemporaryDirectory(prefix="joint-family-") as directory:
            root = Path(directory)
            manifest = fixture(root, seeded)
            plan = self.cli(root, "family", "--manifest", "family.json", "--joint", "--plan")
            self.assertEqual(json.loads(plan.stdout)["execution"], "joint")
            self.assertFalse((root / "outputs").exists())
            self.cli(root, "family", "--manifest", "family.json", "--joint")
            output = root / "outputs"
            results = json.loads((output / "results.json").read_text())
            self.assertEqual([row["status"] for row in results], ["complete"] * 3)
            monitor = json.loads((output / "monitor.json").read_text())
            self.assertEqual(monitor["origin"], "existing" if seeded else "scratch")
            curves = [json.loads(x) for x in (output / "learning-curves.jsonl").read_text().splitlines()]
            for arm in ("dspark", "ce", "selector"):
                train = [r for r in curves if r["arm"] == arm and r["kind"] == "train"]
                self.assertEqual([r["step"] for r in train], [1, 2])
                self.assertTrue(all(r["joint_step_seconds"] > 0 for r in train))
                self.assertTrue(all(r["joint_valid_tokens_per_second"] > 0 for r in train))
                self.assertFalse(any("acceptance_rate" in r for r in train))
                evaluations = [r["step"] for r in curves if r["arm"] == arm and r["kind"] == "evaluation"]
                self.assertEqual(evaluations, [0, 1, 2])
            states = {}
            for name in ("dspark", "ce", "selector"):
                stage = output / name
                checkpoint = stage / f"{name}-latest"
                state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
                self.assertEqual(state["global_step"], 2)
                states[name] = state["draft_state_dict"]
                metrics = [json.loads(line) for line in (stage / "metrics.jsonl").read_text().splitlines()]
                self.assertTrue(any(row["step"] == 1 and "eval/avg_loss" in row for row in metrics))
                self.cli(root, "export", "--to", "hf", "--checkpoint", str(checkpoint),
                         "--draft-config", str(stage / "draft.json"), "--output-dir", str(stage / "hf-export"),
                         "--embedding-source", str(root / "target"))
                loaded = AutoDraftModel.from_pretrained(stage / "hf-export").state_dict()
                for key, value in states[name].items():
                    torch.testing.assert_close(loaded[key].cpu(), value.cpu())
            common = set.intersection(*(set(state) for state in states.values()))
            backbone = [key for key in common if key.startswith("layers.")]
            self.assertTrue(backbone)
            for key in backbone:
                torch.testing.assert_close(states["dspark"][key], states["ce"][key], rtol=0, atol=0)
                torch.testing.assert_close(states["dspark"][key], states["selector"][key], rtol=0, atol=0)
            if seeded:
                initial = AutoDraftModel.from_pretrained(root / "seed").state_dict()
                self.assertTrue(any(not torch.equal(states["dspark"][key].float(), initial[key].float())
                                    for key in backbone))
            manifest["output_dir"] = "resumed"
            (root / "resume.json").write_text(json.dumps(manifest))
            self.cli(root, "family", "--manifest", "resume.json", "--joint", "--resume-from",
                     str(output / "joint-step1.pt"))
            resumed_curves = [json.loads(x) for x in (root / "resumed" / "learning-curves.jsonl").read_text().splitlines()]
            self.assertEqual({r["step"] for r in resumed_curves if r["kind"] == "train"}, {2})
            for name in states:
                restored = torch.load(root / "resumed" / name / f"{name}-latest" / "training_state.pt",
                                      map_location="cpu", weights_only=False)["draft_state_dict"]
                for key, value in states[name].items():
                    torch.testing.assert_close(restored[key], value, rtol=0, atol=0)
            if seeded:
                # A continual run uses the trained DSpark export, new data,
                # a fresh optimizer, and several configured family objectives.
                import shutil
                shutil.copytree(root / "train", root / "new-train")
                path = root / "new-train" / "00.ckpt"
                sample = torch.load(path, weights_only=True)
                sample["hidden_states"] += 0.25
                torch.save(sample, path)
                base = json.loads((root / "base.json").read_text())
                base["model"].pop("draft_model_config")
                base["model"]["draft_checkpoint_path"] = str(output / "dspark" / "hf-export")
                base["data"]["hidden_states_path"] = str(root / "new-train")
                base["training"]["max_steps"] = 1
                base["training"]["accumulation_steps"] = 1
                (root / "continual-base.json").write_text(json.dumps(base))
                manifest.update(base_config="continual-base.json", output_dir="continual")
                manifest["runs"][0]["overrides"] = ["training.dspark_acceptance_loss=eal",
                                                    "training.dspark_acceptance_loss_alpha=0.1"]
                manifest["runs"].extend([
                    {"name": "wtv", "strategy": "dspark", "overrides": [
                        "training.dspark_acceptance_loss=wtv", "training.dspark_acceptance_loss_alpha=0.1"]},
                    {"name": "dpace", "strategy": "dflash", "overrides": [
                        "training.loss_type=dpace", "training.retrain_anchor_alignment=true"]},
                ])
                (root / "continual.json").write_text(json.dumps(manifest))
                self.cli(root, "family", "--manifest", "continual.json", "--joint")
                for name in ("dspark", "ce", "selector", "wtv", "dpace"):
                    state = torch.load(root / "continual" / name / f"{name}-latest" / "training_state.pt",
                                       map_location="cpu", weights_only=True)
                    self.assertEqual(state["global_step"], 1)
                    self.assertTrue(any(not torch.equal(state["draft_state_dict"][key], states["dspark"][key])
                                        for key in backbone))

    def test_existing_dflash_monitor(self):
        from draftfit.modeling.auto import AutoDraftModel, AutoDraftModelConfig
        from draftfit.training.model_loading import convert_family_draft_config
        with tempfile.TemporaryDirectory(prefix="joint-dflash-monitor-") as directory:
            root = Path(directory)
            fixture(root, False)
            config = AutoDraftModelConfig.from_file(str(root / "draft.json"))
            config = convert_family_draft_config(config, "dflash")
            AutoDraftModel.from_config(config).save_pretrained(root / "seed")
            base = json.loads((root / "base.json").read_text())
            base["model"].pop("draft_model_config")
            base["model"]["draft_checkpoint_path"] = str(root / "seed")
            base["training"]["max_steps"] = 1
            (root / "base.json").write_text(json.dumps(base))
            self.cli(root, "family", "--manifest", "family.json", "--joint")
            meta = json.loads((root / "outputs/monitor.json").read_text())
            self.assertEqual(meta["origin"], "existing")
            rows = [json.loads(x) for x in (root / "outputs/learning-curves.jsonl").read_text().splitlines()]
            self.assertEqual({r["step"] for r in rows if r["kind"] == "checkpoint"}, {0, 1})
            self.assertEqual({r["arm"] for r in rows if r["kind"] == "train"}, {"dspark", "ce", "selector"})

    def test_scratch_joint_train_export_and_exact_resume(self):
        self.exercise(False)

    def test_existing_draft_joint_train_export_and_exact_resume(self):
        self.exercise(True)


if __name__ == "__main__":
    unittest.main()
