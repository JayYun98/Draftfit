"""Real CPU family CLI: one seed, independent training branches, eval and export."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - minimal installations
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class FamilyParallelCPU(unittest.TestCase):
    def test_jobs_two_trains_and_exports_three_independent_stages(self):
        from safetensors.torch import save_file
        from transformers import Qwen3Config

        from draftfit.inference.capture_manifest import build_capture_manifest, write_capture_manifest
        from draftfit.modeling.auto import AutoDraftModel
        from draftfit.training.model_loading import convert_family_draft_config
        from tests.test_dflash2_integration import tiny_config

        with tempfile.TemporaryDirectory(prefix="family-parallel-cpu-") as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            Qwen3Config(
                vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
                head_dim=8, tie_word_embeddings=False,
            ).save_pretrained(target)
            save_file({
                "model.embed_tokens.weight": torch.randn(64, 32),
                "lm_head.weight": torch.randn(64, 32),
            }, str(target / "model.safetensors"))

            source = convert_family_draft_config(tiny_config(), "dspark")
            source.dflash_config = {
                **source.dflash_config,
                "sample_from_anchor": False,
                "markov_rank": 4,
                "enable_confidence_head": False,
            }
            seed = root / "seed"
            AutoDraftModel.from_config(source).save_pretrained(seed)

            capture = build_capture_manifest(
                strategy="dspark", capture_method="target_hidden",
                capture_layers=(1, 5),
                feature_names=("input_ids", "loss_mask", "hidden_states", "target_last_hidden_states"),
                target_model=str(target), target_revision="0123456789abcdef0123456789abcdef01234567",
                target_feature="target_last_hidden_states", target_repr="hidden_state",
            )
            for split, count in (("train", 4), ("holdout", 2)):
                folder = root / split
                folder.mkdir()
                write_capture_manifest(str(folder / "capture_manifest.json"), capture)
                for index in range(count):
                    torch.save({
                        "input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long),
                        "loss_mask": torch.tensor([0, 1, 1, 1, 1, 1, 1, 0]),
                        "hidden_states": torch.randn(8, 64, dtype=torch.bfloat16),
                        "target_last_hidden_states": torch.randn(8, 32, dtype=torch.bfloat16),
                    }, folder / f"{index:02d}.ckpt")

            base = {
                "model": {
                    "target_model_path": str(target),
                    "target_revision": "0123456789abcdef0123456789abcdef01234567",
                    "draft_checkpoint_path": str(seed),
                    "draft_model_config": str(seed / "config.json"),
                    "mask_token_id": 63,
                    "torch_dtype": "bfloat16",
                },
                "data": {
                    "hidden_states_path": str(root / "train"),
                    "eval_hidden_states_path": str(root / "holdout"),
                    "max_length": 8,
                },
                "training": {
                    "strategy": "dspark", "num_epochs": 1, "max_steps": 2,
                    "batch_size": 1, "num_anchors": 1, "eval_interval": 1,
                    "save_interval": 1, "log_interval": 1,
                    "fsdp_sharding": "NO_SHARD", "attention_backend": "sdpa",
                },
            }
            (root / "base.json").write_text(json.dumps(base))
            family = root / "family.json"
            family.write_text(json.dumps({
                "base_config": "base.json", "output_dir": "outputs",
                "runs": [
                    {"name": "seed-train", "strategy": "dspark"},
                    {"name": "ce", "strategy": "dflash", "from": "seed-train"},
                    {"name": "selector", "strategy": "dflash2", "from": "seed-train"},
                ],
            }))
            env = os.environ.copy()
            env.update({
                "SPECFORGE_DEVICE": "cpu", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            })
            for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
                env.pop(key, None)
            completed = subprocess.run(
                [sys.executable, "-m", "draftfit.cli", "family", "--manifest", str(family), "--jobs", "2"],
                cwd=root, env=env, text=True, capture_output=True, timeout=180,
            )
            output = root / "outputs"
            worker_logs = "\n".join(
                f"{path.parent.name}: {path.read_text(errors='replace')[-4000:]}"
                for path in sorted(output.glob("*/worker.log"))
            )
            self.assertEqual(completed.returncode, 0,
                             completed.stdout + completed.stderr + "\n" + worker_logs)
            results = json.loads((output / "results.json").read_text())
            self.assertEqual([row["status"] for row in results], ["complete"] * 3)
            by_name = {row["name"]: row for row in results}
            parent = by_name["seed-train"]
            ce, selector = by_name["ce"], by_name["selector"]
            self.assertLess(parent["finished_at"], min(ce["started_at"], selector["started_at"]))
            self.assertLess(max(ce["started_at"], selector["started_at"]),
                            min(ce["finished_at"], selector["finished_at"]))
            self.assertNotEqual(ce["pid"], selector["pid"])
            source_state = AutoDraftModel.from_pretrained(seed).state_dict()
            stage_receipts = {}
            for name in ("seed-train", "ce", "selector"):
                stage = output / name
                checkpoint = stage / f"{name}-latest"
                shared = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
                self.assertEqual(shared["global_step"], 2)
                self.assertEqual(shared["strategy"], json.loads((stage / "train.json").read_text())["training"]["strategy"])
                self.assertTrue((checkpoint / "training_state_rank0.pt").is_file())
                changed = sum(not torch.equal(value, source_state[key])
                              for key, value in shared["draft_state_dict"].items()
                              if key in source_state and value.shape == source_state[key].shape)
                self.assertGreater(changed, 0)
                rows = [json.loads(line) for line in (stage / "metrics.jsonl").read_text().splitlines()]
                self.assertTrue(any(row.get("step") == 1 and "eval/avg_loss" in row for row in rows))
                exported = stage / "hf-export"
                done = subprocess.run([
                    sys.executable, "-m", "draftfit.cli", "export", "--to", "hf",
                    "--checkpoint", str(checkpoint), "--draft-config", str(stage / "draft.json"),
                    "--output-dir", str(exported), "--embedding-source", str(target),
                ], cwd=root, env=env, text=True, capture_output=True, timeout=60)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                reloaded = AutoDraftModel.from_pretrained(exported)
                for key, value in shared["draft_state_dict"].items():
                    torch.testing.assert_close(reloaded.state_dict()[key].cpu(), value.cpu())
                stage_receipts[name] = {
                    "global_step": shared["global_step"], "changed_tensors": changed,
                    "eval_step_1": True, "export_reload_equal": True,
                    "worker_log": str(Path("outputs") / name / "worker.log"),
                }

            evidence_path = os.environ.get("FAMILY_CPU_EVIDENCE_DIR")
            if evidence_path:
                evidence = Path(evidence_path).resolve()
                self.assertFalse(evidence.exists(), f"evidence directory already exists: {evidence}")
                evidence.parent.mkdir(parents=True, exist_ok=True)
                evidence.mkdir()
                shutil.copytree(output, evidence / "outputs")
                shutil.copy2(family, evidence / "family.json")
                shutil.copy2(root / "base.json", evidence / "base.json")
                (evidence / "receipt.json").write_text(json.dumps({
                    "command": [sys.executable, "-m", "draftfit.cli", "family", "--manifest", "family.json", "--jobs", "2"],
                    "python": sys.version.split()[0],
                    "dependencies": {name: version(name) for name in ("torch", "accelerate", "datasets", "transformers")},
                    "feature_rows": {"train": 4, "holdout": 2},
                    "results": results, "stages": stage_receipts,
                    "family_stdout": completed.stdout, "family_stderr": completed.stderr,
                    "assertions": {"parent_before_children": True, "children_overlap": True, "distinct_child_pids": True},
                }, indent=2))


if __name__ == "__main__":
    unittest.main()
