"""A real typed family matrix shares eval data and hands off checkpoint roots."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from draftfit.family import build_family_plan, run_family
from draftfit.config import Config
from draftfit.inference.capture_manifest import build_capture_manifest, write_capture_manifest
from draftfit.training.assembly import _configured_logger
from draftfit.training.model_loading import convert_family_draft_config
from tests.test_dflash2_integration import tiny_config


class FamilyWorkflowTest(unittest.TestCase):
    def test_common_scratch_config_converts_per_family_and_rejects_joint_job_conflict(self):
        from tests.test_family_joint_cpu import fixture
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root, seeded=False)
            path = str(root / "family.json")
            _, stages = build_family_plan(path)
            self.assertEqual([s.draft_config.architectures[0] for s in stages],
                             ["DSparkDraftModel", "DFlashDraftModel", "DFlash2DraftModel"])
            self.assertTrue(all(s.config.model.draft_checkpoint_path is None for s in stages))
            with self.assertRaisesRegex(ValueError, "do not combine"):
                run_family(path, joint=True, jobs=2)
            with self.assertRaisesRegex(ValueError, "requires --joint"):
                run_family(path, resume_from="checkpoint.pt")
            self.assertFalse((root / "outputs").exists())
            from draftfit.modeling.draft.taichu_dflash2 import TaichuDFlash2Config
            special = tiny_config(config_class=TaichuDFlash2Config,
                                  architectures=["TaichuDFlash2DraftModel"])
            special.to_json_file(str(root / "taichu.json"))
            manifest = json.loads(Path(path).read_text())
            manifest["runs"][2]["draft_config"] = "./taichu.json"
            Path(path).write_text(json.dumps(manifest))
            _, stages = build_family_plan(path)
            self.assertEqual(stages[2].draft_config.architectures, ["TaichuDFlash2DraftModel"])

    def test_scratch_conversion_uses_run_mask_before_dflash2_validation(self):
        from tests.test_family_joint_cpu import fixture
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root, seeded=False)
            source = json.loads((root / "draft.json").read_text())
            source["dflash_config"].pop("mask_token_id")
            (root / "draft.json").write_text(json.dumps(source))
            _, stages = build_family_plan(str(root / "family.json"))
            self.assertEqual(stages[2].draft_config.dflash_config["mask_token_id"], 63)

    def test_explicit_and_parent_configs_apply_overrides_during_plan(self):
        from tests.test_family_joint_cpu import fixture
        for use_parent in (False, True):
            with self.subTest(use_parent=use_parent), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = fixture(root, seeded=False)
                manifest["runs"] = [
                    {"name": "source", "strategy": "dspark"},
                    {"name": "changed", "strategy": "dspark",
                     **({"from": "source"} if use_parent else {"draft_config": "./draft.json"}),
                     "overrides": ["model.target_layer_ids=[0, 4]"]},
                ]
                (root / "family.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "target layers differ from capture manifest"):
                    build_family_plan(str(root / "family.json"))

    def test_planned_overrides_match_reload_without_mutating_parent(self):
        from draftfit.algorithms.builtin import builtin_algorithm_registry
        from draftfit.training.model_loading import resolve_draft_config
        from tests.test_family_joint_cpu import fixture
        for use_parent in (False, True):
            with self.subTest(use_parent=use_parent), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = fixture(root, seeded=False)
                manifest["runs"] = [
                    {"name": "source", "strategy": "dspark"},
                    {"name": "changed", "strategy": "dspark",
                     **({"from": "source"} if use_parent else {"draft_config": "./draft.json"}),
                     "overrides": ["model.draft_block_size=8"]},
                ]
                (root / "family.json").write_text(json.dumps(manifest))
                _, stages = build_family_plan(str(root / "family.json"))
                self.assertEqual(stages[0].draft_config.block_size, 4)
                self.assertEqual(stages[1].draft_config.block_size, 8)
                output = Path(stages[1].config.model.draft_model_config)
                output.parent.mkdir(parents=True)
                stages[1].draft_config.to_json_file(str(output))
                provider = builtin_algorithm_registry().resolve("dspark").providers.model.draft_config
                reloaded = resolve_draft_config(stages[1].config, provider=provider)
                self.assertEqual(reloaded.block_size, stages[1].draft_config.block_size)
                self.assertEqual(reloaded.dflash_config, stages[1].draft_config.dflash_config)

    def _capture_fixture(self, root, **changes):
        from tests.test_family_joint_cpu import fixture
        fixture(root, seeded=False)
        base_path = root / "base.json"
        base = json.loads(base_path.read_text())
        base["model"]["target_revision"] = None
        base_path.write_text(json.dumps(base))
        fields = dict(
            strategy="dspark", capture_method="dspark", capture_layers=(1, 5),
            feature_names=("input_ids", "loss_mask", "hidden_states", "target_last_hidden_states"),
            target_model=base["model"]["target_model_path"],
            target_revision="a" * 40, tokenizer_version="same-tokenizer",
            target_feature="target_last_hidden_states", target_repr="hidden_state",
        )
        fields.update(changes)
        for split in ("train", "holdout"):
            write_capture_manifest(str(root / split / "capture_manifest.json"),
                                   build_capture_manifest(**fields))
        return fields

    def test_capture_revisions_must_match_without_config_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = self._capture_fixture(root)
            fields["target_revision"] = "b" * 40
            write_capture_manifest(str(root / "holdout" / "capture_manifest.json"),
                                   build_capture_manifest(**fields))
            with self.assertRaisesRegex(ValueError, "train/holdout target revisions differ"):
                build_family_plan(str(root / "family.json"))

    def test_capture_rejects_validly_hashed_unsupported_contracts(self):
        for change in ({"target_repr": "logits"}, {"target_repr": None},
                       {"target_feature": "logits"}, {"target_feature": None},
                       {"capture_method": "eagle3"}, {"capture_method": "dflash"}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self._capture_fixture(root, **change)
                with self.assertRaisesRegex(ValueError, "unsupported capture"):
                    build_family_plan(str(root / "family.json"))

    def test_capture_accepts_native_legacy_and_absent_manifests(self):
        for method in ("dspark", "target_hidden", "absent", "one_absent", "dflash"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                changes = {"capture_method": method if method in ("dspark", "target_hidden") else "dspark"}
                if method == "dflash":
                    changes.update(strategy="dflash", capture_method="dflash",
                                   feature_names=("input_ids", "loss_mask", "hidden_states"),
                                   target_repr=None, target_feature=None)
                self._capture_fixture(root, **changes)
                if method in ("absent", "one_absent"):
                    (root / "holdout" / "capture_manifest.json").unlink()
                    if method == "absent":
                        (root / "train" / "capture_manifest.json").unlink()
                if method == "dflash":
                    path = root / "family.json"
                    manifest = json.loads(path.read_text())
                    manifest["runs"] = manifest["runs"][1:]
                    path.write_text(json.dumps(manifest))
                _, stages = build_family_plan(str(root / "family.json"))
                self.assertEqual(len(stages), 2 if method == "dflash" else 3)

    def test_official_anchor_dspark_plan_covers_continual_ablation_and_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = convert_family_draft_config(tiny_config(), "dspark")
            source.dflash_config = {**source.dflash_config, "sample_from_anchor": True}
            (root / "official").mkdir()
            source.to_json_file(str(root / "official" / "config.json"))
            base = {
                "model": {"target_model_path": "target/model"},
                "data": {"hidden_states_path": str(root / "train"),
                         "eval_hidden_states_path": str(root / "holdout")},
                "training": {"strategy": "dspark", "eval_interval": 1, "max_steps": 2},
            }
            (root / "base.json").write_text(json.dumps(base), encoding="utf-8")
            capture = build_capture_manifest(
                strategy="dspark", capture_method="target_hidden", capture_layers=(1, 5),
                feature_names=("input_ids", "loss_mask", "hidden_states", "target_last_hidden_states"),
                target_model="target/model",
                target_revision="0123456789abcdef0123456789abcdef01234567",
                target_feature="target_last_hidden_states", target_repr="hidden_state",
            )
            for split in ("train", "holdout"):
                (root / split).mkdir()
                write_capture_manifest(str(root / split / "capture_manifest.json"), capture)
            runs = [
                {"name": "official", "strategy": "dspark", "checkpoint": "./official"},
                {"name": "continual", "strategy": "dspark", "from": "official"},
                {"name": "ablation", "strategy": "dspark", "from": "official",
                 "overrides": ["training.dspark_acceptance_loss=eal",
                               "training.dspark_acceptance_loss_alpha=0.1"]},
                {"name": "dflash", "strategy": "dflash", "from": "official",
                 "overrides": ["training.retrain_anchor_alignment=true"]},
                {"name": "dflash2", "strategy": "dflash2", "from": "official",
                 "overrides": ["training.retrain_anchor_alignment=true"]},
            ]
            (root / "family.json").write_text(json.dumps({
                "base_config": "base.json", "output_dir": "outputs", "runs": runs,
            }), encoding="utf-8")
            _, stages = build_family_plan(str(root / "family.json"))
            self.assertEqual([stage.config.training.strategy for stage in stages],
                             ["dspark", "dspark", "dspark", "dflash", "dflash2"])
            self.assertTrue(stages[1].draft_config.dflash_config["sample_from_anchor"])
            self.assertFalse(stages[3].draft_config.dflash_config["sample_from_anchor"])
            self.assertFalse(stages[4].draft_config.dflash_config["sample_from_anchor"])
            planned = subprocess.run(
                [sys.executable, "-m", "draftfit.cli", "family", "--manifest",
                 str(root / "family.json"), "--plan"],
                capture_output=True, text=True, check=True,
            )
            self.assertEqual([run["name"] for run in json.loads(planned.stdout)["runs"]],
                             ["official", "continual", "ablation", "dflash", "dflash2"])
            runs[3]["overrides"] = []
            (root / "family.json").write_text(json.dumps({
                "base_config": "base.json", "output_dir": "outputs", "runs": runs,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sample_from_anchor=false"):
                build_family_plan(str(root / "family.json"))

    def test_metric_stream_records_train_and_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = Config.model_validate({
                "model": {"target_model_path": "target/model"},
                "data": {"hidden_states_path": "/train"},
                "training": {"strategy": "dflash"},
                "output_dir": directory,
            })
            logger = _configured_logger(cfg)
            logger({"loss": 2.0}, 1)
            logger({"eval/avg_loss": 1.5, "eval/avg_acc": 0.25}, 1)
            logger.close()
            rows = [json.loads(line) for line in (Path(directory) / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["train/loss"], 2.0)
            self.assertEqual(rows[1]["eval/avg_acc"], 0.25)

    def test_plan_and_sequential_handoff_with_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            draft = convert_family_draft_config(tiny_config(), "dflash")
            draft.to_json_file(str(root / "draft.json"))
            custom_heads = convert_family_draft_config(draft, "dspark")
            custom_heads.dflash_config = {**custom_heads.dflash_config, "markov_rank": 4}
            custom_heads.to_json_file(str(root / "heads.json"))
            base = {
                "model": {"target_model_path": "target/model", "draft_model_config": str(root / "draft.json")},
                "data": {"hidden_states_path": str(root / "train"),
                         "eval_hidden_states_path": str(root / "holdout")},
                "training": {"strategy": "dflash", "eval_interval": 1, "max_steps": 2},
            }
            (root / "base.json").write_text(json.dumps(base), encoding="utf-8")
            capture = build_capture_manifest(
                strategy="dspark", capture_method="target_hidden", capture_layers=(1, 5),
                feature_names=("input_ids", "loss_mask", "hidden_states", "target_last_hidden_states"),
                target_model="target/model",
                target_revision="0123456789abcdef0123456789abcdef01234567",
                target_feature="target_last_hidden_states", target_repr="hidden_state",
            )
            for split in ("train", "holdout"):
                (root / split).mkdir()
                write_capture_manifest(str(root / split / "capture_manifest.json"), capture)
            manifest = {"base_config": "base.json", "output_dir": "outputs", "runs": [
                {"name": "baseline", "strategy": "dflash"},
                {"name": "heads", "strategy": "dspark", "from": "baseline",
                 "draft_config": "heads.json",
                 "overrides": ["training.dspark_head_only=true"]},
                {"name": "selector", "strategy": "dflash2", "from": "baseline",
                 "overrides": ["training.dflash2_selector_loss_alpha=0.5"]},
                {"name": "eal", "strategy": "dspark", "from": "baseline",
                 "overrides": ["training.dspark_acceptance_loss=eal", "training.dspark_acceptance_loss_alpha=1.0"]},
                {"name": "return", "strategy": "dflash", "from": "heads"},
            ]}
            (root / "family.json").write_text(json.dumps(manifest), encoding="utf-8")
            output, stages = build_family_plan(str(root / "family.json"))
            self.assertEqual([stage.name for stage in stages], ["baseline", "heads", "selector", "eal", "return"])
            self.assertEqual(stages[1].config.model.draft_checkpoint_path, str(output / "baseline"))
            self.assertEqual(stages[1].draft_config.dflash_config["markov_rank"], 4)
            self.assertEqual(stages[2].draft_config.architectures, ["DFlash2DraftModel"])
            self.assertEqual(stages[3].config.training.dspark_acceptance_loss, "eal")
            self.assertEqual(stages[4].draft_config.architectures, ["DFlashDraftModel"])
            self.assertFalse(output.exists())

            def finish(command, check):
                config = json.loads(Path(command[-1]).read_text())
                folder = Path(config["output_dir"])
                (folder / "checkpoint").mkdir()
                (folder / "checkpoint" / "training_state.pt").touch()
                (folder / f"{config['run_id']}-latest").symlink_to(folder / "checkpoint")
                (folder / "metrics.jsonl").write_text(
                    json.dumps({"step": 2, "eval/avg_loss": 1.2, "eval/avg_acc": 0.4}) + "\n",
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0)

            with mock.patch("draftfit.family.subprocess.run", side_effect=finish) as runner:
                result = run_family(str(root / "family.json"))
            self.assertEqual(runner.call_count, 5)
            self.assertEqual([item["status"] for item in result["runs"]], ["complete"] * 5)
            self.assertEqual(result["runs"][1]["metrics"]["eval/avg_acc"], 0.4)
            self.assertTrue((output / "heads" / "draft.json").is_file())

    def test_requires_heldout_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.json").write_text(json.dumps({
                "model": {"target_model_path": "target/model"},
                "data": {"hidden_states_path": "/train"},
                "training": {"strategy": "dflash"},
            }), encoding="utf-8")
            (root / "family.json").write_text(json.dumps({
                "base_config": "base.json", "output_dir": "outputs",
                "runs": [{"name": "baseline", "strategy": "dflash"}],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "held-out eval"):
                build_family_plan(str(root / "family.json"))


if __name__ == "__main__":
    unittest.main()
