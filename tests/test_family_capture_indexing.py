"""Family planning must apply source indexing before capture-manifest checks."""

import json
import tempfile
import unittest
from pathlib import Path

from draftfit.algorithms.builtin import builtin_algorithm_registry
from draftfit.family import build_family_plan
from draftfit.training.model_loading import resolve_draft_config
from tests.test_family_joint_cpu import fixture
from tests.test_runtime.test_speculators_warmstart import _payload


class FamilyCaptureIndexingTest(unittest.TestCase):
    def test_common_and_per_arm_sources_receive_declaration_and_reload(self):
        for custom in (False, True):
            with self.subTest(custom=custom), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                manifest = fixture(root, seeded=False)
                payload = _payload("dspark")
                del payload["draft_capture_layer_indexing"]
                payload["aux_hidden_state_layer_ids"] = [2, 6]
                payload["mask_token_id"] = 63
                payload["transformer_layer_config"].update(
                    vocab_size=64,
                    hidden_size=32,
                    intermediate_size=64,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                )
                payload["speculators_config"]["verifier"]["name_or_path"] = str(
                    root / "target"
                )
                (root / "draft.json").write_text(json.dumps(payload))
                base_path = root / "base.json"
                base = json.loads(base_path.read_text())
                base["model"]["draft_capture_layer_indexing"] = "embedding_inclusive"
                base_path.write_text(json.dumps(base))
                manifest["runs"] = [
                    {
                        "name": "draft",
                        "strategy": "dspark",
                        **({"draft_config": "./draft.json"} if custom else {}),
                    }
                ]
                path = root / "family.json"
                path.write_text(json.dumps(manifest))
                _, stages = build_family_plan(str(path))
                stage = stages[0]
                self.assertEqual(
                    stage.draft_config.dflash_config["target_layer_ids"], [1, 5]
                )
                self.assertEqual(
                    stage.draft_config.draft_capture_source_layer_ids, [2, 6]
                )
                output = Path(stage.config.model.draft_model_config)
                output.parent.mkdir(parents=True)
                stage.draft_config.to_json_file(str(output))
                reloaded = resolve_draft_config(
                    stage.config,
                    provider=builtin_algorithm_registry()
                    .resolve("dspark")
                    .providers.model.draft_config,
                )
                self.assertEqual(reloaded.dflash_config["target_layer_ids"], [1, 5])
                self.assertEqual(
                    reloaded.draft_capture_layer_indexing, "embedding_inclusive"
                )
