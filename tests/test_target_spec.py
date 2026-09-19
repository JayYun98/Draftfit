import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from draftfit.algorithms.common.dflash_family_model import OnlineDFlashModel
from draftfit.algorithms.model_providers import populate_dspark_generated_config
from draftfit.cli import main as cli_main
from draftfit.config import Config
from draftfit.target_spec import TargetSpec


def _inspection(lane="hybrid_stateful", state="conv_plus_kv"):
    return {
        "schema_version": "target_inspect_v1",
        "source": "org/lfm2.5",
        "revision": "0123456789abcdef",
        "facts": {
            "architecture_lane": lane,
            "state_kind": state,
            "num_hidden_layers": 8,
            "layer_types": ["conv", "full_attention"] * 4,
            "chat_template_present": True,
        },
        "recommendations": {
            "draft_depth_candidates": [3, 5],
            "target_tap_candidates": [1, 3, 5, 7],
            "train_block_candidates": [7, 16],
            "required_gates": ["state_snapshot_rollback_replay"],
        },
    }


class TargetSpecTest(unittest.TestCase):
    def test_scaffold_round_trips_through_typed_config_and_writes_sidecar(self):
        spec = TargetSpec.from_inspection(_inspection())
        with tempfile.TemporaryDirectory() as directory:
            config_path, spec_path = spec.write_scaffold(
                str(Path(directory) / "run.json")
            )
            config = Config.from_file(config_path)
            self.assertEqual(config.model.target_layer_ids, [1, 3, 5, 7])
            self.assertEqual(config.model.draft_num_hidden_layers, 3)
            self.assertEqual(config.model.draft_block_size, 7)
            self.assertEqual(
                json.loads(Path(spec_path).read_text())["schema_version"],
                "target_spec_v1",
            )
            self.assertEqual(spec.support["serve"], "state_replay_required")
            before = Path(config_path).read_bytes()
            with self.assertRaises(FileExistsError):
                spec.write_scaffold(config_path)
            self.assertEqual(Path(config_path).read_bytes(), before)

    def test_cli_scaffold_reports_required_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inspection = root / "inspection.json"
            output = root / "run.json"
            inspection.write_text(json.dumps(_inspection()), encoding="utf-8")
            self.assertEqual(
                cli_main(
                    [
                        "target",
                        "scaffold",
                        "--inspection",
                        str(inspection),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            self.assertTrue(output.is_file())
            self.assertTrue(Path(str(output) + ".target-spec.json").is_file())

    def test_dspark_generated_config_uses_target_taps_and_attention_mode(self):
        payload = {
            "num_hidden_layers": 3,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
        }
        cfg = SimpleNamespace(model=SimpleNamespace(target_layer_ids=[2, 6]))
        populate_dspark_generated_config(
            payload, SimpleNamespace(num_hidden_layers=8), cfg
        )
        self.assertEqual(payload["dflash_config"]["target_layer_ids"], [2, 6])
        self.assertEqual(payload["dflash_config"]["attention_mode"], "gqa")
        self.assertEqual(payload["dflash_config"]["projector_type"], "dspark")

    def test_uniform_anchor_sampling_is_reproducible_and_random_is_not_required(self):
        model = OnlineDFlashModel(
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            mask_token_id=0,
            block_size=2,
            num_anchors=3,
            anchor_sampling="uniform",
        )
        mask = torch.ones((1, 10), dtype=torch.float32)
        first = model._sample_anchor_positions(10, mask, torch.device("cpu"))[0]
        second = model._sample_anchor_positions(10, mask, torch.device("cpu"))[0]
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.tolist(), [[0, 4, 8]])


if __name__ == "__main__":
    unittest.main()
