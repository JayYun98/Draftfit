import json, os, tempfile, unittest

class TestDSparkServingConfigExport(unittest.TestCase):
    def test_hf_config_exposes_dspark_runtime_fields(self):
        from draftfit.export.to_hf import _add_dspark_sglang_config

        with tempfile.TemporaryDirectory(prefix="dspark_export_config_") as directory:
            path = os.path.join(directory, "config.json")
            payload = {
                "block_size": 7,
                "dflash_config": {
                    "projector_type": "dspark",
                    "markov_rank": 4,
                    "markov_head_type": "gated",
                    "mask_token_id": 6,
                    "target_layer_ids": [1, 6, 11, 16, 21],
                },
            }
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            _add_dspark_sglang_config(directory)

            with open(path, encoding="utf-8") as handle:
                exported = json.load(handle)

        self.assertEqual(
            exported["dspark_config"],
            {"markov_rank": 4, "markov_head_type": "gated", "mask_token_id": 6},
        )
        self.assertEqual(exported["markov_rank"], 4)
        self.assertEqual(exported["markov_head_type"], "gated")
        self.assertEqual(exported["dflash_config"], payload["dflash_config"])

    def test_hf_config_leaves_non_dspark_export_unchanged(self):
        from draftfit.export.to_hf import _add_dspark_sglang_config

        with tempfile.TemporaryDirectory(prefix="dflash_export_config_") as directory:
            path = os.path.join(directory, "config.json")
            payload = {"dflash_config": {"projector_type": "dflash"}}
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            changed = _add_dspark_sglang_config(directory)

            with open(path, encoding="utf-8") as handle:
                exported = json.load(handle)

        self.assertFalse(changed)
        self.assertEqual(exported, payload)
