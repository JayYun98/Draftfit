"""Target pinning must reach actual loaders, not just exported metadata."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from speculative_train_platform.config import Config
from speculative_train_platform.modeling.target.target_utils import (
    TargetEmbeddingsAndHead,
    load_target_config,
)
from speculative_train_platform.training.assembly import (
    _load_text_tokenizer,
    _prompt_cache_key,
)


class TargetRevisionTests(unittest.TestCase):
    def test_config_tokenizer_and_fallback_use_same_revision(self):
        cfg = Config(
            model={"target_model_path": "org/target", "target_revision": "abc123"},
            data={"hidden_states_path": "features"},
            training={"strategy": "dflash"},
        )
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=SimpleNamespace(pad_token_id=0),
        ) as tokenizer:
            _load_text_tokenizer(cfg)
            self.assertEqual(tokenizer.call_args.kwargs["revision"], "abc123")
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "config.json"
            filename.write_text(json.dumps({"hidden_size": 2, "vocab_size": 4}))
            with (
                patch(
                    "speculative_train_platform.modeling.target.target_utils.AutoConfig.from_pretrained",
                    side_effect=ValueError("unknown"),
                ) as config,
                patch(
                    "speculative_train_platform.modeling.target.target_utils.hf_hub_download",
                    return_value=str(filename),
                ) as download,
            ):
                self.assertEqual(
                    load_target_config("org/target", revision="abc123").hidden_size, 2
                )
                self.assertEqual(config.call_args.kwargs["revision"], "abc123")
                self.assertEqual(download.call_args.kwargs["revision"], "abc123")
        previous = _prompt_cache_key(cfg)
        cfg.model.target_revision = "def456"
        self.assertNotEqual(previous, _prompt_cache_key(cfg))

    def test_target_weights_use_revision(self):
        config = SimpleNamespace(hidden_size=2, vocab_size=4, tie_word_embeddings=False)
        with (
            patch(
                "speculative_train_platform.modeling.target.target_utils.load_target_config",
                return_value=config,
            ) as load,
            patch(
                "speculative_train_platform.modeling.target.target_utils.snapshot_download",
                return_value="/fake/cache",
            ) as download,
            patch.object(TargetEmbeddingsAndHead, "_load_weights"),
        ):
            TargetEmbeddingsAndHead.from_pretrained(
                "org/target", revision="abc123", device="cpu"
            )
            self.assertEqual(load.call_args.kwargs["revision"], "abc123")
            self.assertEqual(download.call_args.kwargs["revision"], "abc123")

    def test_managed_capture_server_revision(self):
        from speculative_train_platform.training.capture_contract import (
            ServerCaptureContract,
        )
        from tests.test_runtime.test_launch_plan import (
            _managed_config,
            build_launch_plan,
        )

        with tempfile.TemporaryDirectory() as directory:
            cfg = _managed_config(str(Path(directory) / "attempt"))
            cfg.model.target_revision = "abc123"
            with patch(
                "speculative_train_platform.training.capture_contract.resolve_server_capture_contract",
                return_value=ServerCaptureContract(
                    method="dflash",
                    aux_layer_ids=(1, 2, 3),
                    target_hidden_size=4,
                    target_vocab_size=8,
                    draft_vocab_size=8,
                ),
            ):
                plan = build_launch_plan(cfg, config_path="config.yaml")
            argv = next(
                s.command.argv
                for s in plan.services
                if s.command.label == "capture-server-0"
            )
            self.assertEqual(argv[argv.index("--revision") + 1], "abc123")

    def test_offline_capture_revision(self):
        from scripts.prepare_hidden_states import build_target_model, parse_args

        with patch(
            "sys.argv",
            [
                "prepare",
                "--target-model-path",
                "org/target",
                "--target-revision",
                "abc123",
                "--data-path",
                "data.jsonl",
            ],
        ):
            args = parse_args()
        with patch("scripts.prepare_hidden_states.load_offline_capture") as load:
            build_target_model(args, SimpleNamespace(dtype="bfloat16"), [1, 2, 3])
            self.assertEqual(load.call_args.kwargs["revision"], "abc123")


if __name__ == "__main__":
    unittest.main()
