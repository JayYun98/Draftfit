"""Real tiny Qwen/Gemma online capture versus the offline joint loader."""

import copy
import gc
import hashlib
import json
import os
import random
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch
from transformers import AutoModel, Qwen3Config
from transformers.models.gemma4_unified.configuration_gemma4_unified import (
    Gemma4UnifiedTextConfig,
)

from draftfit.config.schema import TrainingConfig
from draftfit.offline_capture.transformers import OfflineTransformersCapture
from draftfit.training.joint_family import _make_loader
from draftfit.training.joint_online import JointOnlineSource


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _rng():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _assert_rng(test, before):
    after = _rng()
    test.assertEqual(before[0], after[0])
    test.assertEqual(before[1][0], after[1][0])
    np.testing.assert_array_equal(before[1][1], after[1][1])
    test.assertEqual(before[1][2:], after[1][2:])
    test.assertTrue(torch.equal(before[2], after[2]))


class JointOnlineSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_tmp = tempfile.TemporaryDirectory()
        cls.models = {}
        configs = {
            "qwen": Qwen3Config(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
            ),
            "gemma": Gemma4UnifiedTextConfig(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                global_head_dim=16,
                num_global_key_value_heads=1,
                layer_types=["sliding_attention", "full_attention"],
                sliding_window=4,
            ),
        }
        for name, config in configs.items():
            target = Path(cls.models_tmp.name) / name
            AutoModel.from_config(config, attn_implementation="eager").save_pretrained(
                target
            )
            cls.models[name] = str(target)

    @classmethod
    def tearDownClass(cls):
        cls.models_tmp.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.train = [
            {
                "id": "train-0",
                "input_ids": [2, 3, 4, 5, 6],
                "loss_mask": [0, 1, 1, 1, 0],
            },
            {
                "id": "train-1",
                "input_ids": [7, 8, 9, 0],
                "loss_mask": [0, 1, 1, 0],
                "attention_mask": [1, 1, 1, 0],
            },
            {"input_ids": [10, 11, 12, 13], "loss_mask": [0, 1, 1, 1]},
            {"input_ids": [14, 15, 16], "loss_mask": [0, 1, 1]},
            {"input_ids": [17, 18], "loss_mask": [1, 1]},
        ]
        self.evaluation = [
            {"id": "eval-0", "input_ids": [19, 20, 21], "loss_mask": [0, 1, 1]}
        ]
        _write(self.root / "train.jsonl", self.train)
        _write(self.root / "eval.jsonl", self.evaluation)
        self.stage = NS(
            config=NS(
                model=NS(
                    target_model_path=self.models["qwen"],
                    target_revision=None,
                    torch_dtype="float32",
                    cache_dir=None,
                    trust_remote_code=False,
                    hf_attn_implementation="eager",
                    target_layer_ids=None,
                ),
                data=NS(
                    prompts_path=str(self.root / "train.jsonl"),
                    eval_data_path=str(self.root / "eval.jsonl"),
                    max_length=8,
                ),
                training=TrainingConfig(batch_size=2, seed=123),
            ),
            draft_config=NS(dflash_config={"target_layer_ids": [0, 1]}),
        )

    def loader(self, source, **kwargs):
        options = dict(epoch=0, shuffle=False, drop_last=False)
        options.update(kwargs)
        return source.loader(self.stage, **options)[0]

    def test_real_qwen_gemma_offline_parity_and_dtype(self):
        for family in ("qwen", "gemma"):
            for needs_dspark in (False, True):
                for dtype in ("float32", "bfloat16"):
                    with self.subTest(family=family, dspark=needs_dspark, dtype=dtype):
                        self.stage.config.model.target_model_path = self.models[family]
                        self.stage.config.model.torch_dtype = dtype
                        # Explicit override changes tap width from two layers to one.
                        self.stage.config.model.target_layer_ids = [1]
                        source = JointOnlineSource(self.stage, needs_dspark)
                        before_files = set(self.root.rglob("*"))
                        batches = list(self.loader(source))
                        self.assertEqual(set(self.root.rglob("*")), before_files)
                        self.assertEqual(source.capture_calls, 3)
                        self.assertEqual(source.captured_samples, 5)
                        self.assertGreater(source.capture_seconds, 0)
                        self.assertIsNone(source.capture_peak_allocated_bytes)
                        self.assertIsNone(source.capture_peak_reserved_bytes)
                        self.assertEqual(source._teacher.capture_layers, [1])
                        self.assertFalse(source._teacher._model.training)
                        self.assertTrue(
                            all(
                                not p.requires_grad
                                for p in source._teacher._model.parameters()
                            )
                        )
                        for batch in batches:
                            self.assertEqual(
                                batch.tensors["hidden_states"].shape[-1], 32
                            )
                            for name, value in batch.tensors.items():
                                self.assertEqual(value.device.type, "cpu")
                                self.assertFalse(value.requires_grad)
                                if "hidden_states" in name:
                                    self.assertEqual(value.dtype, getattr(torch, dtype))
                        # Independent, unpadded per-sample capture, saved and read
                        # through the existing real offline joint loader.
                        teacher = OfflineTransformersCapture.from_pretrained(
                            self.models[family],
                            torch_dtype=getattr(torch, dtype),
                            device="cpu",
                            attn_implementation="eager",
                        )
                        teacher.set_capture_layers(
                            [1], capture_method="dspark" if needs_dspark else "dflash"
                        )
                        folder = self.root / f"offline-{family}-{needs_dspark}-{dtype}"
                        folder.mkdir()
                        for index, row in enumerate(self.train):
                            length = sum(
                                row.get("attention_mask", [1] * len(row["input_ids"]))
                            )
                            ids = torch.tensor([row["input_ids"][:length]])
                            loss = torch.tensor([row["loss_mask"][:length]])
                            captured = teacher.capture(
                                input_ids=ids,
                                loss_mask=loss,
                                attention_mask=torch.ones_like(ids),
                            )
                            raw = {
                                "input_ids": ids[0],
                                "loss_mask": loss[0],
                                "hidden_states": captured.hidden_states[0],
                            }
                            if needs_dspark:
                                raw["target_last_hidden_states"] = (
                                    captured.last_hidden_states[0]
                                )
                            torch.save(raw, folder / f"{index:02}.ckpt")
                        offline, count = _make_loader(
                            str(folder),
                            self.stage,
                            needs_dspark=needs_dspark,
                            epoch=0,
                            shuffle=False,
                            drop_last=False,
                        )
                        self.assertEqual(count, 5)
                        for actual, expected in zip(batches, offline, strict=True):
                            self.assertEqual(actual.strategy, expected.strategy)
                            self.assertEqual(
                                actual.tensors.keys(), expected.tensors.keys()
                            )
                            for key in expected.tensors:
                                torch.testing.assert_close(
                                    actual.tensors[key], expected.tensors[key]
                                )

    def test_lazy_single_teacher_seek_drop_last_and_eval(self):
        original = OfflineTransformersCapture.from_pretrained
        with patch.object(
            OfflineTransformersCapture, "from_pretrained", wraps=original
        ) as load:
            source = JointOnlineSource(self.stage, True)
            loader = self.loader(source)
            loader.seek(len(loader))
            self.assertEqual(list(loader), [])
            load.assert_not_called()
            self.assertEqual(source.capture_calls, 0)
            loader.seek(1)
            batches = list(loader)
            self.assertEqual([len(batch.sample_ids) for batch in batches], [2, 1])
            self.assertEqual(source.captured_samples, 3)
            self.assertEqual(len(list(self.loader(source, drop_last=True))), 2)
            self.assertEqual(len(list(self.loader(source, evaluation=True))), 1)
            load.assert_called_once_with(
                self.models["qwen"],
                device=torch.device("cpu"),
                revision=None,
                torch_dtype=torch.float32,
                cache_dir=None,
                trust_remote_code=False,
                attn_implementation="eager",
            )
            # seek is one-shot: next iteration includes the entire dataset.
            self.assertEqual(len(list(loader)), 3)
            for invalid in (-1, 4, 0.5, True):
                with self.assertRaises(ValueError):
                    loader.seek(invalid)

    def test_rng_resume_shuffle_and_no_feature_retention(self):
        original_load = OfflineTransformersCapture.from_pretrained
        original_capture = OfflineTransformersCapture.capture

        def consume_rng():
            random.random()
            np.random.random()
            torch.rand(7)

        def load(*args, **kwargs):
            consume_rng()
            return original_load(*args, **kwargs)

        def capture(teacher, **kwargs):
            consume_rng()
            return original_capture(teacher, **kwargs)

        with (
            patch.object(
                OfflineTransformersCapture, "from_pretrained", side_effect=load
            ),
            patch.object(OfflineTransformersCapture, "capture", capture),
        ):
            before = _rng()
            source = JointOnlineSource(self.stage, True)
            batches = list(self.loader(source, epoch=2, shuffle=True))
            _assert_rng(self, before)
            resumed = JointOnlineSource(self.stage, True)
            loader = self.loader(resumed, epoch=2, shuffle=True)
            loader.seek(1)
            for actual, expected in zip(loader, batches[1:], strict=True):
                self.assertEqual(actual.sample_ids, expected.sample_ids)
                for key in actual.tensors:
                    self.assertTrue(
                        torch.equal(actual.tensors[key], expected.tensors[key])
                    )
            _assert_rng(self, before)
            self.assertEqual(resumed.capture_calls, 2)
            stream = iter(self.loader(source))
            batch = next(stream)
            refs = [weakref.ref(value) for value in batch.tensors.values()]
            del batch
            gc.collect()
            self.assertTrue(all(ref() is None for ref in refs))
            self.assertTrue(
                all(
                    value.device.type == "cpu"
                    for rows in source._rows.values()
                    for row in rows
                    for value in row.values()
                    if isinstance(value, torch.Tensor)
                )
            )
            self.assertNotEqual(
                self.loader(source, epoch=2, shuffle=True).indices,
                self.loader(source, epoch=3, shuffle=True).indices,
            )

    def test_rng_restored_on_construction_and_capture_failure(self):
        source = JointOnlineSource(self.stage, True)

        def fail(*args, **kwargs):
            torch.rand(3)
            random.random()
            np.random.random()
            raise RuntimeError("injected teacher failure")

        before = _rng()
        with patch.object(
            OfflineTransformersCapture, "from_pretrained", side_effect=fail
        ):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                next(iter(self.loader(source)))
        _assert_rng(self, before)
        self.assertIsNone(source._teacher)
        next(iter(self.loader(source)))
        before = _rng()
        with patch.object(OfflineTransformersCapture, "capture", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                next(iter(self.loader(source)))
        _assert_rng(self, before)
        self.assertEqual(source.capture_calls, 1)

    def test_fingerprint_hashes_exact_bytes_and_returns_copy(self):
        source = JointOnlineSource(self.stage, False)
        identity = source.fingerprint_sources()
        json.dumps(identity)
        for name in ("train", "eval"):
            self.assertEqual(
                identity[name]["sha256"],
                hashlib.sha256((self.root / f"{name}.jsonl").read_bytes()).hexdigest(),
            )
        identity["train"]["sha256"] = "modified"
        self.assertNotEqual(identity, source.fingerprint_sources())
        _write(self.root / "train.jsonl", self.train[:2])
        # Existing source still owns the validated snapshot of token rows.
        self.assertEqual(len(source._rows[False]), 5)
        self.assertNotEqual(
            source.fingerprint_sources(),
            JointOnlineSource(self.stage, False).fingerprint_sources(),
        )

    def test_bad_token_rows_fail_before_teacher_load(self):
        valid = {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]}
        invalid = [
            {},
            [],
            {**valid, "input_ids": []},
            {**valid, "input_ids": [1, True, 3]},
            {**valid, "input_ids": [1, 2.0, 3]},
            {**valid, "input_ids": [1, -2, 3]},
            {**valid, "input_ids": [1, 2**63, 3]},
            {"input_ids": list(range(9)), "loss_mask": [1] * 9},
            {**valid, "loss_mask": [0, 1]},
            {**valid, "loss_mask": [0, 2, 1]},
            {**valid, "loss_mask": [0, 1.0, 1]},
            {**valid, "loss_mask": [0, 0, 0]},
            {**valid, "loss_mask": [1, 0, 1]},
            {**valid, "attention_mask": [0, 1, 1]},
            {**valid, "attention_mask": [1, 0, 1]},
            {**valid, "attention_mask": [0, 0, 0]},
            {**valid, "attention_mask": [1, 1, 0]},
            {**valid, "attention_mask": [1, 1]},
            {**valid, "attention_mask": [1, 1, 2]},
            {**valid, "messages": []},
            {**valid, "text": "raw"},
        ]
        with patch.object(OfflineTransformersCapture, "from_pretrained") as load:
            for row in invalid:
                with self.subTest(row=row):
                    _write(self.root / "train.jsonl", [row])
                    with self.assertRaises(ValueError):
                        JointOnlineSource(self.stage, False)
            for text in ("", "\n", "{invalid json}\n"):
                (self.root / "train.jsonl").write_text(text)
                with self.assertRaises(ValueError):
                    JointOnlineSource(self.stage, False)
            load.assert_not_called()

    def test_cross_split_overlap_by_tokens_or_ids(self):
        cases = [
            {**self.train[0], "id": "different-id"},
            {**self.evaluation[0], "id": "train-0"},
            {**self.evaluation[0], "sample_id": "train-1"},
            # Same active tokens, different explicit padding and loss mask.
            {"input_ids": [7, 8, 9], "loss_mask": [1, 1, 1]},
        ]
        for row in cases:
            with self.subTest(row=row):
                _write(self.root / "eval.jsonl", [row])
                with self.assertRaisesRegex(ValueError, "overlap"):
                    JointOnlineSource(self.stage, True)

    def test_capture_peaks_include_eval_and_survive_parent_resets(self):
        source = JointOnlineSource(self.stage, True)
        # Load the real CPU teacher first. Only CUDA accounting is simulated;
        # the capture itself still executes the actual Qwen model.
        next(iter(self.loader(source)))
        device = torch.device("cuda", 0)
        with (
            patch("draftfit.training.joint_online.get_device", return_value=device),
            patch.object(torch.cuda, "get_rng_state_all", return_value=[]),
            patch.object(torch.cuda, "set_rng_state_all"),
            patch.object(
                torch.cuda, "max_memory_allocated", side_effect=[400, 100, 500]
            ) as allocated,
            patch.object(
                torch.cuda, "max_memory_reserved", side_effect=[800, 200, 700]
            ) as reserved,
            patch.object(torch.cuda, "reset_peak_memory_stats") as reset,
        ):
            next(iter(self.loader(source, evaluation=True)))
            self.assertEqual(source.capture_peak_allocated_bytes, 400)
            self.assertEqual(source.capture_peak_reserved_bytes, 800)
            next(iter(self.loader(source)))
            self.assertEqual(source.capture_peak_allocated_bytes, 400)
            self.assertEqual(source.capture_peak_reserved_bytes, 800)
            next(iter(self.loader(source)))
            self.assertEqual(source.capture_peak_allocated_bytes, 500)
            self.assertEqual(source.capture_peak_reserved_bytes, 800)
            allocated.assert_called_with(device)
            reserved.assert_called_with(device)
            reset.assert_not_called()

    def test_duplicate_full_ids_with_different_attention_rejected(self):
        _write(
            self.root / "train.jsonl",
            [
                {
                    "input_ids": [1, 2, 3],
                    "loss_mask": [1, 1, 0],
                    "attention_mask": [1, 1, 0],
                },
            ],
        )
        _write(
            self.root / "eval.jsonl",
            [
                {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]},
            ],
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            JointOnlineSource(self.stage, True)

    def test_loader_rejects_incompatible_stage(self):
        source = JointOnlineSource(self.stage, True)
        other = copy.deepcopy(self.stage)
        other.config.model.target_layer_ids = [1]
        with self.assertRaisesRegex(ValueError, "share"):
            source.loader(other, 0, False, False)


if __name__ == "__main__":
    unittest.main()
