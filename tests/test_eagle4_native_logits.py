"""Saved native logits reach the registered EAGLE4 training step on CPU."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from draftfit.algorithms.builtin import builtin_algorithm_registry
from draftfit.algorithms.eagle4.native_logits import (
    NATIVE_KEYS,
    normalize_native_sample,
    prepare_native_logits_target,
    save_native_logits_features,
)
from draftfit.config import Config
from draftfit.modeling.draft.eagle4 import LlamaForCausalLMEagle4
from draftfit.runtime.contracts import TrainBatch
from draftfit.runtime.data_plane.feature_dataloader import FeatureDataLoader
from draftfit.runtime.data_plane.feature_store import LocalFeatureStore
from draftfit.runtime.data_plane.offline_reader import OfflineManifestReader
from draftfit.training.strategies.base import _prepare_eagle_target
from tests.test_modeling.test_eagle4 import tiny_config


def native_row(length=8, start=3, vocab=32):
    mask = torch.zeros(length, dtype=torch.long)
    mask[start - 1 : -1] = 1
    return {
        "input_ids": torch.arange(1, length + 1),
        "loss_mask": mask,
        "aux_hidden_state": torch.randn(length, 48),
        "native_logits": torch.arange((length - start) * vocab)
        .reshape(length - start, vocab)
        .float()
        / vocab,
        "native_logit_start": torch.tensor(start),
    }


def save_rows(path, rows):
    return save_native_logits_features(
        path,
        rows,
        target_model="external-teacher",
        target_revision="immutable-revision",
        target_hidden_size=16,
        target_vocab_size=32,
        capture_layers=[0, 1, 2],
        capture_backend="external-native",
        capture_settings={"runtime": "synthetic CPU fixture"},
    )


class NativeLogitsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.registration = builtin_algorithm_registry().resolve("eagle4")
        self.provider = self.registration.providers.offline_for("text")

    def load_rows(self, path, *, max_len=20, eager=False):
        reader = self.provider.build_reader(
            str(path), run_id="native", ttt_length=2, max_len=max_len
        )
        reader.validate_files = eager
        refs = reader.read()
        loader = FeatureDataLoader(
            LocalFeatureStore(),
            refs=refs,
            batch_size=len(refs),
            strategy="eagle4",
            per_sample_transform=self.provider.build_normalizer(max_len=max_len),
            collate_fn=self.provider.build_collator(),
            gc_interval_s=None,
        )
        try:
            return next(iter(loader))
        finally:
            loader.close()

    def test_original_suffix_alignment_and_single_shift(self):
        row = native_row()
        sample = normalize_native_sample(row, 20)
        ids, logits, mask = prepare_native_logits_target(
            sample["input_ids"],
            sample["target"],
            sample["loss_mask"],
            sample["native_logit_start"],
            "cpu",
            vocab_size=32,
        )
        torch.testing.assert_close(ids[0], torch.tensor([2, 3, 4, 5, 6, 7, 8, 0]))
        torch.testing.assert_close(logits[0, 2:7], row["native_logits"])
        self.assertEqual(logits[0, :2].count_nonzero().item(), 0)
        self.assertEqual(logits[0, -1].count_nonzero().item(), 0)
        torch.testing.assert_close(mask, row["loss_mask"][None, :, None])

    def test_rejects_invalid_native_targets(self):
        sample = normalize_native_sample(native_row(), 20)
        changes = [
            {"target": sample["target"].clone().requires_grad_()},
            {"target": sample["target"].clone().fill_(float("nan"))},
            {"target": sample["target"].clone().fill_(float("inf"))},
            {"target": sample["target"][..., :-1]},
            {"target": sample["target"][:, :-1]},
            {"native_logit_start": torch.tensor([0])},
            {"native_logit_start": torch.tensor([8])},
            {"native_logit_start": torch.tensor([3.0])},
            {"native_logit_start": None},
            {"loss_mask": torch.ones_like(sample["loss_mask"])},
            {"loss_mask": torch.zeros_like(sample["loss_mask"])},
            {"loss_mask": sample["loss_mask"].clone().float().fill_(0.5)},
            {"input_ids": sample["input_ids"].clone().fill_(32)},
            {"attention_mask": torch.tensor([[1, 1, 0, 1, 1, 1, 1, 1]])},
        ]
        for change in changes:
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                values = {**sample, **change}
                prepare_native_logits_target(
                    values["input_ids"],
                    values["target"],
                    values["loss_mask"],
                    values["native_logit_start"],
                    "cpu",
                    vocab_size=32,
                    attention_mask=values["attention_mask"],
                )

    def test_saved_reader_loader_strategy_full_draft_step_without_target_head(self):
        rows = [native_row(), native_row(length=10, start=4)]
        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict("os.environ", {"SPECFORGE_DEVICE": "cpu"}),
        ):
            path = Path(root) / "features"
            save_rows(path, rows)
            batch = self.load_rows(path, eager=True)
            self.assertEqual(batch.metadata["target_repr"], "native_logits")
            self.assertEqual(batch.metadata["target_vocab_size"], 32)
            self.assertEqual(tuple(batch.tensors["target"].shape), (2, 6, 32))
            run = Config.model_validate(
                {
                    "model": {
                        "target_model_path": str(Path(root) / "missing-target"),
                        "torch_dtype": "float32",
                    },
                    "data": {"hidden_states_path": str(path)},
                    "training": {
                        "strategy": "eagle4",
                        "attention_backend": "sdpa",
                        "ttt_length": 2,
                    },
                }
            )
            draft = LlamaForCausalLMEagle4(tiny_config())
            with patch(
                "draftfit.modeling.target.target_head.TargetHead.from_pretrained",
                side_effect=AssertionError("external head loaded"),
            ):
                parts = self.registration.providers.model.build_training_model(
                    run, draft, draft.config, None, None
                )
            self.assertIsNone(parts.target_head)
            strategy = self.registration.providers.step.build(
                parts.model, target_head=None
            )
            observed = {}

            def record(module, args, kwargs):
                observed.update(
                    {
                        k: v.detach().clone()
                        for k, v in kwargs.items()
                        if isinstance(v, torch.Tensor)
                    }
                )

            hook = parts.model.register_forward_pre_hook(record, with_kwargs=True)
            before = {n: p.detach().clone() for n, p in draft.named_parameters()}
            optimizer = torch.optim.SGD(
                [p for p in draft.parameters() if p.requires_grad], lr=0.1
            )
            try:
                output = strategy.forward_loss(batch)
                self.assertTrue(torch.isfinite(output.loss))
                output.loss.backward()
                for name, parameter in draft.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                optimizer.step()
            finally:
                hook.remove()
            for index, row in enumerate(rows):
                first, length = (
                    row["native_logit_start"].item(),
                    row["input_ids"].numel(),
                )
                torch.testing.assert_close(
                    observed["target"][index, first - 1 : length - 1],
                    row["native_logits"],
                )
                torch.testing.assert_close(
                    observed["input_ids"][index, : length - 1], row["input_ids"][1:]
                )
                self.assertEqual(observed["input_ids"][index, length - 1].item(), 0)
            for name in (
                "fc.weight",
                "midlayer.self_attn.q_proj.weight",
                "lm_head.weight",
            ):
                self.assertFalse(
                    torch.equal(before[name], dict(draft.named_parameters())[name]),
                    name,
                )
            for name, parameter in draft.named_parameters():
                if not parameter.requires_grad:
                    torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    def test_native_contract_declares_storage_and_required_start(self):
        contract = self.registration.spec.feature_contract("offline", "text")
        self.assertIn("native_logits", contract.allowed_target_representations)
        self.assertIn("native_logit_start", contract.optional_tensors)
        self.assertEqual(
            contract.storage.required_tensors_for("native_logits"), NATIVE_KEYS
        )
        with self.assertRaises(ValueError):
            contract.storage.required_tensors_for("logits")

    def test_native_reader_requires_manifest_and_correct_feature_contract(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            with self.assertRaisesRegex(ValueError, "capture_manifest"):
                OfflineManifestReader(
                    str(path),
                    strategy="eagle4",
                    feature_keys=tuple(NATIVE_KEYS),
                    target_repr="native_logits",
                )
            save_rows(path / "native", [native_row()])
            with self.assertRaisesRegex(ValueError, "features"):
                OfflineManifestReader(str(path / "native"), strategy="eagle4")

    def test_manifest_vocab_and_tensor_vocab_mismatches_fail_at_strategy(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "native"
            save_rows(path, [native_row()])
            batch = self.load_rows(path)
            from draftfit.algorithms.eagle4.model import Eagle4TrainingModel

            model = Eagle4TrainingModel(LlamaForCausalLMEagle4(tiny_config()), length=2)
            strategy = self.registration.providers.step.build(model)
            batch.metadata["target_vocab_size"] = 31
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                strategy.forward_loss(batch)
            batch.metadata["target_vocab_size"] = 32
            batch.tensors["target"] = batch.tensors["target"][..., :-1]
            with self.assertRaisesRegex(ValueError, "vocab"):
                strategy.forward_loss(batch)

    def test_hidden_state_and_shifted_logits_keep_existing_semantics(self):
        ids, logits, mask = (
            torch.tensor([[2, 3, 0]]),
            torch.randn(1, 3, 32),
            torch.tensor([[1, 1, 0]]),
        )
        actual = _prepare_eagle_target(
            target_head=None,
            target_repr="logits",
            input_ids=ids,
            target=logits,
            loss_mask=mask,
            device=torch.device("cpu"),
        )
        for value, expected in zip(actual, (ids, logits, mask)):
            torch.testing.assert_close(value, expected)
        from draftfit.modeling.target.target_head import TargetHead

        head = torch.nn.Linear(16, 32, bias=False)
        head.preprocess = TargetHead.preprocess.__get__(head)
        hidden = torch.randn(1, 3, 16)
        shifted_ids, projected, shifted_mask = _prepare_eagle_target(
            target_head=head,
            target_repr="hidden_state",
            input_ids=ids,
            target=hidden,
            loss_mask=mask,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(shifted_ids, torch.tensor([[3, 0, 0]]))
        torch.testing.assert_close(projected[:, :-1], head(hidden[:, 1:]))
        torch.testing.assert_close(shifted_mask, mask[..., None])

    def test_native_truncation_preserves_original_start_and_masks_new_tail(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "native"
            row = native_row()
            save_rows(path, [row])
            batch = self.load_rows(path, max_len=6)
            self.assertEqual(batch.tensors["native_logit_start"].item(), 3)
            self.assertEqual(batch.tensors["loss_mask"][0, -1].item(), 0)
            torch.testing.assert_close(
                batch.tensors["target"][0], row["native_logits"][:3]
            )
            with self.assertRaisesRegex(ValueError, "suffix"):
                self.load_rows(path, max_len=3)

    def test_corrupted_saved_features_fail_during_registered_loading(self):
        for corruption in ("supervision", "missing_start", "nonfinite", "attached"):
            with (
                self.subTest(corruption=corruption),
                tempfile.TemporaryDirectory() as root,
            ):
                path = Path(root) / "native"
                row = native_row()
                save_rows(path, [row])
                if corruption == "supervision":
                    row["loss_mask"][0] = 1
                elif corruption == "missing_start":
                    del row["native_logit_start"]
                elif corruption == "nonfinite":
                    row["native_logits"][0, 0] = float("nan")
                else:
                    row["native_logits"].requires_grad_()
                torch.save(row, path / "00000000.ckpt")
                with self.assertRaises((KeyError, ValueError)):
                    self.load_rows(path)

    def test_existing_hidden_state_reader_normalizer_and_mixed_collation(self):
        from draftfit.algorithms.eagle3.data import normalize_offline_sample

        raw = native_row()
        raw.pop("native_logits")
        raw.pop("native_logit_start")
        raw["hidden_state"] = torch.randn(1, 8, 16)
        raw["aux_hidden_state"] = raw["aux_hidden_state"][None]
        with tempfile.TemporaryDirectory() as root:
            torch.save(raw, Path(root) / "legacy.ckpt")
            batch = self.load_rows(root)
            self.assertEqual(batch.metadata["target_repr"], "hidden_state")
            expected = normalize_offline_sample(raw, max_len=20)
            for name, value in expected.items():
                torch.testing.assert_close(batch.tensors[name], value)
            with self.assertRaisesRegex(ValueError, "mix"):
                self.provider.build_collator()(
                    [expected, normalize_native_sample(native_row(), 20)]
                )

    def test_already_shifted_logits_real_strategy_matches_native_step(self):
        from draftfit.algorithms.eagle4.model import Eagle4TrainingModel

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "native"
            row = native_row()
            save_rows(path, [row])
            native = self.load_rows(path)
            dense = torch.zeros(1, 8, 32)
            dense[0, 2:7] = row["native_logits"]
            shifted = TrainBatch(
                sample_ids=["online"],
                strategy="eagle4",
                tensors={
                    **native.tensors,
                    "target": dense,
                    "input_ids": torch.tensor([[2, 3, 4, 5, 6, 7, 8, 0]]),
                    "loss_mask": native.tensors["loss_mask"][..., None],
                },
                metadata={"target_repr": "logits"},
            )
            model = Eagle4TrainingModel(
                LlamaForCausalLMEagle4(tiny_config()), length=2
            ).eval()
            strategy = self.registration.providers.step.build(model)
            with torch.no_grad():
                native_loss = strategy.forward_loss(native).loss
                online_loss = strategy.forward_loss(shifted).loss
            torch.testing.assert_close(native_loss, online_loss, rtol=0, atol=0)

    def test_hidden_state_evaluation_keeps_external_head_loading(self):
        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict("os.environ", {"SPECFORGE_DEVICE": "cpu"}),
        ):
            root = Path(root)
            save_rows(root / "native", [native_row()])
            (root / "legacy-eval").mkdir()
            run = Config.model_validate(
                {
                    "model": {"target_model_path": "unused", "torch_dtype": "float32"},
                    "data": {
                        "hidden_states_path": str(root / "native"),
                        "eval_hidden_states_path": str(root / "legacy-eval"),
                    },
                    "training": {
                        "strategy": "eagle4",
                        "attention_backend": "sdpa",
                        "eval_interval": 1,
                    },
                }
            )
            draft = LlamaForCausalLMEagle4(tiny_config())
            head = torch.nn.Linear(16, 32)
            with patch(
                "draftfit.modeling.target.target_head.TargetHead.from_pretrained",
                return_value=head,
            ) as load:
                parts = self.registration.providers.model.build_training_model(
                    run, draft, draft.config, None, None
                )
            load.assert_called_once()
            self.assertIs(parts.target_head, head)

    def test_native_padding_must_not_contain_unclaimed_logits(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "native"
            save_rows(path, [native_row(), native_row(length=10, start=4)])
            batch = self.load_rows(path)
            batch.tensors["target"][0, 5, 0] = 1
            with self.assertRaisesRegex(ValueError, "padding"):
                prepare_native_logits_target(
                    batch.tensors["input_ids"],
                    batch.tensors["target"],
                    batch.tensors["loss_mask"],
                    batch.tensors["native_logit_start"],
                    "cpu",
                    vocab_size=32,
                    attention_mask=batch.tensors["attention_mask"],
                )

    def test_saver_rejects_bad_dimensions_and_does_not_publish_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "native"
            row = native_row(vocab=31)
            with self.assertRaisesRegex(ValueError, "dimensions"):
                save_rows(path, [row])
            self.assertFalse((path / "capture_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
