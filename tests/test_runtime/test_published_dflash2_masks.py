"""Published DFlash2 masks and conditional identity-backbone conversion on CPU."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from draftfit.algorithms.common.dflash_family_model import (
    OnlineDFlash2Model,
    OnlineDFlashModel,
)
from draftfit.export.to_hf import export_to_hf
from draftfit.modeling.auto import AutoDraftModel
from draftfit.modeling.draft.dflash2 import DFlash2Config
from draftfit.training.model_loading import (
    _draft_config_from_dict,
    convert_family_draft_config,
    load_draft_config_source,
    warm_start_draft_model,
)
from tests.test_runtime.test_speculators_warmstart import _payload, _target
from tests.test_utils.test_speculators_mask import _published_reference


def _source_config(layers, window, non_causal):
    payload = _payload()
    payload["draft_capture_layer_indexing"] = "embedding_inclusive"
    payload["sliding_window_non_causal"] = non_causal
    # Published metadata governs each layer, independently of native DF2 defaults.
    payload["is_causal"] = True
    payload["transformer_layer_config"].update(
        num_hidden_layers=len(layers),
        layer_types=layers,
        sliding_window=window,
    )
    return _draft_config_from_dict(payload, target_config=_target())


def _wrapper(draft, kind=OnlineDFlash2Model, backend="sdpa", head=None, embedding=None):
    draft.config._attn_implementation = backend
    return kind(
        draft_model=draft,
        target_lm_head=(
            head if head is not None else torch.nn.Linear(16, 32, bias=False)
        ),
        target_embed_tokens=(
            embedding if embedding is not None else torch.nn.Embedding(32, 16)
        ),
        mask_token_id=31,
        block_size=4,
        attention_backend=backend,
    )


class PublishedDFlash2MaskTest(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(7)

    def test_actual_wrapper_masks_match_reference_sdpa_and_flex(self):
        anchors = torch.tensor([[0, 6, 11], [3, 7, 10]])
        keep = torch.tensor([[True, True, False], [True, False, True]])
        inputs = dict(
            input_ids=torch.arange(16).expand(2, -1),
            hidden_states=torch.randn(2, 16, 48),
            loss_mask=torch.ones(2, 16),
        )
        for layers in (["sliding_attention"], ["full_attention", "sliding_attention"]):
            for window in (1, 3):
                for non_causal in (False, True):
                    source = _source_config(layers, window, non_causal)
                    payload = source.to_dict()
                    payload.update(
                        architectures=["DFlash2DraftModel"],
                        model_type="qwen3_dflash2",
                        target_num_hidden_layers=28,
                    )
                    # Direct construction isolates runtime mask parity from the loader guard.
                    draft = AutoDraftModel.from_config(DFlash2Config.from_dict(payload))
                    for backend in ("sdpa", "flex_attention"):
                        with self.subTest(
                            layers=layers,
                            window=window,
                            non_causal=non_causal,
                            backend=backend,
                        ):
                            model = _wrapper(draft, backend=backend)
                            masks = []

                            def record(module, args, kwargs):
                                masks.append(kwargs["attention_mask"])

                            def flex_forward(**kwargs):
                                masks.append(kwargs["attention_mask"])
                                return kwargs["noise_embedding"]

                            handle = (
                                draft.register_forward_pre_hook(
                                    record, with_kwargs=True
                                )
                                if backend == "sdpa"
                                else None
                            )
                            context = (
                                mock.patch.object(
                                    draft, "forward", side_effect=flex_forward
                                )
                                if backend == "flex_attention"
                                else torch.no_grad()
                            )
                            try:
                                with (
                                    context,
                                    mock.patch.object(
                                        model,
                                        "_sample_anchor_positions",
                                        return_value=(anchors, keep),
                                    ),
                                ):
                                    model._forward_draft_blocks(**inputs)
                            finally:
                                if handle is not None:
                                    handle.remove()
                            for kind, width in (
                                ("full_attention", None),
                                ("sliding_attention", window),
                            ):
                                expected = _published_reference(
                                    anchors, keep, 16, 4, width, not non_causal
                                )
                                actual = masks[0][kind]
                                if backend == "sdpa":
                                    self.assertTrue(torch.equal(actual, expected), kind)
                                else:
                                    for batch in range(2):
                                        self.assertTrue(
                                            torch.equal(
                                                actual.mask_mod(
                                                    batch,
                                                    0,
                                                    torch.arange(12)[:, None],
                                                    torch.arange(28)[None, :],
                                                ),
                                                expected[batch, 0],
                                            ),
                                            kind,
                                        )

    def test_identity_fixture_warm_start_hidden_parity_export_reload_and_backward(self):
        inputs = dict(
            input_ids=torch.arange(16).unsqueeze(0),
            hidden_states=torch.randn(1, 16, 48),
            loss_mask=torch.ones(1, 16),
        )
        anchors, keep = torch.tensor([[2, 9]]), torch.tensor([[True, True]])
        for non_causal in (False, True):
            with (
                self.subTest(non_causal=non_causal),
                tempfile.TemporaryDirectory() as d,
            ):
                root = Path(d)
                source = AutoDraftModel.from_config(
                    _source_config(
                        ["full_attention", "sliding_attention"], 3, non_causal
                    )
                )
                source.save_pretrained(root / "seed")
                config = convert_family_draft_config(source.config, "dflash2")
                draft = AutoDraftModel.from_config(config)
                report = warm_start_draft_model(
                    draft, str(root / "seed"), draft_config=config, strategy="dflash2"
                )
                self.assertTrue(report.missing_keys)
                for key, value in source.state_dict().items():
                    torch.testing.assert_close(
                        draft.state_dict()[key], value, rtol=0, atol=0
                    )
                for layer in draft.layers:
                    for convolution in (layer.attention_conv, layer.mlp_conv):
                        # Conditional identity proof only: HF post_init initializes
                        # these Linear weights. Production initialization is unchanged.
                        torch.nn.init.zeros_(convolution.kernel_projection.weight)
                        probe = torch.randn(1, 8, 16)
                        prepared, kernel = convolution.prepare(probe)
                        torch.testing.assert_close(
                            convolution.finish(prepared, kernel), probe, rtol=0, atol=0
                        )
                original = _wrapper(source, kind=OnlineDFlashModel)
                converted = _wrapper(
                    draft, head=original.lm_head, embedding=original.embed_tokens
                )

                def hidden(model):
                    with mock.patch.object(
                        model, "_sample_anchor_positions", return_value=(anchors, keep)
                    ):
                        return model._forward_draft_blocks(**inputs)[2]

                expected = hidden(original)
                torch.testing.assert_close(hidden(converted), expected, rtol=0, atol=0)
                config_path = root / "draft.json"
                config.to_json_file(config_path)
                checkpoint = root / "training_state.pt"
                torch.save(
                    {"strategy": "dflash2", "draft_state_dict": draft.state_dict()},
                    checkpoint,
                )
                export_to_hf(str(checkpoint), str(config_path), str(root / "export"))
                reloaded = load_draft_config_source(str(root / "export"))
                restored = AutoDraftModel.from_pretrained(
                    root / "export", config=reloaded, dtype=torch.float32
                )
                self.assertEqual(reloaded.draft_capture_source_layer_ids, [1, 13, 25])
                self.assertEqual(
                    reloaded.dflash_config["target_layer_ids"], [0, 12, 24]
                )
                self.assertEqual(reloaded.dflash_config["pure_draft_prefix_len"], 1)
                self.assertEqual(
                    reloaded.speculators_attention_semantics, "anchor_relative"
                )
                self.assertEqual(
                    reloaded.dflash_config["sliding_window_non_causal"], non_causal
                )
                model = _wrapper(
                    restored, head=original.lm_head, embedding=original.embed_tokens
                )
                torch.testing.assert_close(hidden(model), expected, rtol=0, atol=0)
                with mock.patch.object(
                    model, "_sample_anchor_positions", return_value=(anchors, keep)
                ):
                    loss, _, _ = model(**inputs)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for parameter in (
                    restored.candidate_selector.hidden_projection.weight,
                    restored.layers[0].attention_conv.kernel_projection.weight,
                ):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.norm().item(), 0)
