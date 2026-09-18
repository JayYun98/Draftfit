"""Single-device, text-only Hugging Face teacher for offline feature capture."""

from __future__ import annotations

import torch

from draftfit.inference.capture import TeacherCaptureBatch as OfflineCaptureBatch


class OfflineTransformersCapture:
    """Capture Llama/Qwen decoder block outputs before final normalization.

    SGLang's dense EAGLE/DFlash hooks capture the residual stream entering the
    next block. HF's final hidden_states entry is already normalized, so block
    hooks are necessary to preserve the same boundary for a last-layer tap.
    This does not certify numerical parity with a GPU inference backend.
    """

    def __init__(self, model):
        self._model = model.base_model
        if self._model.config.model_type not in {"llama", "qwen2", "qwen3"}:
            raise ValueError(
                "Transformers offline capture supports llama, qwen2, qwen3 text decoders only"
            )
        if not hasattr(self._model, "layers") or not hasattr(self._model, "norm"):
            raise ValueError("target must expose decoder layers and final norm")
        self._model.requires_grad_(False).eval()
        self.capture_layers = None
        self.capture_method = "eagle3"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        revision=None,
        trust_remote_code=False,
        torch_dtype=None,
        device="cpu",
        cache_dir=None,
        attn_implementation="sdpa",
    ):
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
        )
        return cls(model.to(device))

    def set_capture_layers(self, layer_ids=None, *, capture_method="eagle3"):
        if capture_method not in {"eagle3", "dflash", "dspark"}:
            raise ValueError("capture_method must be eagle3, dflash, or dspark")
        depth = len(self._model.layers)
        if layer_ids is None and capture_method == "eagle3":
            layer_ids = [1, depth // 2 - 1, depth - 4]
        layers = list(layer_ids or [])
        if not layers or any(type(i) is not int or not 0 <= i < depth for i in layers):
            raise ValueError(
                f"capture layers must be integer indices within target depth {depth}"
            )
        if len(set(layers)) != len(layers):
            raise ValueError("capture layers must be unique")
        if layers != sorted(layers):
            raise ValueError(
                "capture layers must be ascending to match SGLang feature order"
            )
        if capture_method == "eagle3" and len(layers) != 3:
            raise ValueError("eagle3 requires exactly three capture layers")
        self.capture_layers = layers
        self.capture_method = capture_method

    @torch.no_grad()
    def capture(self, *, input_ids, attention_mask, loss_mask):
        if self.capture_layers is None:
            raise ValueError("set_capture_layers must be called before capture")
        if input_ids.ndim != 2 or not all(input_ids.shape):
            raise ValueError("input_ids must have nonempty [batch, sequence] shape")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be integer token IDs")
        for name, mask in (
            ("attention_mask", attention_mask),
            ("loss_mask", loss_mask),
        ):
            if mask.shape != input_ids.shape or not torch.all(
                (mask == 0) | (mask == 1)
            ):
                raise ValueError(f"{name} must be binary and match input_ids shape")
        if not torch.all(attention_mask.bool().any(dim=1)):
            raise ValueError("each sequence must contain an attended token")
        active = attention_mask.bool()
        inside = active.cumsum(-1).bool() & active.flip(-1).cumsum(-1).bool().flip(-1)
        if torch.any(inside & ~active):
            raise ValueError(
                "attention_mask must describe contiguous tokens with left or right padding"
            )
        device = self._model.get_input_embeddings().weight.device
        ids = input_ids.to(device)
        attention = attention_mask.to(device)
        loss = loss_mask.to(device)
        if torch.any(loss.bool() & ~attention.bool()):
            raise ValueError("loss_mask cannot select padding tokens")
        if torch.any(ids < 0) or torch.any(
            ids >= self._model.get_input_embeddings().num_embeddings
        ):
            raise ValueError("input_ids must be inside the target vocabulary")
        states = {}
        handles = []

        def save_layer(index):
            def save(module, args, output):
                states[index] = (
                    (output[0] if isinstance(output, tuple) else output)
                    .detach()
                    .clone()
                )

            return save

        try:
            for index in self.capture_layers:
                handles.append(
                    self._model.layers[index].register_forward_hook(save_layer(index))
                )
            self._model.eval()
            output = self._model(
                input_ids=ids,
                attention_mask=attention,
                position_ids=(attention.long().cumsum(-1) - 1).clamp_min(0),
                use_cache=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        last = output.last_hidden_state
        expected = (*ids.shape, self._model.config.hidden_size)
        captured = [states[i] for i in self.capture_layers]
        if any(tuple(state.shape) != expected for state in [*captured, last]):
            raise ValueError("target returned incompatible hidden-state shapes")
        if any(not torch.isfinite(state).all() for state in [*captured, last]):
            raise ValueError("target returned non-finite hidden states")
        return OfflineCaptureBatch(
            hidden_states=torch.cat(captured, dim=-1),
            last_hidden_states=last.detach(),
            input_ids=ids,
            attention_mask=attention,
            loss_mask=loss,
        )
