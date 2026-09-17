"""TP1 text capture using vLLM 0.22.1's official hidden-state connector.

This adapter has CPU boundary tests, not certified GPU numerical parity.
Reference: vllm v0.22.1 examples/features/speculative_decoding/
extract_hidden_states_offline.py and EagleModelMixin.
Workers use mp/spawn to isolate an existing parent process group. Launch callers
under an ``if __name__ == '__main__'`` guard; GPU execution remains unvalidated.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import tempfile
from pathlib import Path

import torch

from dspark.inference.capture import TeacherCaptureBatch as OfflineCaptureBatch


def _load_norm(model, config, revision, cache_dir=None):
    """Read only final RMSNorm from a safetensors checkpoint, never a full teacher."""
    from safetensors import safe_open
    from transformers.utils import cached_file

    options = dict(
        revision=revision,
        cache_dir=cache_dir,
        _raise_exceptions_for_missing_entries=False,
    )
    index = cached_file(model, "model.safetensors.index.json", **options)
    filename = "model.safetensors"
    if index is not None:
        filename = json.loads(Path(index).read_text())["weight_map"][
            "model.norm.weight"
        ]
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".safetensors")
        ):
            raise ValueError("invalid norm checkpoint shard filename")
    path = cached_file(model, filename, **options)
    if path is None:
        raise ValueError("vLLM offline capture requires safetensors model.norm.weight")
    with safe_open(path, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor("model.norm.weight")
    if (
        weight.shape != (config.hidden_size,)
        or not weight.is_floating_point()
        or not torch.isfinite(weight).all()
    ):
        raise ValueError("invalid final norm weight")
    name = {"llama": "LlamaRMSNorm", "qwen2": "Qwen2RMSNorm", "qwen3": "Qwen3RMSNorm"}[
        config.model_type
    ]
    module = importlib.import_module(
        f"transformers.models.{config.model_type}.modeling_{config.model_type}"
    )
    norm = getattr(module, name)(config.hidden_size, eps=config.rms_norm_eps)
    norm.load_state_dict({"weight": weight})
    return norm.eval().requires_grad_(False)


class OfflineVLLMCapture:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        revision=None,
        trust_remote_code=False,
        torch_dtype=None,
        max_model_len=None,
        gpu_memory_utilization=0.8,
        cache_dir=None,
    ):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        if config.model_type not in {"llama", "qwen2", "qwen3"} or getattr(
            config, "quantization_config", None
        ):
            raise ValueError(
                "vLLM offline capture supports unquantized llama/qwen2/qwen3 text models only"
            )
        if torch_dtype not in (None, torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("capture dtype must be float16, bfloat16, or float32")
        if max_model_len is not None and (
            type(max_model_len) is not int or max_model_len < 2
        ):
            raise ValueError("max_model_len must be an integer >= 2")
        if (
            type(gpu_memory_utilization) not in (int, float)
            or not 0 < gpu_memory_utilization < 1
        ):
            raise ValueError("gpu_memory_utilization must be between 0 and 1")
        if not math.isfinite(config.rms_norm_eps) or config.rms_norm_eps <= 0:
            raise ValueError("invalid final norm epsilon")
        instance = cls()
        instance.config = config
        revision = getattr(config, "_commit_hash", None) or revision
        instance._norm = _load_norm(
            pretrained_model_name_or_path, config, revision, cache_dir
        )
        instance._options = dict(
            model=str(pretrained_model_name_or_path),
            revision=revision,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype or "auto",
            gpu_memory_utilization=gpu_memory_utilization,
            download_dir=cache_dir,
        )
        if max_model_len is not None:
            instance._options["max_model_len"] = max_model_len
        instance._llm = None
        instance._directory = None
        instance.capture_layers = None
        return instance

    def set_capture_layers(self, layer_ids=None, *, capture_method="eagle3"):
        if int(os.environ.get("WORLD_SIZE", "1")) != 1 or (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() != 1
        ):
            raise ValueError("vLLM offline capture requires a single process (TP1/PP1)")
        if self._llm is not None:
            raise ValueError("capture layers cannot change after vLLM initialization")
        if capture_method not in {"eagle3", "dflash", "dspark"}:
            raise ValueError("capture_method must be eagle3, dflash, or dspark")
        depth = self.config.num_hidden_layers
        if layer_ids is None and capture_method == "eagle3":
            layer_ids = [1, depth // 2 - 1, depth - 4]
        layers = list(layer_ids or [])
        if not layers or any(type(i) is not int or not 0 <= i < depth for i in layers):
            raise ValueError(
                "capture layers must be integer indices within target depth"
            )
        if len(set(layers)) != len(layers) or (
            capture_method == "eagle3" and len(layers) != 3
        ):
            raise ValueError("capture layers must be unique; eagle3 requires three")
        if layers != sorted(layers):
            raise ValueError("capture layers must be in ascending execution order")
        method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
        if method not in (None, "spawn"):
            raise ValueError("vLLM capture requires VLLM_WORKER_MULTIPROC_METHOD=spawn")
        import vllm
        from vllm.config.kv_transfer import KVTransferConfig

        if vllm.__version__ != "0.22.1":
            raise ValueError("vLLM offline capture requires vllm==0.22.1")
        # vLLM emits in traversal order, regardless of requested slot ordering.
        slots = sorted({i + 1 for i in layers} | {depth})
        self._directory = tempfile.TemporaryDirectory(prefix="specforge-vllm-")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        try:
            self._llm = vllm.LLM(
                **self._options,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                distributed_executor_backend="mp",
                enforce_eager=True,
                enable_prefix_caching=False,
                enable_chunked_prefill=False,
                skip_tokenizer_init=True,
                speculative_config={
                    "method": "extract_hidden_states",
                    "num_speculative_tokens": 1,
                    "draft_model_config": {
                        "hf_config": {"eagle_aux_hidden_state_layer_ids": slots}
                    },
                },
                kv_transfer_config=KVTransferConfig(
                    kv_connector="ExampleHiddenStatesConnector",
                    kv_role="kv_producer",
                    kv_connector_extra_config={
                        "shared_storage_path": self._directory.name
                    },
                ),
            )
        except BaseException:
            self._directory.cleanup()
            self._directory = None
            raise
        finally:
            if method is None:
                os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
        self._slots = slots
        self.capture_layers = layers
        self.capture_method = capture_method

    def _checked_path(self, raw):
        if not isinstance(raw, str) or not raw:
            raise ValueError("missing hidden_states_path")
        path = Path(raw)
        root = Path(self._directory.name).resolve()
        if path.suffix != ".safetensors" or path.parent.resolve() != root:
            raise ValueError(
                "hidden-state file must be directly inside private capture directory"
            )
        for entry in (path, Path(raw + ".lock")):
            if entry.is_symlink() or (entry.exists() and not entry.is_file()):
                raise ValueError(
                    "hidden-state and lock paths must be regular files, not symlinks"
                )
        return str(path.resolve())

    @torch.no_grad()
    def capture(self, *, input_ids, attention_mask, loss_mask):
        if self._llm is None:
            raise ValueError("set_capture_layers must be called before capture")
        if (
            input_ids.ndim != 2
            or not all(input_ids.shape)
            or input_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("input_ids must be nonempty integer [batch, sequence]")
        for mask in (attention_mask, loss_mask):
            if mask.shape != input_ids.shape or not torch.all(
                (mask == 0) | (mask == 1)
            ):
                raise ValueError(
                    "attention/loss masks must be binary and match input_ids"
                )
        ids, attention, loss = (
            t.detach().cpu() for t in (input_ids, attention_mask, loss_mask)
        )
        if torch.any(ids < 0) or torch.any(ids >= self.config.vocab_size):
            raise ValueError("input_ids outside target vocabulary")
        if not attention.bool().any(dim=1).all() or torch.any(
            loss.bool() & ~attention.bool()
        ):
            raise ValueError("empty prompt or loss_mask selecting padding")
        for mask in attention:
            positions = mask.nonzero().flatten()
            if torch.any(positions[1:] - positions[:-1] != 1):
                raise ValueError("attention_mask cannot contain interior padding holes")
        prompts = [
            {"prompt_token_ids": row[mask.bool()].tolist()}
            for row, mask in zip(ids, attention)
        ]
        if "max_model_len" in self._options and any(
            len(p["prompt_token_ids"]) + 1 > self._options["max_model_len"]
            for p in prompts
        ):
            raise ValueError("prompt plus one generation token exceeds max_model_len")
        from vllm import SamplingParams
        from vllm.distributed.kv_transfer.kv_connector.v1 import (
            example_hidden_states_connector as connector,
        )

        outputs = self._llm.generate(
            prompts,
            SamplingParams(max_tokens=1, temperature=0, detokenize=False),
            use_tqdm=False,
        )
        if len(outputs) != len(prompts):
            raise ValueError("vLLM returned wrong response count")
        seen_ids, seen_paths = set(), set()
        hidden = last = None
        for row, (output, prompt) in enumerate(zip(outputs, prompts)):
            expected_ids = prompt["prompt_token_ids"]
            request_id = getattr(output, "request_id", None)
            if (
                not isinstance(request_id, str)
                or request_id in seen_ids
                or not getattr(output, "finished", False)
            ):
                raise ValueError(
                    "vLLM response unfinished or duplicate/missing request identity"
                )
            seen_ids.add(request_id)
            if output.prompt_token_ids != expected_ids:
                raise ValueError("vLLM response prompt/order mismatch")
            params = getattr(output, "kv_transfer_params", None)
            path = self._checked_path(
                params.get("hidden_states_path") if isinstance(params, dict) else None
            )
            if path in seen_paths:
                raise ValueError("duplicate hidden-state file")
            seen_paths.add(path)
            payload = connector.load_hidden_states(path)
            try:
                tokens, states = payload["token_ids"], payload["hidden_states"]
                if (
                    tokens.dtype not in (torch.int32, torch.int64)
                    or tokens.shape != (len(expected_ids),)
                    or tokens.tolist() != expected_ids
                ):
                    raise ValueError(
                        "hidden-state token IDs differ from submitted prompt"
                    )
                expected_shape = (
                    len(expected_ids),
                    len(self._slots),
                    self.config.hidden_size,
                )
                if (
                    states.shape != expected_shape
                    or states.device.type != "cpu"
                    or states.dtype
                    not in (torch.float16, torch.bfloat16, torch.float32)
                    or not torch.isfinite(states).all()
                ):
                    raise ValueError(
                        "invalid hidden-state shape, dtype, or finite values"
                    )
                requested_dtype = self._options["dtype"]
                if requested_dtype != "auto" and states.dtype != requested_dtype:
                    raise ValueError("hidden-state dtype differs from requested dtype")
                if hidden is None:
                    hidden = states.new_zeros(
                        (*ids.shape, len(self.capture_layers) * self.config.hidden_size)
                    )
                    last = states.new_zeros((*ids.shape, self.config.hidden_size))
                elif states.dtype != hidden.dtype:
                    raise ValueError("inconsistent hidden-state dtype across responses")
                self._norm.to(dtype=states.dtype, device=states.device)
                normalized = self._norm(
                    states[:, self._slots.index(self.config.num_hidden_layers)]
                )
                if not torch.isfinite(normalized).all():
                    raise ValueError("nonfinite normalized final hidden states")
                hidden[row, attention[row].bool()] = torch.cat(
                    [states[:, self._slots.index(i + 1)] for i in self.capture_layers],
                    dim=-1,
                )
                last[row, attention[row].bool()] = normalized
            finally:
                connector.cleanup_hidden_states(self._checked_path(path))
        return OfflineCaptureBatch(hidden, last, ids, attention, loss)

    def close(self):
        # Release the engine before its private connector directory.
        if self._llm is not None:
            self._llm.llm_engine.engine_core.shutdown()
        self._llm = None
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
