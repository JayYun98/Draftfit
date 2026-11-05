"""Resident causal generation with bounded, empirical CUDA batch selection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from draftfit.training.hardware_tuning import (
    candidates,
    choose,
    cuda_spec,
    generation_limit,
)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


class _GenerationDeadline(TimeoutError):
    """Internal deadline or incomplete probe, distinct from model errors."""


class AdaptiveGenerationBatcher:
    """Generate continuations in order; receipts contain no prompt text or tokens.

    Supply immutable model/revision identity to enable cross-instance receipt reuse.
    Automatic selection requires CUDA. Fixed batching also works on CPU.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        batch_size=1,
        auto_batch=False,
        max_batch_size=128,
        tuning_seconds=60,
        memory_fraction=0.9,
        deadline_unix=None,
        receipt_path=None,
        identity=None,
        generation_kwargs=None,
    ):
        for value in (batch_size, max_batch_size):
            if type(value) is not int or value < 1:
                raise ValueError("batch sizes must be positive integers")
        if batch_size > max_batch_size:
            raise ValueError("batch_size exceeds max_batch_size")
        for value in (tuning_seconds, memory_fraction):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("tuning bounds must be finite numbers")
        if tuning_seconds <= 0 or not 0 < memory_fraction <= 0.9:
            raise ValueError("positive tuning time and at least 10% headroom required")
        if deadline_unix is not None and (
            type(deadline_unix) not in (int, float) or not math.isfinite(deadline_unix)
        ):
            raise ValueError("deadline_unix must be finite")
        self.model, self.tokenizer = model, tokenizer
        self.device = model.device
        if auto_batch and self.device.type != "cuda":
            raise ValueError("auto_batch requires a CUDA model")
        if getattr(model.config, "is_encoder_decoder", False):
            raise ValueError("batcher requires a causal decoder model")
        self.batch_size, self.auto_batch = batch_size, auto_batch
        self.max_batch_size = max_batch_size
        self.tuning_seconds, self.memory_fraction = tuning_seconds, memory_fraction
        self.deadline_unix = deadline_unix
        self.receipt_path = Path(receipt_path) if receipt_path is not None else None
        self.identity = identity
        self.kwargs = dict(generation_kwargs or {})
        generation_config = getattr(model, "generation_config", None)
        self.kwargs.setdefault(
            "max_new_tokens", getattr(generation_config, "max_new_tokens", None) or 128
        )
        if (
            type(self.kwargs["max_new_tokens"]) is not int
            or self.kwargs["max_new_tokens"] < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        if (
            self.kwargs.get(
                "num_return_sequences",
                getattr(generation_config, "num_return_sequences", None) or 1,
            )
            != 1
        ):
            raise ValueError("exactly one continuation per prompt is required")
        if "max_length" in self.kwargs:
            raise ValueError("use max_new_tokens rather than max_length")
        self._contract = _digest(
            {
                "identity": identity,
                "model": model.config.to_dict(),
                "precision": str(model.dtype),
                "generation_config": (
                    generation_config.to_dict() if generation_config else None
                ),
                "generation_kwargs": self.kwargs,
                "tokenizer": {
                    "class": type(tokenizer).__name__,
                    "name": getattr(tokenizer, "name_or_path", ""),
                    "vocabulary": tokenizer.get_vocab(),
                    "special_tokens": tokenizer.special_tokens_map,
                    "backend": (
                        tokenizer.backend_tokenizer.to_str()
                        if hasattr(tokenizer, "backend_tokenizer")
                        else None
                    ),
                },
                "max_batch_size": max_batch_size,
                "memory_fraction": memory_fraction,
            }
        )
        self.evidence = {
            "version": 1,
            "contract": self._contract,
            "mode": "cuda" if self.device.type == "cuda" else "cpu",
            "status": "pending",
            "fallbacks": [],
        }
        self._selected = None

    def _check_deadline(self, end=None):
        end = self.deadline_unix if end is None else end
        if end is not None and time.time() >= end:
            raise _GenerationDeadline("generation deadline exceeded")

    @contextmanager
    def _rng(self):
        cpu = torch.get_rng_state()
        cuda = (
            torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None
        )
        try:
            yield
        finally:
            torch.set_rng_state(cpu)
            if cuda is not None:
                torch.cuda.set_rng_state(cuda, self.device)

    def _encode(self, prompts):
        tokenizer = self.tokenizer
        side, pad = tokenizer.padding_side, tokenizer.pad_token
        try:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is None:
                    raise ValueError("tokenizer needs a pad or EOS token")
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer(
                prompts,
                padding=True,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )
        finally:
            tokenizer.padding_side, tokenizer.pad_token = side, pad

    def _run(self, prompts, *, probe_end=None):
        self._check_deadline(probe_end)
        encoded = {k: v.to(self.device) for k, v in self._encode(prompts).items()}
        width = encoded["input_ids"].shape[1]
        kwargs = dict(self.kwargs)
        kwargs.setdefault(
            "pad_token_id",
            (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        )
        end = probe_end if probe_end is not None else self.deadline_unix
        self._check_deadline(end)
        if end is not None:
            kwargs["max_time"] = min(
                kwargs.get("max_time") or math.inf, max(0.001, end - time.time())
            )
        if probe_end is not None:
            kwargs.update(
                do_sample=False, num_beams=1, min_new_tokens=kwargs["max_new_tokens"]
            )
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            with torch.inference_mode():
                output = self.model.generate(**encoded, **kwargs)
            sequences = output.sequences if hasattr(output, "sequences") else output
            continuation = sequences[:, width:]
            if (
                probe_end is not None
                and continuation.shape[1] != kwargs["max_new_tokens"]
            ):
                raise _GenerationDeadline(
                    "generation probe did not complete its token budget"
                )
            self._check_deadline(end)
            if probe_end is not None:
                return None
            return self.tokenizer.batch_decode(continuation, skip_special_tokens=True)
        finally:
            for module, training in modes:
                module.training = training

    def _budget(self):
        spec = cuda_spec(torch, self.device)
        # Reserve headroom against total device capacity, including other processes.
        budget = max(
            0,
            torch.cuda.memory_reserved(self.device)
            + spec["free_bytes"]
            - spec["total_bytes"] * (1 - self.memory_fraction),
        )
        hardware = {k: v for k, v in spec.items() if k != "free_bytes"}
        return spec, hardware, min(budget, spec["total_bytes"] * self.memory_fraction)

    def _write(self):
        if self.receipt_path is None:
            return
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.receipt_path.parent, delete=False
            ) as stream:
                temporary = stream.name
                json.dump(self.evidence, stream, sort_keys=True, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, self.receipt_path)
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def _select(self, prompt, length):
        spec, hardware, budget = self._budget()
        upper = generation_limit(
            self.model.config,
            spec,
            length + self.kwargs["max_new_tokens"],
            self.max_batch_size,
        )
        saved = self.evidence if "hardware" in self.evidence else None
        if (
            saved is None
            and self.receipt_path is not None
            and self.identity is not None
        ):
            try:
                saved = json.loads(self.receipt_path.read_text())
            except (FileNotFoundError, ValueError):
                saved = None
        if (
            isinstance(saved, dict)
            and saved.get("version") == 1
            and saved.get("contract") == self._contract
            and saved.get("hardware") == hardware
            and type(saved.get("prompt_tokens")) is int
            and saved["prompt_tokens"] >= length
        ):
            self.evidence = saved
            ceiling = saved.get("fallback_ceiling")
            if type(ceiling) is int and ceiling > 0:
                upper = min(upper, ceiling)
            row = saved.get("selected", {})
            if (
                saved.get("status") == "ok"
                and isinstance(row, dict)
                and type(row.get("batch_size")) is int
                and 0 < row["batch_size"] <= upper
                and row.get("status") == "ok"
                and type(row.get("peak_bytes")) in (int, float)
                and 0 <= row["peak_bytes"] <= budget
            ):
                self.evidence = saved
                self.evidence["reused"] = True
                self._selected = row
                self.batch_size = row["batch_size"]
                return
        else:
            self.evidence.pop("fallback_ceiling", None)
        self._selected = None
        self.evidence.update(
            status="tuning",
            hardware=hardware,
            prompt_tokens=length,
            trials=[],
            reused=False,
        )
        self.evidence.pop("selected", None)
        self.evidence.pop("final_batch_size", None)
        self._write()
        end = min(
            self.deadline_unix if self.deadline_unix is not None else math.inf,
            time.time() + self.tuning_seconds,
        )
        try:
            for batch in candidates(upper):
                if time.time() >= end:
                    break
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(self.device)
                row = {
                    "batch_size": batch,
                    "memory_budget_bytes": budget,
                    "hardware": hardware,
                }
                try:
                    with self._rng():
                        torch.cuda.synchronize(self.device)
                        start = time.perf_counter()
                        self._run([prompt] * batch, probe_end=end)
                        torch.cuda.synchronize(self.device)
                        seconds = time.perf_counter() - start
                        self._check_deadline(end)
                        row.update(
                            status="ok",
                            samples_per_second=batch / seconds,
                            peak_bytes=torch.cuda.max_memory_reserved(self.device),
                        )
                except torch.cuda.OutOfMemoryError:
                    row["status"] = "oom"
                except _GenerationDeadline:
                    row["status"] = "timeout"
                finally:
                    torch.cuda.empty_cache()
                self.evidence["trials"].append(row)
                if row["status"] != "ok" or row["peak_bytes"] > budget:
                    break
            selected = choose(self.evidence["trials"], self.memory_fraction)
            self.batch_size = selected["batch_size"]
            self._selected = selected
            self.evidence.update(status="ok", selected=selected)
        except BaseException:
            self.evidence["status"] = "failed"
            raise
        finally:
            self._write()

    def generate(self, prompts: list[str]) -> list[str]:
        """Return one continuation per prompt, without truncating any input."""
        if not isinstance(prompts, list) or any(
            not isinstance(p, str) for p in prompts
        ):
            raise TypeError("prompts must be a list of strings")
        if not prompts:
            return []
        self._check_deadline()
        lengths = [
            len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompts
        ]
        longest = max(range(len(prompts)), key=lengths.__getitem__)
        config = (
            self.model.config.get_text_config()
            if hasattr(self.model.config, "get_text_config")
            else self.model.config
        )
        context = getattr(config, "max_position_embeddings", None)
        if (
            context is not None
            and lengths[longest] + self.kwargs["max_new_tokens"] > context
        ):
            raise ValueError(
                "prompt plus max_new_tokens exceeds model context capacity"
            )
        if self.auto_batch:
            self._select(prompts[longest], lengths[longest])
        output, offset = [], 0
        self.evidence["status"] = "generating"
        self.evidence.pop("final_batch_size", None)
        try:
            while offset < len(prompts):
                pending = prompts[offset : offset + self.batch_size]
                cpu = torch.get_rng_state()
                cuda = (
                    torch.cuda.get_rng_state(self.device)
                    if self.device.type == "cuda"
                    else None
                )
                oom = False
                try:
                    result = self._run(pending)
                except torch.cuda.OutOfMemoryError:
                    if self.device.type != "cuda":
                        raise
                    torch.set_rng_state(cpu)
                    torch.cuda.set_rng_state(cuda, self.device)
                    self.evidence["status"] = "invalidated"
                    self.evidence.pop("selected", None)
                    self._selected = None
                    self._write()
                    if len(pending) == 1:
                        self.evidence["failed_batch_size"] = 1
                        raise
                    previous = self.batch_size
                    self.batch_size = max(1, self.batch_size // 2)
                    self.evidence["fallback_ceiling"] = self.batch_size
                    self.evidence["fallbacks"].append(
                        {
                            "from_batch_size": previous,
                            "attempted_batch_size": len(pending),
                            "batch_size": self.batch_size,
                            "reason": "cuda_oom",
                        }
                    )
                    self._write()
                    oom = True
                if oom:
                    torch.cuda.empty_cache()
                    continue
                if len(result) != len(pending):
                    raise RuntimeError(
                        "model returned an unexpected number of continuations"
                    )
                output.extend(result)
                offset += len(pending)
            self.evidence.update(status="ok", final_batch_size=self.batch_size)
            return output
        except BaseException:
            self.evidence["status"] = "failed"
            raise
        finally:
            self._write()
