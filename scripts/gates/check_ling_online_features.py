"""Capture one real online Ling sample, then compare the same IDs offline.

Run both modes on GPU1 after training stops. Online needs the existing patched
GPU0 server and MOONCAKE_{LOCAL_HOSTNAME,METADATA_SERVER,MASTER_SERVER_ADDR}.
Offline loads the same local target on GPU1; it never starts a cloud instance.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

LAYERS = [3, 7, 11, 15, 19]


def compare_features(online, offline):
    report = {}
    if set(online) != set(offline):
        raise ValueError("online/offline feature names differ")
    for name in sorted(online):
        a, b = online[name].detach().cpu(), offline[name].detach().cpu()
        if a.shape != b.shape:
            raise ValueError(f"{name}: shape mismatch {a.shape} != {b.shape}")
        floating = a.is_floating_point() or b.is_floating_point()
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        equal = finite and (torch.allclose(a.float(), b.float(), atol=0.02, rtol=0.02) if floating else torch.equal(a, b))
        report[name] = {
            "passed": bool(equal), "shape": list(a.shape),
            "finite": finite,
            "max_abs_error": float((a.float() - b.float()).abs().max()) if finite else None,
            "online_sha256": hashlib.sha256(a.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
            "offline_sha256": hashlib.sha256(b.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["online", "offline"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:30000")
    parser.add_argument("--attention-backend", default="fa3")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("actual GPU feature gate cannot pass on CPU")
    from transformers import AutoConfig

    model = str(Path(args.model_path).resolve(strict=True))
    config = AutoConfig.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    if config.hidden_size != 1536 or config.vocab_size != 157184:
        raise ValueError("expected Ling tiny hidden_size=1536, vocab_size=157184")
    identity = {"model_path": model, "config_sha256": hashlib.sha256((Path(model) / "config.json").read_bytes()).hexdigest(), "layers": LAYERS, "attention_backend": args.attention_backend}
    destination = Path(args.output_dir)
    if args.mode == "online":
        from transformers import AutoTokenizer
        from specforge.algorithms.builtin import builtin_algorithm_registry
        from specforge.inference.adapters.server_capture import SGLangServerCaptureAdapter, ServerCaptureSchema
        from specforge.inference.capture import CaptureConfig
        from specforge.runtime.contracts import PromptTask, SampleRef
        from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore

        destination.mkdir(parents=True, exist_ok=False)
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
        ids = tokenizer.encode("Explain how a hash table resolves a collision with a short example.", add_special_tokens=True)
        if not 3 <= len(ids) <= 64:
            raise ValueError("gate prompt must tokenize to 3..64 tokens")
        loss = [0] + [1] * (len(ids) - 2) + [0]
        run = "ling-feature-gate-" + uuid.uuid4().hex
        store = MooncakeFeatureStore(store_id=run, setup_kwargs={
            "local_hostname": os.environ["MOONCAKE_LOCAL_HOSTNAME"],
            "metadata_server": os.environ["MOONCAKE_METADATA_SERVER"],
            "master_server_addr": os.environ["MOONCAKE_MASTER_SERVER_ADDR"],
            "protocol": os.environ.get("MOONCAKE_PROTOCOL", "tcp"),
            "rdma_devices": "", "global_segment_size": 1 << 28, "local_buffer_size": 1 << 28,
        })
        layout = builtin_algorithm_registry().resolve("dspark").providers.server_streaming_for("text").layout
        schema = ServerCaptureSchema(aux_feature=layout.aux_feature, last_hidden_feature=layout.last_hidden_feature, passthrough=layout.passthrough, attention_mask_feature=layout.attention_mask_feature)
        capture = CaptureConfig.from_strategy(
            required_features={"input_ids", "loss_mask", layout.aux_feature, layout.last_hidden_feature},
            aux_hidden_state_layer_ids=LAYERS, target_repr="hidden_state", target_hidden_size=1536, target_vocab_size=157184,
        )
        adapter = SGLangServerCaptureAdapter(args.server_url, store, run_id=run, algorithm="dspark", schema=schema)
        task = PromptTask(task_id="sample", run_id=run, source_id="feature-gate", payload={"input_ids": ids, "loss_mask": loss}, max_length=len(ids))
        (ref,) = adapter.produce_refs([task], capture=capture)
        if not isinstance(ref, SampleRef):
            raise RuntimeError(f"online capture failed: {ref}")
        features, handle = store.get(ref)
        try:
            saved = {name: tensor.detach().cpu().clone() for name, tensor in features.items()}
        finally:
            store.release(handle)
        if saved["input_ids"].tolist() != [ids]:
            raise ValueError("online capture changed requested token IDs")
        with (destination / "online.pt").open("xb") as output:
            torch.save({"identity": identity, "features": saved}, output)
        print(json.dumps({"online_captured": True, "tokens": len(ids), "shapes": {k: list(v.shape) for k, v in saved.items()}}))
        return

    from specforge.distributed import init_distributed
    from specforge.offline_capture.sglang import OfflineSGLangCapture

    saved = torch.load(destination / "online.pt", map_location="cpu", weights_only=True)
    if saved["identity"] != identity:
        raise ValueError("model path/config/layers/backend differ between captures")
    for key, value in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29678"}.items():
        os.environ.setdefault(key, value)
    init_distributed(tp_size=1)
    try:
        target = OfflineSGLangCapture.from_pretrained(
            model, torch_dtype=torch.bfloat16, trust_remote_code=True,
            attention_backend=args.attention_backend, linear_attn_backend="triton",
            linear_attn_verify_backend="triton", enable_deterministic_inference=True,
            mem_fraction_static=0.5, context_length=4103, max_running_requests=1,
            max_total_tokens=4103, disable_radix_cache=True,
        )
        target.set_capture_layers(LAYERS, capture_method="dspark")
        online = saved["features"]
        ids = online["input_ids"].cuda()
        with torch.no_grad():
            result = target.capture(input_ids=ids, attention_mask=torch.ones_like(ids), loss_mask=online["loss_mask"].cuda())
        offline = {"input_ids": result.input_ids, "loss_mask": result.loss_mask, "hidden_states": result.hidden_states, "target_last_hidden_states": result.last_hidden_states}
        features = compare_features(online, offline)
        report = {"passed": all(v["passed"] for v in features.values()), "atol": 0.02, "rtol": 0.02, "identity": identity, "features": features}
        with (destination / "comparison.json").open("x") as output:
            json.dump(report, output, indent=2)
        print(json.dumps(report))
        if not report["passed"]:
            raise AssertionError("online/offline tensor parity failed; see comparison.json")
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
