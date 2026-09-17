"""Loopback-only owned teacher HTTP service, using the spec_capture wire format.

Run ``python -m dspark.inference.teacher_server --help`` for launch flags.
Requests capture prompt features only; this is not a general generation API.
The synchronous server serializes model/store access and bounds HTTP bodies,
batch sizes and prompt lengths. CPU tests do not certify vLLM GPU parity.
"""

import argparse
import ipaddress
import json
import logging
import os
import signal
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from dspark.inference.capture_sink import CaptureSink

logger = logging.getLogger(__name__)


class TeacherService:
    def __init__(
        self,
        *,
        teacher,
        sink,
        max_model_len=4096,
        max_batch_size=16,
        backend="transformers",
        target_model="unknown",
        target_revision="unknown",
    ):
        if (
            type(max_model_len) is not int
            or max_model_len < 2
            or type(max_batch_size) is not int
            or max_batch_size < 1
        ):
            raise ValueError("invalid teacher request limits")
        self.teacher, self.sink = teacher, sink
        self.max_model_len, self.max_batch_size = max_model_len, max_batch_size
        self.identity = dict(
            backend=backend, target_model=target_model, target_revision=target_revision
        )

    def generate(self, body):
        if not isinstance(body, dict) or set(body) - {
            "input_ids",
            "spec_capture",
            "extra_key",
            "sampling_params",
        }:
            raise ValueError(
                "only input_ids/spec_capture/extra_key/sampling_params are supported"
            )
        ids, specs = body.get("input_ids"), body.get("spec_capture")
        if (
            not isinstance(ids, list)
            or not ids
            or not isinstance(specs, list)
            or len(ids) != len(specs)
            or len(ids) > self.max_batch_size
        ):
            raise ValueError(
                "input_ids and spec_capture must be equally sized nonempty batches within limit"
            )
        if body.get("sampling_params") != {"temperature": 0.0, "max_new_tokens": 1}:
            raise ValueError(
                "teacher capture requires temperature=0 and max_new_tokens=1"
            )
        identities = set()
        for tokens, spec in zip(ids, specs):
            if (
                not isinstance(tokens, list)
                or not 1 <= len(tokens) < self.max_model_len
                or any(type(t) is not int or t < 0 for t in tokens)
            ):
                raise ValueError(
                    "prompt must contain bounded nonnegative integer tokens"
                )
            self.sink.validate(spec, len(tokens))
            identity = (spec["store_id"], spec["sample_id"], spec["gen"])
            if identity in identities:
                raise ValueError("duplicate capture identity in batch")
            identities.add(identity)
        rows = []
        for tokens, spec in zip(ids, specs):
            try:
                input_ids = torch.tensor([tokens], dtype=torch.long)
                mask = torch.ones_like(input_ids)
                captured = self.teacher.capture(
                    input_ids=input_ids, attention_mask=mask, loss_mask=mask
                )
                if captured.input_ids.detach().cpu().tolist() != [tokens]:
                    raise ValueError("teacher returned mismatched input token IDs")
                config = (
                    getattr(self.teacher, "config", None) or self.teacher._model.config
                )
                for tensor, width in (
                    (
                        captured.hidden_states,
                        config.hidden_size * len(self.teacher.capture_layers),
                    ),
                    (captured.last_hidden_states, config.hidden_size),
                ):
                    if tuple(tensor.shape) != (
                        1,
                        len(tokens),
                        width,
                    ) or tensor.dtype not in (
                        torch.float16,
                        torch.bfloat16,
                        torch.float32,
                    ):
                        raise ValueError(
                            "teacher returned mismatched hidden-state shape or dtype"
                        )
                result = self.sink.put_sample(
                    spec,
                    aux=captured.hidden_states[0],
                    last_hidden=captured.last_hidden_states[0],
                )
                result.update(self.identity)
            except Exception as exc:
                logger.exception("teacher capture failed")
                result = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "sample_id": spec["sample_id"],
                }
            rows.append({"meta_info": {"spec_capture": result}})
        return rows

    def close(self):
        close = getattr(self.teacher, "close", None)
        if close is not None:
            close()


def make_server(
    service, *, host="127.0.0.1", port=30000, max_body_bytes=8 * 1024 * 1024
):
    address = ipaddress.ip_address(host)
    if not address.is_loopback or address.version != 4:
        raise ValueError("teacher HTTP service requires an IPv4 loopback address")
    if type(max_body_bytes) is not int or max_body_bytes < 1:
        raise ValueError("max_body_bytes must be positive")

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def _reply(self, status, payload):
            data = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._reply(
                200 if self.path == "/health" else 404,
                (
                    {"healthy": True, **service.identity}
                    if self.path == "/health"
                    else {"error": "not found"}
                ),
            )

        def do_POST(self):
            if self.path != "/generate":
                self._reply(404, {"error": "not found"})
                return
            if (
                self.headers.get("Origin")
                or self.headers.get_content_type() != "application/json"
                or self.headers.get("Transfer-Encoding")
            ):
                self._reply(
                    400,
                    {"error": "requires local application/json with Content-Length"},
                )
                return
            try:
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) != 1:
                    raise ValueError("requires one Content-Length")
                length = int(lengths[0])
                if not 0 < length <= max_body_bytes:
                    self._reply(413, {"error": "body exceeds limit or is empty"})
                    return
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("incomplete request body")
                body = json.loads(raw)
                result = service.generate(body)
            except (ValueError, TypeError, OverflowError, RecursionError) as exc:
                self._reply(400, {"error": str(exc)})
                return
            self._reply(200, result)

        def log_message(self, format, *args):
            logger.info(format, *args)

    return HTTPServer((host, port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-backend", choices=("transformers", "vllm"), required=True
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--capture-method", choices=("eagle3", "dflash", "dspark"), default="eagle3"
    )
    parser.add_argument("--aux-layer-ids", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--max-body-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args(argv)
    address = ipaddress.ip_address(args.host)
    if not address.is_loopback or address.version != 4:
        parser.error("teacher host must be IPv4 loopback")
    if (
        args.max_model_len < 2
        or args.max_batch_size < 1
        or args.max_body_bytes < 1
        or not 0 <= args.port <= 65535
    ):
        parser.error("invalid teacher HTTP limits or port")
    store_id = os.environ.get("DISAGG_STORE_ID")
    if not store_id:
        parser.error("DISAGG_STORE_ID must identify the permitted capture store")
    common = dict(
        revision=args.revision,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=getattr(torch, args.dtype),
    )
    if args.target_backend == "transformers":
        from dspark.offline_capture.transformers import OfflineTransformersCapture

        teacher = OfflineTransformersCapture.from_pretrained(
            args.model_path,
            **common,
            device=args.device,
            attn_implementation=args.attn_implementation,
        )
    else:
        from dspark.offline_capture.vllm import OfflineVLLMCapture

        teacher = OfflineVLLMCapture.from_pretrained(
            args.model_path,
            **common,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    try:
        teacher.set_capture_layers(
            args.aux_layer_ids, capture_method=args.capture_method
        )
        from dspark.runtime.data_plane.mooncake_store import MooncakeFeatureStore

        setup = {
            "local_hostname": os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "localhost"),
            "metadata_server": os.environ.get(
                "MOONCAKE_METADATA_SERVER", "http://localhost:8080/metadata"
            ),
            "global_segment_size": int(
                os.environ.get("MOONCAKE_GLOBAL_SEGMENT_SIZE", 1 << 30)
            ),
            "local_buffer_size": int(
                os.environ.get("MOONCAKE_LOCAL_BUFFER_SIZE", 1 << 30)
            ),
            "protocol": os.environ.get("MOONCAKE_PROTOCOL", "tcp"),
            "rdma_devices": os.environ.get("MOONCAKE_RDMA_DEVICES", ""),
            "master_server_addr": os.environ.get(
                "MOONCAKE_MASTER_SERVER_ADDR", "localhost:50051"
            ),
        }
        transport = MooncakeFeatureStore(store_id=store_id, setup_kwargs=setup)
        sink = CaptureSink(transport, aux_layer_ids=args.aux_layer_ids)
        model_config = getattr(teacher, "config", None) or teacher._model.config
        resolved_revision = (
            getattr(model_config, "_commit_hash", None) or args.revision or "unknown"
        )
        service = TeacherService(
            teacher=teacher,
            sink=sink,
            max_model_len=args.max_model_len,
            max_batch_size=args.max_batch_size,
            backend=args.target_backend,
            target_model=args.model_path,
            target_revision=resolved_revision,
        )
        with make_server(
            service, host=args.host, port=args.port, max_body_bytes=args.max_body_bytes
        ) as server:
            logging.basicConfig(level=logging.INFO)
            logger.info(
                "owned %s teacher listening on %s:%s",
                args.target_backend,
                args.host,
                args.port,
            )
            server.serve_forever()
    finally:
        close = getattr(teacher, "close", None)
        if close is not None:
            close()


def _terminate(signum, frame):
    # Unwind HTTP/model and in-flight sink writes on managed runtime shutdown.
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _terminate)
    main()
