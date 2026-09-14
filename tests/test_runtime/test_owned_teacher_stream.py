"""Real CPU teacher/HTTP capture and trainer tensors, with a fake RDMA backend."""

import dataclasses
import tempfile
import threading
import unittest
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.common.dflash_family_model import OnlineDFlash2Model
from specforge.inference.adapters.server_capture import TeacherServerCaptureAdapter
from specforge.inference.capture import CaptureConfig, CaptureMismatchError
from specforge.inference.capture_sink import CaptureSink
from specforge.inference.teacher_server import TeacherService, make_server
from specforge.launch import build_disagg_online_producer
from specforge.modeling.draft.dflash2 import DFlash2DraftModel
from specforge.offline_capture.transformers import OfflineTransformersCapture
from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel
from tests.test_dflash2_integration import tiny_config
from tests.test_runtime.test_server_capture import (
    _capture_schema,
    _FakeMooncakeStore,
    _task,
)


class OwnedTeacherStreamTest(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(17)
        self.teacher = (
            Qwen3ForCausalLM(
                Qwen3Config(
                    hidden_size=32,
                    intermediate_size=64,
                    vocab_size=64,
                    num_hidden_layers=3,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=8,
                    max_position_embeddings=64,
                    tie_word_embeddings=False,
                )
            )
            .eval()
            .requires_grad_(False)
        )
        capture = OfflineTransformersCapture(self.teacher)
        capture.set_capture_layers([0, 2], capture_method="dflash")
        self.backend = _FakeMooncakeStore()
        self.store = MooncakeFeatureStore(store=self.backend, store_id="run0")
        service = TeacherService(
            teacher=capture,
            sink=CaptureSink(self.store, aux_layer_ids=[0, 2]),
            max_model_len=64,
            max_batch_size=4,
            backend="transformers",
            target_model="tiny-fixture",
            target_revision="local",
        )
        self.addCleanup(service.close)
        server = make_server(service, host="127.0.0.1", port=0)
        self.addCleanup(server.server_close)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)
        self.adapter = TeacherServerCaptureAdapter(
            f"http://127.0.0.1:{server.server_address[1]}",
            self.store,
            run_id="run0",
            algorithm="dflash2",
            schema=_capture_schema("dflash2"),
            backend="transformers",
            timeout_s=5,
            target_model_version="tiny-fixture",
            target_revision="local",
        )
        self.contract = CaptureConfig.from_strategy(
            required_features={"input_ids", "loss_mask", "hidden_states"},
            aux_hidden_state_layer_ids=(0, 2),
            target_repr=None,
            target_hidden_size=32,
            target_vocab_size=64,
            draft_vocab_size=64,
        )

    def test_live_teacher_channel_loader_and_finite_draft_update(self):
        registration = builtin_algorithm_registry().resolve("dflash2")
        tasks = [_task(0, 12), _task(1, 16)]
        with tempfile.TemporaryDirectory() as directory:
            channel = StreamingRefChannel(str(Path(directory) / "refs.jsonl"))
            channel.publish_consumer_quantum(1)
            _, drive = build_disagg_online_producer(
                algorithm=registration,
                feature_source=self.adapter,
                prompts=[dataclasses.asdict(task) for task in tasks],
                feature_store=self.store,
                channel=channel,
                run_id="run0",
                target_hidden_size=32,
                target_vocab_size=64,
                draft_vocab_size=64,
                target_repr=None,
                aux_hidden_state_layer_ids=(0, 2),
            )
            self.assertEqual(drive(), 2)
            self.assertTrue(channel.is_closed())
            refs = channel.poll()
            self.assertEqual(len(refs), 2)
            self.assertTrue(all(ref.metadata["generation"] == 1 for ref in refs))
            provider = registration.providers.server_streaming_for("text")
            loader = FeatureDataLoader(
                self.store,
                refs=refs,
                batch_size=2,
                strategy="dflash2",
                collate_fn=provider.build_collator(),
                gc_interval_s=None,
            )
            try:
                (batch,) = list(loader)
            finally:
                loader.close()
            self.assertEqual(tuple(batch.tensors["hidden_states"].shape), (2, 16, 64))
            # Compare real server output against the same frozen teacher locally.
            expected = OfflineTransformersCapture(self.teacher)
            expected.set_capture_layers([0, 2], capture_method="dflash")
            task_by_id = {f"run0:{task.task_id}": task for task in tasks}
            for row, sample_id in enumerate(batch.sample_ids):
                task = task_by_id[sample_id]
                ids = torch.tensor([task.payload["input_ids"]])
                result = expected.capture(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    loss_mask=torch.tensor([task.payload["loss_mask"]]),
                )
                self.assertTrue(
                    torch.allclose(
                        batch.tensors["hidden_states"][row, : ids.shape[1]],
                        result.hidden_states[0],
                        atol=1e-6,
                        rtol=1e-5,
                    )
                )
            draft = DFlash2DraftModel(
                tiny_config(target_num_hidden_layers=3, target_layer_ids=[0, 2])
            )
            wrapper = OnlineDFlash2Model(
                draft_model=draft,
                target_lm_head=self.teacher.lm_head,
                target_embed_tokens=self.teacher.get_input_embeddings(),
                mask_token_id=63,
                block_size=4,
                num_anchors=2,
                anchor_sampling="uniform",
                attention_backend="sdpa",
            )
            watched = draft.candidate_selector.hidden_projection.weight
            before = watched.detach().clone()
            optimizer = torch.optim.AdamW(draft.parameters(), lr=1e-3)
            loss, _, _ = wrapper(**batch.tensors)
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            self.assertTrue(torch.isfinite(watched.grad).all().item())
            self.assertGreater(watched.grad.norm().item(), 0)
            optimizer.step()
            self.assertFalse(torch.equal(watched, before))
            channel.mark_consumed(2)
            self.store.drain_pending_removals(retry_interval_s=0)
            self.assertFalse(self.backend._d)

    def test_lost_http_response_retry_and_terminal_cleanup(self):
        post = self.adapter.post_fn
        requests = []

        def lose_response(url, json_body, timeout):
            requests.append(json_body["spec_capture"][0])
            result = post(url, json_body=json_body, timeout=timeout)
            if len(requests) in (1, 3):
                raise ConnectionError("lost after actual HTTP write")
            return result

        self.adapter.post_fn = lose_response
        task = _task(0, 12)
        with self.assertRaisesRegex(ConnectionError, "lost after actual"):
            self.adapter.produce_refs([task], capture=self.contract)
        initial_keys = set(self.backend._d)
        self.assertTrue(initial_keys)
        (ref,) = self.adapter.produce_refs(
            [dataclasses.replace(task, attempt=1)], capture=self.contract
        )
        self.assertEqual(ref.metadata["generation"], 1)
        self.assertEqual(set(self.backend._d), initial_keys)
        self.assertEqual([request["gen"] for request in requests], [1, 1])
        self.assertTrue(requests[1]["replace"])
        self.assertEqual(self.store.discard_external_attempts(), 0)
        self.store.abort(ref.sample_id, reason="test-complete")
        self.assertFalse(self.backend._d)
        with self.assertRaises(ConnectionError):
            self.adapter.produce_refs([_task(1, 12)], capture=self.contract)
        self.assertEqual(
            self.store.discard_external_attempts(reason="terminal-test"), 1
        )
        self.store.drain_pending_removals(retry_interval_s=0)
        self.assertFalse(self.backend._d)

    def test_wrong_teacher_identity_is_not_adopted_and_can_be_reclaimed(self):
        self.adapter.target_model_version = "different-target"
        with self.assertRaisesRegex(CaptureMismatchError, "teacher identity mismatch"):
            self.adapter.produce_refs([_task(0, 12)], capture=self.contract)
        self.assertTrue(self.backend._d)
        self.assertEqual(self.store.health()["provisional_external"], 1)
        self.assertEqual(
            self.store.discard_external_attempts(reason="wrong-teacher"), 1
        )
        self.store.drain_pending_removals(retry_interval_s=0)
        self.assertFalse(self.backend._d)


if __name__ == "__main__":
    unittest.main()
