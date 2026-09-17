"""CPU contract tests for the optional composable FSDP2 train backend."""

from __future__ import annotations

import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.nn as nn


class _Composite(nn.Module):
    def __init__(self):
        super().__init__()
        self.draft_model = nn.Linear(3, 2)
        self.frozen_target = nn.Linear(3, 2)
        self.frozen_target.requires_grad_(False)

    def forward(self, x):
        return self.draft_model(x)


class _AuxDraft(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(3, 3)
        self.head = nn.Linear(3, 2)

    def forward(self, x):
        return self.body(x)

    def apply_logits_head(self, x):
        return self.head(x)


class FSDP2BackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if dist.is_available() and not dist.is_initialized():
            cls._store_dir = tempfile.mkdtemp(prefix="specforge_fsdp2_pg_")
            dist.init_process_group(
                "gloo",
                store=dist.FileStore(os.path.join(cls._store_dir, "store"), 1),
                rank=0,
                world_size=1,
            )
            cls._created_pg = True
        else:
            cls._created_pg = False

    @classmethod
    def tearDownClass(cls):
        if cls._created_pg:
            dist.destroy_process_group()

    def _backend(self, model, sharding_strategy="SHARD_GRAD_OP"):
        from torch.distributed.device_mesh import init_device_mesh

        from dspark.optimizer import BF16Optimizer
        from dspark.training.backend import FSDP2TrainingBackend, ParallelConfig

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        parallel = ParallelConfig(
            world_size=1,
            sharding_strategy=sharding_strategy,
            param_dtype=torch.float32,
            fsdp_process_group=dist.group.WORLD,
            dp_device_mesh=mesh,
        )
        backend = FSDP2TrainingBackend(
            parallel,
            optimizer_factory=lambda module: BF16Optimizer(
                module, lr=1e-2, total_steps=2
            ),
        )
        backend.prepare_model(model, optimizer_target=model.draft_model)
        return backend

    def test_shards_only_trainable_draft_and_roundtrips_state(self):
        torch.manual_seed(0)
        model = _Composite()
        backend = self._backend(model)
        self.assertEqual(backend.name, "fsdp2")
        self.assertTrue(hasattr(model.draft_model, "set_requires_gradient_sync"))
        self.assertFalse(hasattr(model.frozen_target, "set_requires_gradient_sync"))

        x = torch.randn(4, 3)
        loss = model(x).square().mean()
        self.assertTrue(torch.isfinite(loss))
        backend.backward(loss, is_boundary=True)
        grad_norm = backend.step()
        self.assertTrue(grad_norm is None or torch.isfinite(torch.as_tensor(grad_norm)))
        state = backend.state_dict()
        self.assertEqual(
            set(state["model"]), {"draft_model.weight", "draft_model.bias"}
        )
        self.assertIn("optimizer_state_dict", state["optimizer"])
        self.assertEqual(state["rng"]["device_type"], "cpu")

        restored = _Composite()
        backend2 = self._backend(restored)
        backend2.load_state_dict(state)
        for left, right in zip(
            model.draft_model.parameters(), restored.draft_model.parameters()
        ):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(
            backend2.optimizer.scheduler.last_epoch,
            backend.optimizer.scheduler.last_epoch,
        )

    def test_non_boundary_backward_disables_then_boundary_reenables_sync(self):
        model = _Composite()
        backend = self._backend(model)
        backend.backward(model(torch.ones(1, 3)).sum(), is_boundary=False)
        backend.backward(model(torch.ones(1, 3)).sum(), is_boundary=True)
        backend.step()

    def test_full_shard_auxiliary_head_outside_forward(self):
        model = _Composite()
        model.draft_model = _AuxDraft()
        backend = self._backend(model, "FULL_SHARD")
        hidden = model(torch.ones(2, 3))
        loss = model.draft_model.apply_logits_head(hidden).square().mean()
        backend.backward(loss)
        backend.step()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
