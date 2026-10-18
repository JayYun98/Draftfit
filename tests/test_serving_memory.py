"""Real tiny Qwen joint training; CUDA gates require explicit operator opt-in."""
import copy
import os
import random
import unittest
from unittest import mock

import numpy as np
import torch

from draftfit.algorithms.builtin import builtin_algorithm_registry
from draftfit.training.joint_family import _backward_window
from draftfit.training import serving_memory
from tests import test_joint_verification as fixtures


class ServingMemoryTest(unittest.TestCase):
    execution_device = 'cpu'

    def setUp(self):
        # Composition avoids collecting/inheriting the unrelated verifier suite.
        self.fixture = fixtures.JointVerificationTest()
        self.fixture.execution_device = self.execution_device
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

    def handoff(self):
        f = self.fixture
        return serving_memory.release_training_models(f.bundles, f.optimizer, f.source._teacher)

    def assert_released(self, receipt, before):
        f = self.fixture
        self.assertEqual(receipt['mode'], 'released_training_memory')
        self.assertIs(receipt['target_resident'], False)
        roots = [b.model for b in f.bundles] + [f.source._teacher._model]
        for root in roots:
            for tensor in [*root.parameters(), *root.buffers()]:
                self.assertEqual(tensor.device.type, 'cpu')
            self.assertTrue(all(not m.training for m in root.modules()))
        for tensor, _, _, gradient, _ in before['parameters']:
            self.assertIsNone(tensor.grad)
            self.assertEqual(tensor.device.type, 'cpu')
            if gradient is not None:
                self.assertEqual(gradient.device.type, 'cpu')
        for state in f.optimizer.optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    self.assertEqual(value.device.type, 'cpu')
        if self.execution_device == 'cuda':
            self.assertIs(receipt['cuda_release_observed'], True)
            self.assertGreater(receipt['model_cuda_bytes'], 0)
            for device, previous in receipt['cuda_allocated_before'].items():
                self.assertLess(receipt['cuda_allocated_after'][device], previous)
                # With no retained views, at least all registered model storage
                # must actually leave the allocator (optimizer adds more).
                self.assertGreaterEqual(previous - receipt['cuda_allocated_after'][device],
                                        receipt['model_cuda_bytes'])
        else:
            self.assertIs(receipt['cuda_release_observed'], False)

    def consume_rng(self):
        random.random()
        np.random.random()
        torch.rand(3)
        if self.execution_device == 'cuda':
            torch.rand(3, device='cuda')

    def test_model_optimizer_identity_values_rng_and_modes_restored(self):
        f = self.fixture
        f.bundles[0].draft_model.layers[0].eval()
        before = f.snapshot()
        with self.handoff() as receipt:
            self.assert_released(receipt, before)
            self.consume_rng()
        f.assert_restored(before)

    def test_verifier_failure_restores_buffers_and_all_training_state(self):
        f = self.fixture
        before = f.snapshot()
        rotary = f.source._teacher._model.rotary_emb
        name, original = next(rotary.named_buffers(recurse=False))
        saved = original.detach().clone()
        with self.assertRaisesRegex(RuntimeError, 'verifier failed'):
            with self.handoff() as receipt:
                self.assert_released(receipt, before)
                self.consume_rng()
                rotary.register_buffer(name, torch.zeros_like(original), persistent=False)
                raise RuntimeError('verifier failed')
        self.assertIs(getattr(rotary, name), original)
        self.assertTrue(torch.equal(original, saved))
        f.assert_restored(before)

    def test_next_real_update_exact_after_success_and_failure(self):
        f = self.fixture
        registry = builtin_algorithm_registry()
        strategies = [registry.resolve(s.config.training.strategy).providers.step.build(b.model)
                      for s, b in zip(f.stages, f.bundles)]
        loader, _ = f.source.loader(f.stages[0], 0, True, True)
        window = [next(iter(loader))]
        before = f.snapshot()

        def update():
            _backward_window(f.stages, strategies, f.bundles, window)
            f.optimizer.step()
            return copy.deepcopy(f.optimizer.state_dict()), [
                copy.deepcopy(b.draft_model.state_dict()) for b in f.bundles]

        expected = update()
        for fail in (False, True):
            with self.subTest(verifier_fails=fail):
                for parameter, _, value, _, grad_value in before['parameters']:
                    parameter.data.copy_(value)
                    parameter.grad = None if grad_value is None else grad_value.clone()
                f.optimizer.load_state_dict(copy.deepcopy(before['optimizer']))
                random.setstate(before['python'])
                np.random.set_state(before['numpy'])
                torch.set_rng_state(before['torch'])
                if before['cuda_rng'] is not None:
                    torch.cuda.set_rng_state_all(before['cuda_rng'])
                try:
                    with self.handoff():
                        self.consume_rng()
                        if fail:
                            raise RuntimeError('controlled verifier failure')
                except RuntimeError as exc:
                    if not fail or str(exc) != 'controlled verifier failure':
                        raise
                f.assert_tree_equal(expected, update())

    def test_unsupported_teacher_hooks_and_storage_rejected_before_mutation(self):
        f = self.fixture
        before = f.snapshot()
        with self.assertRaisesRegex(ValueError, 'ordinary Torch HF'):
            with serving_memory.release_training_models(f.bundles, f.optimizer, object()):
                self.fail('unsupported teacher admitted')
        root = f.bundles[0].draft_model
        with mock.patch.object(root, '_hf_hook', object(), create=True):
            with self.assertRaisesRegex(ValueError, 'hooked'):
                with self.handoff():
                    self.fail('hooked model admitted')
        weight = next(root.parameters())
        root.register_buffer('_serving_test_alias', weight.detach())
        try:
            with self.assertRaisesRegex(ValueError, 'sharing storage'):
                with self.handoff():
                    self.fail('aliased storage admitted')
        finally:
            del root._serving_test_alias
        root.register_buffer('_serving_test_view', torch.ones(3)[1:])
        try:
            with self.assertRaisesRegex(ValueError, 'views'):
                with self.handoff():
                    self.fail('view admitted')
        finally:
            del root._serving_test_view
        f.assert_restored(before)

    def test_overlap_refused_and_guard_cleared_after_failure(self):
        before = self.fixture.snapshot()
        with self.handoff():
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                with self.handoff():
                    self.fail('nested handoff admitted')
        self.fixture.assert_restored(before)
        with self.handoff():
            pass
        self.fixture.assert_restored(before)


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('DRAFTFIT_RUN_QUEUE_CUDA') == '1',
                     'requires CUDA and DRAFTFIT_RUN_QUEUE_CUDA=1')
class ServingMemoryCUDA(ServingMemoryTest):
    execution_device = 'cuda'

    def test_partial_offload_failure_restores_already_moved_models(self):
        before = self.fixture.snapshot()
        original = serving_memory._move_data
        count = 0

        def fail_once(tensor, device):
            nonlocal count
            if str(device) == 'cpu':
                count += 1
                if count == 3:
                    raise RuntimeError('injected CPU copy failure')
            return original(tensor, device)

        with mock.patch.object(serving_memory, '_move_data', side_effect=fail_once):
            with self.assertRaisesRegex(RuntimeError, 'injected CPU copy failure'):
                with self.handoff():
                    self.fail('failed offload yielded')
        self.assertEqual(count, 3)
        self.fixture.assert_restored(before)
        with self.handoff() as receipt:
            self.assert_released(receipt, before)
        self.fixture.assert_restored(before)
