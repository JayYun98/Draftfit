"""Real tiny-family evaluation uses stable anchors without advancing training RNG."""

import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from draftfit.algorithms.builtin import builtin_algorithm_registry
from draftfit.training.assembly import build_model_bundle
from draftfit.training.joint_family import (
    _evaluate,
    _restore_rng,
    _rng_state,
    _tie_backbones,
)
from tests import test_joint_family_core


class JointEvaluationRNGTest(unittest.TestCase):
    def setUp(self):
        before = _rng_state()
        self.addCleanup(_restore_rng, before)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        env = mock.patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"})
        env.start()
        self.addCleanup(env.stop)
        temporary = tempfile.TemporaryDirectory(prefix="joint-eval-rng-")
        self.addCleanup(temporary.cleanup)
        _, self.stages = test_joint_family_core.JointFamilyCoreTest().stages(
            Path(temporary.name)
        )
        registry = builtin_algorithm_registry()
        torch.manual_seed(7)
        self.bundles = [
            build_model_bundle(
                stage.config,
                algorithm=registry.resolve(stage.config.training.strategy),
            )
            for stage in self.stages
        ]
        _tie_backbones(self.bundles, scratch=True)
        self.strategies = [
            registry.resolve(stage.config.training.strategy).providers.step.build(
                bundle.model
            )
            for stage, bundle in zip(self.stages, self.bundles)
        ]

    def assert_rng_unchanged(self, before):
        after = _rng_state()
        self.assertTrue(torch.equal(before["torch"], after["torch"]))
        self.assertEqual(before["python"], after["python"])
        self.assertEqual(len(before["cuda"]), len(after["cuda"]))
        for expected, actual in zip(before["cuda"], after["cuda"]):
            self.assertTrue(torch.equal(expected, actual))

    def evaluate(self, caller_seed):
        torch.manual_seed(caller_seed)
        random.seed(caller_seed)
        before = _rng_state()
        metrics = _evaluate(self.stages, self.strategies, self.bundles)
        self.assert_rng_unchanged(before)
        self.assertTrue(all(bundle.model.training for bundle in self.bundles))
        return metrics

    def test_fixed_weights_and_data_ignore_callers_training_rng(self):
        first = self.evaluate(101)
        second = self.evaluate(202)
        for stage in self.stages:
            self.assertIn("eval/avg_loss", first[stage.name])
        self.assertIn("eval/teacher_top1_prob", first["dspark"])
        # Compare every diagnostic as well as loss: teacher probabilities depend
        # on the sampled target positions even when all weights are unchanged.
        self.assertEqual(first, second)

    def test_evaluation_uses_configured_seed(self):
        for stage in self.stages:
            stage.config.training.seed = 17
        first = self.evaluate(101)
        for stage in self.stages:
            stage.config.training.seed = 18
        second = self.evaluate(101)
        self.assertNotEqual(first, second)

    def test_forward_exception_restores_rng_and_propagates(self):
        failure = RuntimeError("evaluation forward failed")

        def fail(batch):
            del batch
            torch.rand(3)
            random.random()
            raise failure

        before = _rng_state()
        with mock.patch.object(self.strategies[0], "forward_loss", side_effect=fail):
            with self.assertRaises(RuntimeError) as raised:
                _evaluate(self.stages, self.strategies, self.bundles)
        self.assertIs(raised.exception, failure)
        self.assert_rng_unchanged(before)
        self.assertTrue(all(bundle.model.training for bundle in self.bundles))

    def test_loader_exception_restores_rng_and_propagates(self):
        failure = RuntimeError("evaluation loader failed")

        def fail(*args, **kwargs):
            torch.rand(3)
            random.random()
            raise failure

        before = _rng_state()
        source = SimpleNamespace(loader=fail)
        with self.assertRaises(RuntimeError) as raised:
            _evaluate(self.stages, self.strategies, self.bundles, online_source=source)
        self.assertIs(raised.exception, failure)
        self.assert_rng_unchanged(before)
        self.assertTrue(all(bundle.model.training for bundle in self.bundles))


if __name__ == "__main__":
    unittest.main()
