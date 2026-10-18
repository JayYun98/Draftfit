"""Real CPU checkpoint replay: sidecars are evidence, never trusted execution input."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from draftfit.training import joint_family, joint_verification
from tests.test_joint_deadline import stages_for
from tests.test_joint_online_cpu import online_fixture


class NativeReportReplayTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        env = mock.patch.dict(os.environ, dict(SPECFORGE_DEVICE="cpu", HF_HUB_OFFLINE="1",
                                              TRANSFORMERS_OFFLINE="1"))
        env.start()
        self.addCleanup(env.stop)
        online_fixture(self.root, False)
        path = self.root / "base.json"
        base = json.loads(path.read_text())
        base["training"].update(max_checkpoints=10, joint_verification=dict(
            interval=1, max_prompts=2, max_new_tokens=4, draft_tokens=2, trials=2))
        path.write_text(json.dumps(base))
        self.partial = self.root / "partial"

        def fail(**_):
            raise RuntimeError("observer interrupted")

        with self.assertRaisesRegex(RuntimeError, "observer interrupted"):
            joint_family.run_joint_family(self.partial, stages_for(self.root, self.partial),
                                          milestone_observer=fail)
        self.checkpoint = self.partial / "joint-step1.pt"
        self.sidecar = self.partial / "verification-step1.json"
        self.checkpoint_sha = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.real_generation = joint_verification.evaluate_generation

    def resume(self, name, observer):
        output = self.root / name
        return joint_family.run_joint_family(output, stages_for(self.root, output),
            resume_from=self.checkpoint, milestone_observer=observer)

    def assert_tree_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_tree_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_tree_equal(x, y)
        else:
            self.assertEqual(a, b)

    def snapshot(self, bundles, optimizer):
        return deepcopy(dict(optimizer=optimizer.state_dict(), python=random.getstate(),
            numpy=np.random.get_state(), torch=torch.get_rng_state(),
            drafts=[b.draft_model.state_dict() for b in bundles],
            modes=[[m.training for m in b.model.modules()] for b in bundles]))

    def test_cached_runtime_settings_prompts_and_duplicate_requests_are_recomputed(self):
        original = json.loads(self.sidecar.read_text())
        cached = deepcopy(original)
        cached["bindings"]["runtime"]["dtype"] = "torch.float64"
        cached["bindings"]["config"]["max_new_tokens"] = 99
        cached["bindings"]["prompts"][0].update(prompt_token_ids=[55], prompt_sha256="f" * 64)
        for arm in [cached["baseline"], *cached["candidates"]]:
            arm["runs"][1] = deepcopy(arm["runs"][0])
        self.sidecar.write_text(json.dumps(cached))
        seen = []

        def observe(**kw):
            seen.append(deepcopy(kw["report"]))
            return dict(stop=True)

        with mock.patch.object(joint_verification, "evaluate_generation", wraps=self.real_generation) as generate:
            self.resume("recomputed", observe)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(seen[0]["bindings"], original["bindings"])
        for arm in [seen[0]["baseline"], *seen[0]["candidates"]]:
            self.assertEqual({(r["trial"], r["row_index"]) for r in arm["runs"]},
                             {(0, 0), (0, 1), (1, 0), (1, 1)})
        self.assertEqual(seen[0]["artifact_sha256"], self.checkpoint_sha)
        self.assertEqual(Path(seen[0]["checkpoint_path"]), self.checkpoint.resolve())
        self.assertEqual(json.loads(self.sidecar.read_text()), cached)
        saved = json.loads((self.root / "recomputed/verification-step1.json").read_text())
        self.assertEqual(saved, json.loads(json.dumps(seen[0])))

    def test_invalid_json_cache_is_not_read(self):
        self.sidecar.write_text('{"interrupted write":')
        observer = mock.Mock(return_value=dict(stop=True))
        self.resume("broken-cache", observer)
        observer.assert_called_once()

    def test_fresh_replay_rejects_missing_extra_duplicate_and_invalid_pairs(self):
        # Exercise malformed freshly generated output even on the old uncached path.
        self.sidecar.unlink()
        for side in ("baseline", "candidate"):
            for error in ("duplicate", "missing", "extra", "wrong_trial", "bool_index", "token_count"):
                with self.subTest(side=side, error=error):
                    def corrupt(*args, **kwargs):
                        report = self.real_generation(*args, **kwargs)
                        arm = report["baseline"] if side == "baseline" else report["candidates"][0]
                        if error == "duplicate": arm["runs"][1] = deepcopy(arm["runs"][0])
                        elif error == "missing": arm["runs"].pop()
                        elif error == "extra": arm["runs"].append(deepcopy(arm["runs"][0]))
                        elif error == "wrong_trial": arm["runs"][0]["trial"] = 3
                        elif error == "bool_index": arm["runs"][0]["row_index"] = False
                        else: arm["runs"][0]["generated_tokens"] += 1
                        return report
                    observer = mock.Mock(return_value=dict(stop=False))
                    with mock.patch.object(joint_verification, "evaluate_generation", side_effect=corrupt):
                        with self.assertRaisesRegex(ValueError, "replayed generation"):
                            self.resume(f"{side}-{error}", observer)
                    observer.assert_not_called()
                    self.assertFalse((self.root / f"{side}-{error}/joint-step2.pt").exists())

    def test_replay_preserves_optimizer_rng_and_exact_next_update(self):
        output = self.root / "full"
        joint_family.run_joint_family(output, stages_for(self.root, output))
        checked = []

        def inspect(stages, bundles, source, optimizer, step, config):
            before = self.snapshot(bundles, optimizer)
            report = self.real_generation(stages, bundles, source, optimizer, step, config)
            self.assert_tree_equal(before, self.snapshot(bundles, optimizer))
            checked.append(step)
            return report

        with mock.patch.object(joint_verification, "evaluate_generation", side_effect=inspect):
            self.resume("continued", lambda **_: dict(stop=False))
        self.assertEqual(checked, [1, 2])
        expected = torch.load(output / "joint-step2.pt", weights_only=True)
        actual = torch.load(self.root / "continued/joint-step2.pt", weights_only=True)
        for key in ("drafts", "optimizer", "rng", "step", "epoch", "next_batch", "horizon"):
            self.assert_tree_equal(expected[key], actual[key])
        self.assertEqual(hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(), self.checkpoint_sha)

    def test_replay_generation_failure_restores_state_and_allows_retry(self):
        checked = []

        def fail_generate(*_args, **_kwargs):
            random.random(); np.random.rand(); torch.rand(3)
            raise TimeoutError("injected replay deadline")

        def inspect(stages, bundles, source, optimizer, step, config):
            before = self.snapshot(bundles, optimizer)
            try:
                with mock.patch.object(joint_verification, "_generate", side_effect=fail_generate):
                    return self.real_generation(stages, bundles, source, optimizer, step, config)
            finally:
                self.assert_tree_equal(before, self.snapshot(bundles, optimizer))
                checked.append(step)

        observer = mock.Mock(return_value=dict(stop=True))
        with mock.patch.object(joint_verification, "evaluate_generation", side_effect=inspect):
            with self.assertRaisesRegex(TimeoutError, "injected replay deadline"):
                self.resume("failed-replay", observer)
        self.assertEqual(checked, [1])
        observer.assert_not_called()
        self.assertFalse((self.root / "failed-replay/joint-step2.pt").exists())
        self.resume("retry", observer)
        observer.assert_called_once()
        self.assertEqual(hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(), self.checkpoint_sha)


if __name__ == "__main__":
    unittest.main()
