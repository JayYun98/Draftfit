"""CPU guard for the real GPU gate's tensor comparison."""

import importlib.util
import unittest
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "ling_features",
    Path(__file__).parents[2] / "tests/integration/check_ling_online_features.py",
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class FeatureComparisonTest(unittest.TestCase):
    def test_real_values_and_nonfinite_fail_closed(self):
        a = {
            "hidden_states": torch.ones(1, 3, 4, dtype=torch.bfloat16),
            "input_ids": torch.tensor([[1, 2, 3]]),
        }
        b = {k: v.clone() for k, v in a.items()}
        self.assertTrue(all(v["passed"] for v in gate.compare_features(a, b).values()))
        b["hidden_states"][0, 0, 0] = 2
        report = gate.compare_features(a, b)
        self.assertFalse(report["hidden_states"]["passed"])
        self.assertEqual(report["hidden_states"]["max_abs_error"], 1)
        self.assertNotEqual(
            report["hidden_states"]["online_sha256"],
            report["hidden_states"]["offline_sha256"],
        )
        b["hidden_states"][0, 0, 0] = float("nan")
        self.assertFalse(gate.compare_features(a, b)["hidden_states"]["passed"])
        b["input_ids"][0, 0] = 2
        self.assertFalse(gate.compare_features(a, b)["input_ids"]["passed"])
        with self.assertRaises(ValueError):
            gate.compare_features(a, {"input_ids": b["input_ids"]})


if __name__ == "__main__":
    unittest.main()
