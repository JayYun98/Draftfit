"""CPU check: hardware overrides compose with the model and reach server argv."""

import unittest
from pathlib import Path

import yaml

from speculative_train_platform.config import load_config
from speculative_train_platform.launch_plan import _sglang_argv


class RuntimeProfileTest(unittest.TestCase):
    def test_ling_composition(self):
        root = Path(__file__).resolve().parents[2]
        for sm, attention in (("90", "triton"), ("120", "triton"), ("80", "triton")):
            with self.subTest(sm=sm):
                path = root / f"configs/runtime_profiles/linux-sm{sm}-cu129.yaml"
                overrides = yaml.safe_load(path.read_text())
                cfg = load_config(
                    str(root / "examples/configs/ling-3.0-tiny-dspark-online.yaml"),
                    overrides,
                )
                argv = _sglang_argv(cfg.model)
                self.assertEqual(argv[argv.index("--attention-backend") + 1], attention)
                self.assertEqual(
                    argv[argv.index("--linear-attn-backend") + 1], "triton"
                )
                self.assertEqual(
                    argv[argv.index("--linear-attn-verify-backend") + 1], "triton"
                )
                self.assertIn("--enable-deterministic-inference", argv)
                self.assertIn("--disable-cuda-graph", argv)
                self.assertEqual(cfg.model.sglang_context_length, 4103)
                managed = cfg.deployment.disaggregated.managed_local
                self.assertEqual(managed.trainer_cuda_visible_devices, ["1"])
                self.assertEqual(managed.capture_servers[0].cuda_visible_devices, ["0"])


if __name__ == "__main__":
    unittest.main()
