"""Legacy module globals share canonical classes, registries and CLI behavior."""

import subprocess
import sys
import unittest


class NamespaceCompatibilityTest(unittest.TestCase):
    def test_import_orders_and_legacy_pickle_globals(self):
        for first, second in (("specforge", "dspark"), ("dspark", "specforge")):
            with self.subTest(first=first):
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        f"""
import importlib, pickle
for suffix in ('config.schema', 'runtime.contracts', 'algorithms.builtin'):
    a = importlib.import_module('{first}.' + suffix)
    b = importlib.import_module('{second}.' + suffix)
    assert a is b
    assert a.__spec__.name == 'dspark.' + suffix
from dspark.runtime.contracts import SampleRef
assert pickle.loads(b'cspecforge.runtime.contracts\\nSampleRef\\n.') is SampleRef
assert pickle.loads(pickle.dumps(SampleRef)) is SampleRef
from dspark.runtime.contracts import FeatureSpec
feature = FeatureSpec('hidden_states', (1, 2, 4), 'float32')
current = pickle.dumps(feature, protocol=0)
legacy = current.replace(b'dspark.runtime.contracts', b'specforge.runtime.contracts')
assert pickle.loads(legacy) == feature
assert type(pickle.loads(legacy)) is FeatureSpec
from dspark.config import Config
from specforge.config import Config as OldConfig
assert Config is OldConfig
""",
                    ],
                    check=True,
                )

    def test_legacy_module_entrypoints(self):
        for suffix in ("cli", "assets", "inference.teacher_server"):
            old = subprocess.check_output(
                [sys.executable, "-m", "specforge." + suffix, "--help"], text=True
            )
            new = subprocess.check_output(
                [sys.executable, "-m", "dspark." + suffix, "--help"], text=True
            )
            self.assertEqual(old, new)


if __name__ == "__main__":
    unittest.main()
