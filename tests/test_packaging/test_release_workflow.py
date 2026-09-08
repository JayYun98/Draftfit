"""Exercise the publication identity guard without publishing or network calls."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


class ReleaseWorkflowTest(unittest.TestCase):
    def test_publication_requires_cpu_gate_and_explicit_fork_identity(self):
        root = Path(__file__).resolve().parents[2]
        workflow = yaml.safe_load((root / ".github/workflows/publish_pypi.yaml").read_text())
        job = workflow["jobs"]["build-n-publish"]
        self.assertEqual(job["needs"], "cpu-gate")
        self.assertIn("refs/heads/main", job["if"])
        self.assertIn("SPECFORGE_ENABLE_PUBLISH", job["if"])
        guard = next(step for step in job["steps"] if step.get("name") == "Refuse accidental upstream identity publication")
        code = guard["run"].split("python - <<'PY'\n", 1)[1].rsplit("PY", 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            for name, approved, confirmed, success in (
                ("SpecForge", "SpecForge", "SpecForge", False),
                ("specforgeee", "specforgeee", "specforgeee", False),
                ("example-draft", "", "example-draft", False),
                ("example-draft", "example-draft", "wrong", False),
                ("example-draft", "example-draft", "example-draft", True),
            ):
                with self.subTest(name=name, approved=approved, confirmed=confirmed):
                    (Path(temporary) / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n')
                    result = subprocess.run(
                        [sys.executable, "-O", "-c", code], cwd=temporary, capture_output=True,
                        env={**os.environ, "CONFIRMED_NAME": confirmed, "APPROVED_NAME": approved,
                             "GITHUB_REPOSITORY": "example/fork"}, timeout=10,
                    )
                    self.assertEqual(result.returncode == 0, success, result.stderr)


if __name__ == "__main__":
    unittest.main()
