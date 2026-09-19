import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from draftfit.artifacts import build_artifact_manifest


class ArtifactManifestTest(unittest.TestCase):
    def test_manifest_records_sources_and_validation_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            validation = root / "validation.json"
            config.write_text('{"model": "ling"}\n', encoding="utf-8")
            validation.write_text('{"passed": true, "checks": []}\n', encoding="utf-8")
            manifest = build_artifact_manifest(
                run_id="run-1",
                target_model="inclusionAI/Ling-3.0-tiny",
                target_revision="rev-a",
                tokenizer_version="tok-a",
                backend_revision="sglang-a",
                sources={"config": str(config)},
                validation_paths=[str(validation)],
            )
            self.assertEqual(manifest["schema_version"], "dspark_artifact_v1")
            self.assertEqual(manifest["metadata"]["target_revision"], "rev-a")
            self.assertEqual(len(manifest["sources"]["config"]["sha256"]), 64)
            self.assertTrue(manifest["validation"][0]["passed"])
            self.assertTrue(manifest["artifact_hash"])

    def test_cli_writes_manifest_and_rejects_missing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.jsonl"
            validation = root / "validation.json"
            output = root / "run" / "artifact.json"
            source.write_text('{"id": 1}\n', encoding="utf-8")
            validation.write_text('{"passed": false}\n', encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "draftfit.cli",
                    "artifact",
                    "manifest",
                    "--output",
                    str(output),
                    "--run-id",
                    "run-1",
                    "--target-model",
                    "target",
                    "--target-revision",
                    "rev-a",
                    "--source",
                    f"data={source}",
                    "--validation",
                    str(validation),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["validation"][0]["passed"], False)
            self.assertEqual(json.loads(result.stdout)["output"], str(output))

            missing = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "draftfit.cli",
                    "artifact",
                    "manifest",
                    "--output",
                    str(output),
                    "--run-id",
                    "run-1",
                    "--target-model",
                    "target",
                    "--target-revision",
                    "rev-a",
                    "--source",
                    f"data={root / 'missing'}",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(missing.returncode, 2)


if __name__ == "__main__":
    unittest.main()
