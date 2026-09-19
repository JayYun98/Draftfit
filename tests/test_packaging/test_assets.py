"""Exercise the real wheel in an isolated interpreter without torch/SGLang."""

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from draftfit.assets import DOCUMENTS, export_assets, list_assets


class AssetTests(unittest.TestCase):
    def test_ling_docker_copy_inputs_build_versioned_complete_wheel(self):
        root = Path(__file__).resolve().parents[2]
        dockerfile = root / "patches/sglang/ling-8ba213f/Dockerfile"
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            # Exercise the actual COPY list, without pulling the CUDA base image.
            for line in dockerfile.read_text().splitlines():
                if not line.startswith("COPY "):
                    continue
                _, *inputs, destination = shlex.split(line)
                for name in inputs:
                    original = root / name
                    target = source / destination
                    if original.is_dir():
                        shutil.copytree(
                            original,
                            target,
                            dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(
                                "__pycache__", "*.pyc", "_data"
                            ),
                        )
                    else:
                        target.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(original, target / original.name)
            wheels = Path(temporary) / "wheels"
            result = subprocess.run(
                [
                    "uv",
                    "build",
                    "--python",
                    sys.executable,
                    "--offline",
                    "--no-build-isolation",
                    "--wheel",
                    "--out-dir",
                    str(wheels),
                    str(source),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            version = (root / "version.txt").read_text().strip()
            wheel = wheels / f"draftfit-{version}-py3-none-any.whl"
            with zipfile.ZipFile(wheel) as archive:
                for name in DOCUMENTS:
                    self.assertEqual(
                        archive.read("draftfit/assets/_data/" + name),
                        (root / name).read_bytes(),
                    )

    def test_source_export_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "recipes"
            export_assets(destination)
            self.assertTrue(
                (destination / "configs/ling-3.0-tiny-dspark.json").is_file()
            )
            with self.assertRaises(FileExistsError):
                export_assets(destination)

    def test_clean_wheel_assets_and_cli(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            source = temporary / "source"
            source.mkdir()
            for name in (
                "pyproject.toml",
                "setup.py",
                "MANIFEST.in",
                "README.md",
                "version.txt",
                "LICENSE",
            ):
                shutil.copyfile(root / name, source / name)
            for name in (
                "draftfit",
                "speculative_train_platform",
                "dspark",
                "specforge",
                "configs",
                "examples/configs",
            ):
                shutil.copytree(
                    root / name,
                    source / name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "_data"),
                )
            for name in DOCUMENTS:
                (source / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / name, source / name)
            wheel_dir = temporary / "wheels"
            result = subprocess.run(
                [
                    "uv",
                    "build",
                    "--python",
                    sys.executable,
                    "--no-build-isolation",
                    "--offline",
                    "--wheel",
                    "--out-dir",
                    str(wheel_dir),
                    str(source),
                ],
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "UV_PYTHON_DOWNLOADS": "never"},
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            wheel = next(wheel_dir.glob("*.whl"))
            with zipfile.ZipFile(wheel) as archive:
                direct_payload = {
                    name: archive.read(name)
                    for name in archive.namelist()
                    if name.startswith(
                        (
                            "draftfit/",
                            "speculative_train_platform/",
                            "dspark/",
                            "specforge/",
                        )
                    )
                    or name.endswith("/LICENSE")
                }
            # Build through the source distribution, not just the checkout:
            # missing MANIFEST entries otherwise remain invisible to wheel tests.
            sdist_dir = temporary / "sdist"
            result = subprocess.run(
                [
                    "uv",
                    "build",
                    "--python",
                    sys.executable,
                    "--no-build-isolation",
                    "--offline",
                    "--sdist",
                    "--out-dir",
                    str(sdist_dir),
                    str(source),
                ],
                cwd=source,
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "UV_PYTHON_DOWNLOADS": "never"},
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            rebuilt_dir = temporary / "rebuilt"
            result = subprocess.run(
                [
                    "uv",
                    "build",
                    "--python",
                    sys.executable,
                    "--no-build-isolation",
                    "--offline",
                    "--wheel",
                    "--out-dir",
                    str(rebuilt_dir),
                    str(next(sdist_dir.glob("*.tar.gz"))),
                ],
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "UV_PYTHON_DOWNLOADS": "never"},
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            wheel = next(rebuilt_dir.glob("*.whl"))
            with zipfile.ZipFile(wheel) as archive:
                rebuilt_payload = {
                    name: archive.read(name)
                    for name in archive.namelist()
                    if name.startswith(
                        (
                            "draftfit/",
                            "speculative_train_platform/",
                            "dspark/",
                            "specforge/",
                        )
                    )
                    or name.endswith("/LICENSE")
                }
                self.assertEqual(direct_payload, rebuilt_payload)
                self.assertTrue(
                    any(name.endswith("/LICENSE") for name in archive.namelist())
                )
                for name in archive.namelist():
                    self.assertFalse(
                        name.startswith(("build/", "runs/", ".git/")), name
                    )
                    if name.endswith((".py", ".json", ".yaml", ".yml")):
                        data = archive.read(name)
                        self.assertNotIn(b"/Users/", data, name)
                        self.assertNotIn(
                            b"-----BEGIN " + b"PRIVATE KEY-----", data, name
                        )
            # -I -S excludes the checkout, user/site packages, and PYTHONPATH.
            # zipimport also proves resources work without filesystem-only paths.
            code = "import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('draftfit.assets',run_name='__main__')"
            command = [sys.executable, "-I", "-S", "-c", code, str(wheel)]
            listed = subprocess.run(
                command + ["list"],
                cwd=temporary,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(listed.stdout.splitlines(), list_assets())
            destination = temporary / "installed-recipes"
            subprocess.run(
                command + ["export", str(destination)],
                cwd=temporary,
                capture_output=True,
                text=True,
                check=True,
            )
            for name in list_assets():
                self.assertEqual(
                    (destination / name).read_bytes(), (root / name).read_bytes()
                )
            refused = subprocess.run(
                command + ["export", str(destination)],
                cwd=temporary,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(refused.returncode, 0)


if __name__ == "__main__":
    unittest.main()
