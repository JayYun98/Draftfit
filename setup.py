"""Bundle the existing recipe directories without duplicating their source files."""

from pathlib import Path
from shutil import copyfile

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithAssets(build_py):
    def run(self):
        super().run()
        root = Path(__file__).parent
        for directory in ("configs", "examples/configs"):
            for source in (root / directory).rglob("*"):
                if source.is_file() and source.suffix in (".json", ".yaml", ".yml"):
                    target = (
                        Path(self.build_lib)
                        / "speculative_train_platform/assets/_data"
                        / source.relative_to(root)
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    copyfile(source, target)
        for name in (
            "LICENSE",
            "THIRDPARTY_NOTICES.md",
            "licenses/torchspec-LICENSE",
            "licenses/angelspec-LICENSE",
            "licenses/z-lab-dflash-LICENSE",
            "docs/PUBLIC_WORKFLOW.md",
            "docs/PUBLIC_RELEASE.md",
            "docs/PUBLIC_SUPPORT.md",
            "docs/SOURCE_ATTRIBUTION.md",
            "docs/RUNTIME_PROFILES.md",
            "docs/NATIVE_LING_REPLAY.md",
            "docs/MODEL_VALIDATION.md",
            "docs/PERSONAL_INFERENCE.md",
            "docs/TARGET_EXTENSION.md",
            "docs/PERFORMANCE_GATE.md",
            "docs/RELEASE_GATES.md",
            "docs/CODE_OWNERSHIP.md",
            "docs/ENVIRONMENT.md",
            "docs/OWNED_GPU_ENVIRONMENT.md",
        ):
            target = (
                Path(self.build_lib) / "speculative_train_platform/assets/_data" / name
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            copyfile(root / name, target)


setup(cmdclass={"build_py": BuildWithAssets})
