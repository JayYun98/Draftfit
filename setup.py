"""Bundle the existing recipe directories without duplicating their source files."""

from pathlib import Path
from runpy import run_path
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
                        / "draftfit/assets/_data"
                        / source.relative_to(root)
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    copyfile(source, target)
        for name in run_path(str(root / "draftfit/assets/__init__.py"))["DOCUMENTS"]:
            target = Path(self.build_lib) / "draftfit/assets/_data" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            copyfile(root / name, target)


setup(cmdclass={"build_py": BuildWithAssets})
