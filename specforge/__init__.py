"""Legacy imports resolve to the canonical :mod:`dspark` implementation."""

import importlib
import importlib.abc
import importlib.util
import sys


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, canonical):
        self.canonical = canonical

    def create_module(self, spec):
        module = importlib.import_module(self.canonical)
        self._canonical_spec = module.__spec__
        return module

    def exec_module(self, module):
        module.__spec__ = self._canonical_spec

    def get_code(self, fullname):
        # runpy / ``python -m specforge.some_module`` execute canonical code.
        return compile(
            f"import runpy; runpy.run_module({self.canonical!r}, run_name='__main__', alter_sys=True)",
            f"<legacy {fullname}>",
            "exec",
        )


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("specforge."):
            return None
        canonical = "dspark" + fullname[len("specforge") :]
        spec = importlib.util.find_spec(canonical)
        if spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(canonical),
            is_package=spec.submodule_search_locations is not None,
        )


sys.meta_path.insert(0, _AliasFinder())


def __getattr__(name):
    return getattr(importlib.import_module("dspark"), name)
