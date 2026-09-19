"""Share canonical module objects with all legacy package namespaces."""

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
    def __init__(self, namespace):
        self.namespace = namespace

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.namespace + "."):
            return None
        canonical = "draftfit" + fullname[len(self.namespace) :]
        spec = importlib.util.find_spec(canonical)
        if spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(canonical),
            is_package=spec.submodule_search_locations is not None,
        )


def install(namespace):
    if not any(
        isinstance(finder, _AliasFinder) and finder.namespace == namespace
        for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _AliasFinder(namespace))


def resolve_attribute(name):
    return getattr(importlib.import_module("draftfit"), name)
