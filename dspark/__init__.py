"""Compatibility namespace for existing DSpark imports and pickles."""

from draftfit._legacy import install
from draftfit._legacy import resolve_attribute as __getattr__  # noqa: F401

install(__name__)
