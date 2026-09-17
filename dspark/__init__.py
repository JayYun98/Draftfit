"""Compatibility namespace for existing DSpark imports and pickles."""

from speculative_train_platform._legacy import install
from speculative_train_platform._legacy import (  # noqa: F401
    resolve_attribute as __getattr__,
)

install(__name__)
