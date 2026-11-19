"""Compatibility path for the pinned TreeWY reference."""

import sys

from ..papers.treewy import upstream

sys.modules[__name__] = upstream
