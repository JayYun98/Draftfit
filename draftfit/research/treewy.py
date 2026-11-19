"""Compatibility path for :mod:`draftfit.research.papers.treewy.method`."""

import sys

from .papers.treewy import method

sys.modules[__name__] = method
