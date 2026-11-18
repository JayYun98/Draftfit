"""Compatibility path for :mod:`draftfit.research.papers.cast.tree`."""

import sys

from draftfit.research.papers.cast import tree

sys.modules[__name__] = tree
