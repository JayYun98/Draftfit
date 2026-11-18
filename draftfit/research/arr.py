"""Compatibility path for :mod:`draftfit.research.papers.arr.method`."""

import sys

from .papers.arr import method

sys.modules[__name__] = method
