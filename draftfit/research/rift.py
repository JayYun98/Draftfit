"""Compatibility path for :mod:`draftfit.research.papers.rift.method`."""

import sys

from .papers.rift import method

sys.modules[__name__] = method
