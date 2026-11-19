"""Compatibility path for :mod:`draftfit.research.papers.valuspec.method`."""

import sys

from draftfit.research.papers.valuspec import method

sys.modules[__name__] = method
