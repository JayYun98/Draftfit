"""Compatibility path for :mod:`draftfit.research.papers.query_layout.method`."""

import sys

from .papers.query_layout import method

sys.modules[__name__] = method
