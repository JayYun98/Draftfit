"""Compatibility path for :mod:`draftfit.research.papers.moe_spac.method`."""

import sys

from .papers.moe_spac import method

sys.modules[__name__] = method
