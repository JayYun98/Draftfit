"""Compatibility path for :mod:`draftfit.research.papers.cast.width`."""

import sys

from draftfit.research.papers.cast import width

sys.modules[__name__] = width
