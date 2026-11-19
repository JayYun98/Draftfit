"""Compatibility path for :mod:`draftfit.research.papers.cast._masses`."""

import sys

from draftfit.research.papers.cast import _masses

sys.modules[__name__] = _masses
