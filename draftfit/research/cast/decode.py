"""Compatibility path for :mod:`draftfit.research.papers.cast.decode`."""

import sys

from draftfit.research.papers.cast import decode

sys.modules[__name__] = decode
