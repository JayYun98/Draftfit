"""Compatibility path for :mod:`draftfit.research.papers.cast.verify`."""

import sys

from draftfit.research.papers.cast import verify

sys.modules[__name__] = verify
