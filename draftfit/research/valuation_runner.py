"""Compatibility path for :mod:`draftfit.research.papers.valuspec.recipe`."""

import sys

from draftfit.research.papers.valuspec import recipe

sys.modules[__name__] = recipe
