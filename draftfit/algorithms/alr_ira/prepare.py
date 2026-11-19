"""Compatibility alias for draftfit.research.papers.alr_ira.method."""

import sys

from draftfit.research.papers.alr_ira import method as _implementation

sys.modules[__name__] = _implementation
