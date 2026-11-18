"""Compatibility alias for draftfit.research.papers.osprey.method."""

import sys

from draftfit.research.papers.osprey import method as _implementation

sys.modules[__name__] = _implementation
