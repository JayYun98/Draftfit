"""Compatibility alias for draftfit.research.papers.redline.method."""

import sys

from draftfit.research.papers.redline import method as _implementation

sys.modules[__name__] = _implementation
