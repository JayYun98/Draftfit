"""Compatibility alias for draftfit.research.papers.osprey.tokenizer_alignment."""

import sys

from draftfit.research.papers.osprey import tokenizer_alignment as _implementation

sys.modules[__name__] = _implementation
