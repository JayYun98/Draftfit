"""Compatibility imports for the DForkTree scoring reference."""

from .papers.dforktree import method as _method

CandidateScores = _method.CandidateScores
score_candidates = _method.score_candidates
run_dforktree_scoring = _method.run_dforktree_scoring

__all__ = ["CandidateScores", "score_candidates"]
