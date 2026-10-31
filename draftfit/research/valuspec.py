"""Conditional ValuSpec branch scoring before CAST target verification.

Independent reference implementation of arXiv:2610.31338v1. Our explicit
branch policy scores complete root-to-leaf paths and retains their ancestor
union. CAST's independent marginal scores are never used as valuations.
"""

import math
import time

import torch

from .cast.tree import TreeNode


def lower_quantile(scores, delta):
    """Linear empirical quantile (a declared local interpolation convention)."""
    if not scores or not 0 < delta < 1 or not all(map(math.isfinite, scores)):
        raise ValueError("calibration requires finite scores and 0 < delta < 1")
    ordered = sorted(scores)
    index = (len(ordered) - 1) * delta
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (index - lo) * (ordered[hi] - ordered[lo])


@torch.inference_mode()
def conditional_score(scorer, prefix, branch):
    if not prefix or not branch:
        raise ValueError("conditional branch scoring requires prefix and branch tokens")
    tokens = torch.tensor([list(prefix) + list(branch)], device=scorer.device)
    if tokens.min() < 0 or tokens.max() >= scorer.config.vocab_size:
        raise ValueError("scorer token IDs are outside its vocabulary")
    if tokens.shape[1] > scorer.config.max_position_embeddings:
        raise ValueError("scorer context capacity exceeded")
    # Position len(prefix)-1 predicts the first branch token.
    logits = scorer(tokens, use_cache=False).logits[0, len(prefix) - 1 : -1]
    lp = logits.float().log_softmax(-1)
    score = lp.gather(-1, tokens[0, len(prefix) :, None]).mean().item()
    if not math.isfinite(score):
        raise ValueError("scorer produced a nonfinite branch score")
    return score


def leaf_paths(nodes):
    paths, parents = [], set()
    for index, node in enumerate(nodes):
        if not -1 <= node.parent < index:
            raise ValueError("tree parents must precede children")
        if node.depth != (1 if node.parent == -1 else nodes[node.parent].depth + 1):
            raise ValueError("tree depths must follow parent links")
        parents.add(node.parent)
    for index in range(len(nodes)):
        if index not in parents:
            path, cursor = [], index
            while cursor != -1:
                path.append(cursor)
                cursor = nodes[cursor].parent
            paths.append(list(reversed(path)))
    return paths


def retain_branches(nodes, paths, scores, threshold):
    """Keep complete passing paths, including ancestors with lower means."""
    if len(paths) != len(scores) or not math.isfinite(threshold):
        raise ValueError("invalid branch scores or threshold")
    keep = set()
    for path, score in zip(paths, scores):
        if not math.isfinite(score):
            raise ValueError("branch scores must be finite")
        if score >= threshold:
            keep.update(path)
    remap = {-1: -1}
    retained = []
    for old, node in enumerate(nodes):
        if old in keep:
            if node.parent not in remap:
                raise ValueError("retained branch is missing an ancestor")
            remap[old] = len(retained)
            retained.append(
                TreeNode(node.token, node.depth, remap[node.parent], node.score)
            )
    return retained


class ConditionalBranchFilter:
    def __init__(self, scorer, threshold):
        if not math.isfinite(threshold):
            raise ValueError("valuation threshold must be finite")
        self.scorer, self.threshold = scorer, threshold

    def __call__(self, prefix, nodes):
        device = self.scorer.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        begin = time.perf_counter()
        paths = leaf_paths(nodes)
        scores = [
            conditional_score(self.scorer, prefix, [nodes[i].token for i in path])
            for path in paths
        ]
        result = retain_branches(nodes, paths, scores, self.threshold)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        return result, {
            "candidate_tokens": len(nodes),
            "retained_tokens": len(result),
            "scored_branches": len(paths),
            "scorer_forwards": len(paths),
            "score_and_compact_seconds": time.perf_counter() - begin,
        }
