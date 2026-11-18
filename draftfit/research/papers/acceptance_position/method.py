"""Observed acceptance for attempted proposal tokens (arXiv:2610.30644v1)."""

from collections import defaultdict


def attempted_token(position, token, accepted, p, q, vocab):
    """Summarize one verification attempt using its target and proposal mass."""
    if any(i >= vocab for i in q):
        raise ValueError("proposal distribution is outside the target vocabulary")
    overlap = sum(min(float(p[i]), qi) for i, qi in q.items())
    q_argmax = min(q, key=lambda i: (-q[i], i))
    return {
        "output_position": position,
        "proposed_token": token,
        "accepted": accepted,
        "distribution_overlap": overlap,
        "argmax_agreement": q_argmax == int(p.argmax()),
    }


def position_bins(records, width):
    bins = defaultdict(list)
    for record in records:
        bins[record["output_position"] // width].append(record)
    return [
        {
            "start_position": i * width,
            "attempted_proposals": len(rows),
            "observed_acceptance": sum(r["accepted"] for r in rows) / len(rows),
            "mean_distribution_overlap": sum(r["distribution_overlap"] for r in rows)
            / len(rows),
            "argmax_agreement": sum(r["argmax_agreement"] for r in rows) / len(rows),
        }
        for i, rows in sorted(bins.items())
    ]
