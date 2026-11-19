"""Apply Redline to a measured configuration table."""

import json


def run_redline(config):
    from .method import select

    with open(config.data.input_path) as handle:
        data = json.load(handle)
    result = select(
        data["correct"],
        data["cost"],
        names=data.get("names"),
        **config.calibration.model_dump(),
    )
    return {
        "deployed": result.deployed_name,
        "reference": result.names[result.reference],
        "valid": result.valid.tolist(),
        "pvalues": result.pvalues.tolist(),
        "risk": result.risk.tolist(),
        "n": result.n,
        "alpha": result.alpha,
        "delta": result.delta,
        "cost": result.cost.tolist(),
        "names": result.names,
    }
