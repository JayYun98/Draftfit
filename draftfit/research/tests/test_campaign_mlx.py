import copy
import tempfile
import unittest
from pathlib import Path


class AdapterTests(unittest.TestCase):
    def test_raw_mlx_aggregation_and_rejection(self):
        from draftfit.campaign_mlx import report_from_rows

        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "train.json"
            receipt.write_text("{}")
            rows = [
                dict(
                    kind="provenance",
                    manifest={
                        "target": "fixed",
                        "arms": [
                            {"name": "target", "mode": "baseline"},
                            {"name": "dspark", "mode": "dspark", "path": "/weights"},
                        ],
                    },
                    trials=3,
                    selected_ids=["code:1"],
                    max_new_tokens=4,
                    data_sha256="data",
                    runtime_commit="runtime",
                )
            ]
            for trial in range(3):
                for name, speed in [("target", 10), ("dspark", 20)]:
                    rows.append(
                        dict(
                            kind="measurement",
                            arm=name,
                            trial=trial,
                            id="code:1",
                            domain="code",
                            parity=True,
                            result={"token_ids": [1, 2]},
                            tok_s=speed,
                        )
                    )
            rows.append(dict(kind="summary", failed_arms=[]))
            report = report_from_rows(rows, receipt, "fixed", ["dspark"])
            d = report["candidates"][0]["domains"][0]
            self.assertAlmostEqual(d["baseline_seconds"] / d["seconds"], 2)
            for mode in ("missing", "duplicate", "parity", "tokens", "trials"):
                broken = copy.deepcopy(rows)
                if mode == "missing":
                    del broken[1]
                if mode == "duplicate":
                    broken.insert(1, broken[1])
                if mode == "parity":
                    broken[2]["parity"] = False
                if mode == "tokens":
                    broken[2]["result"]["token_ids"] = [3, 4]
                if mode == "trials":
                    broken[0]["trials"] = 1
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    report_from_rows(broken, receipt, "fixed", ["dspark"])
