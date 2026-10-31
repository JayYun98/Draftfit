"""Replay scheduling, offload-policy constraints, and real diagnostic recipes."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from tests.research.test_latest_kv_tree import save_tokenizer
from tests.test_research_methods import target_model

ROOT = Path(__file__).resolve().parents[2]


class LatestControlTest(unittest.TestCase):
    def test_replay_matches_greedy_and_leaves_draft_after_mismatch(self):
        from draftfit.research.arr import ReplayProposer
        from draftfit.research.reference_decode import generate

        target, ids = target_model(), torch.tensor([[1, 2, 3]])
        expected = target.generate(
            ids, max_new_tokens=10, do_sample=False, eos_token_id=None, pad_token_id=0
        )
        previous = expected[0, 3:].tolist()
        for wrong in (False, True):
            candidate = list(previous)
            if wrong:
                candidate[1] = (candidate[1] + 1) % 32
            proposer = ReplayProposer(3, candidate, 4)
            output, stats = generate(target, ids, proposer, max_new_tokens=10)
            self.assertTrue(torch.equal(output, expected))
            if wrong:
                self.assertFalse(proposer.on_draft)
            else:
                self.assertLess(stats["target_forwards"], 10)

    def test_arr_replacement_last_complete_and_one_vote_per_root(self):
        from draftfit.research.arr import schedule_roots

        initial = iter([[1, 9], [2, 9], [3, 9]])
        revisited = set()

        def new():
            return {"tokens": next(initial), "complete": True}

        def revise(previous):
            if previous == [1, 9]:
                return {"tokens": previous, "complete": True}
            if previous == [2, 9]:
                return {"tokens": [8], "complete": False}
            if previous == [3, 9]:
                revisited.add(3)
                return {"tokens": [4, 9], "complete": True}
            return None

        roots, events = schedule_roots(new, revise, active_roots=2, max_roots=3)
        self.assertEqual(len({r["root_id"] for r in roots}), 3)
        self.assertEqual([r["tokens"] for r in roots], [[1, 9], [2, 9], [4, 9]])
        self.assertEqual(revisited, {3})
        self.assertEqual(sum(e["event"] == "initial" for e in events), 3)

    def test_arr_real_model_budget_and_stationary_config(self):
        from draftfit.research.config import ARRConfig, ReplayOptions
        from draftfit.research.runner import run

        with self.assertRaisesRegex(ValueError, "stationary"):
            ReplayOptions(revision_template="{question} {solution} {round}")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = target_model()
            with torch.no_grad():
                target.lm_head.weight.zero_()  # Deterministic EOS fixed point for every prefix.
            target.save_pretrained(root / "model")
            save_tokenizer(root / "model")
            (root / "input.jsonl").write_text(json.dumps({"input_ids": [1, 2]}) + "\n")
            cfg = ARRConfig(
                method="arr",
                model={"target_model_path": str(root / "model")},
                data={"input_path": str(root / "input.jsonl")},
                generation={"max_new_tokens": 3, "stop_token_ids": [0]},
                refinement={
                    "active_roots": 2,
                    "max_roots": 3,
                    "root_temperature": 0,
                    "max_processed_positions": 1000,
                    "revision_template": "{question} {solution}",
                    "answer_extractor": "token_sequence",
                },
                output_dir=str(root / "result"),
            )
            result = run(cfg)["answers"][0]
            self.assertEqual(result["votes"], {"0": 3})
            self.assertTrue(
                all(r["retirement"] == "fixed_point" for r in result["roots"])
            )
            self.assertLessEqual(result["processed_target_positions"], 1000)
            self.assertEqual(
                result["processed_target_positions"],
                sum(e["stats"]["processed_target_positions"] for e in result["events"]),
            )
            cfg.output_dir = str(root / "too-small")
            cfg.refinement.max_processed_positions = 1
            small = run(cfg)["answers"][0]
            self.assertTrue(small["no_vote"])
            self.assertEqual(small["processed_target_positions"], 0)

    def test_moe_paper_transitions_and_exhaustive_constraint_fallback(self):
        from draftfit.research.config import MoEPolicyOptions
        from draftfit.research.moe_spac import UtilityEstimator, choose_threshold

        estimator = UtilityEstimator(2, gamma=4, max_utility=2, smoothing=0.5)
        self.assertEqual(
            estimator.update([4, 0]), {"utility": [1, 0], "up": [3, 2], "down": [2, 2]}
        )
        self.assertEqual(estimator.update([4, 4])["utility"], [1, 1])
        self.assertEqual(estimator.update([0, 4])["utility"], [0, 1])
        self.assertEqual(UtilityEstimator(1, gamma=1).update([0])["utility"], [1])
        opts = MoEPolicyOptions(
            gamma=4,
            max_utility=2,
            target_layers=1,
            expert_bytes=64,
            cpu_unit_ms=1,
            gpu_unit_ms=1,
            io_unit_ms=0,
            draft_unit_ms=0,
        )
        ratios = [
            {"cpu_ratio": 0.5, "gpu_ratio": 0.5},
            {"cpu_ratio": 0.75, "gpu_ratio": 0.25},
        ]
        result = choose_threshold([1, 2], [2, 2], [0], ratios, 64, opts)
        self.assertEqual(result["selected"]["threshold"], 1)
        self.assertEqual(result["selected"]["prefetch_expert_ids"], [1])
        self.assertTrue(
            choose_threshold([1, 2], [2, 2], [0], ratios, 0, opts)["cpu_fallback"]
        )

    def test_query_specific_masks_noncontiguous_gqa_and_phase_admission(self):
        from draftfit.research.query_layout import compare_layouts

        rng = torch.Generator().manual_seed(9)
        q = torch.randn(4, 3, 5, generator=rng).transpose(0, 1)
        k, v = torch.randn(7, 2, 5, generator=rng), torch.randn(7, 2, 6, generator=rng)
        mask = torch.zeros(3, 14)[:, ::2]
        mask[1, 4:] = -float("inf")
        result = compare_layouts(q, k, v, mask)
        self.assertTrue(result["admitted"])
        expected = []
        for query in range(3):
            per_head = []
            for head in range(4):
                weights = (
                    (k[:, head // 2] @ q[query, head]) / 5**0.5 + mask[query]
                ).softmax(0)
                per_head.append(weights @ v[:, head // 2])
            expected.append(torch.stack(per_head))
        torch.testing.assert_close(
            torch.tensor(result["qb_output"]), torch.stack(expected)
        )
        self.assertFalse(compare_layouts(q, k, v, mask, role="draft")["admitted"])
        with self.assertRaisesRegex(ValueError, "unmasked"):
            compare_layouts(q, k, v, torch.full((3, 7), -float("inf")))

    def test_real_numeric_cli_recipes(self):
        rows = {
            "treewy": {
                "q": [[[1.0, 0.0]], [[0.0, 1.0]]],
                "k": [[[1.0, 0.0]], [[0.0, 1.0]]],
                "v": [[[2.0]], [[3.0]]],
                "g": [[-0.1], [-0.2]],
                "beta": [[0.5], [0.5]],
                "parent": [-1, 0],
                "initial_state": [[[0.0, 0.0]]],
                "accepted_node": 1,
            },
            "query_layout": {
                "q": [[[1.0, 0.0]]],
                "k": [[[1.0, 0.0]], [[0.0, 1.0]]],
                "v": [[[2.0]], [[3.0]]],
                "mask": [[0.0, 0.0]],
            },
            "moe_spac": {
                "layer": 0,
                "frequencies": [4, 0],
                "resident_expert_ids": [],
                "ratios": [{"cpu_ratio": 0.5, "gpu_ratio": 0.5}] * 2,
                "free_bytes": 128,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for method, row in rows.items():
                data = root / f"{method}.jsonl"
                data.write_text(json.dumps(row) + "\n")
                raw = {
                    "method": method,
                    "data": {"input_path": str(data)},
                    "output_dir": str(root / method),
                }
                if method == "moe_spac":
                    raw["policy"] = dict(
                        gamma=4,
                        max_utility=2,
                        target_layers=1,
                        expert_bytes=64,
                        cpu_unit_ms=1,
                        gpu_unit_ms=1,
                        io_unit_ms=0,
                        draft_unit_ms=0,
                    )
                recipe = root / f"{method}.json"
                recipe.write_text(json.dumps(raw))
                result = subprocess.run(
                    [sys.executable, "-m", "draftfit.research", "-c", str(recipe)],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(json.loads(result.stdout)["answers"]), 1)
                self.assertTrue((root / method / "result.json").is_file())


if __name__ == "__main__":
    unittest.main()
