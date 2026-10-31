"""Real-model verification and independent tree/state oracles for new recipes."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from tests.test_research_methods import draft_model, target_model


def save_tokenizer(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({str(i): i for i in range(32)}, unk_token="0"))
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="0", pad_token="0"
    ).save_pretrained(path)


class LatestKVTreeTest(unittest.TestCase):
    def test_conditional_branch_scores_and_ancestor_union(self):
        from draftfit.research.cast.tree import TreeNode
        from draftfit.research.valuspec import (
            conditional_score,
            leaf_paths,
            lower_quantile,
            retain_branches,
        )

        scorer = target_model()
        prefix, branch = [1, 2], [3, 4, 5]
        with torch.no_grad():
            expected = sum(
                float(
                    scorer(torch.tensor([prefix + branch[:i]]))
                    .logits[0, -1]
                    .log_softmax(-1)[token]
                )
                for i, token in enumerate(branch)
            ) / len(branch)
        self.assertAlmostEqual(
            conditional_score(scorer, prefix, branch), expected, places=6
        )
        nodes = [
            TreeNode(1, 1, -1, -9),
            TreeNode(2, 2, 0, -9.1),
            TreeNode(3, 1, -1, -1),
        ]
        paths = leaf_paths(nodes)
        self.assertEqual(paths, [[0, 1], [2]])
        kept = retain_branches(nodes, paths, [-0.5, -2], -1)
        self.assertEqual([n.token for n in kept], [1, 2])
        self.assertEqual([n.parent for n in kept], [-1, 0])
        self.assertEqual(retain_branches(nodes, paths, [-2, -2], -1), [])
        self.assertEqual(lower_quantile([-4, -2, 0], 0.25), -3)

    def test_valuspec_real_cast_greedy_parity_including_empty_tree(self):
        from draftfit.research.cast.decode import cast_generate
        from draftfit.research.valuspec import ConditionalBranchFilter

        target, draft = target_model(), draft_model(4).eval()
        ids = torch.tensor([[1, 2, 3]])
        expected = target.generate(
            ids, max_new_tokens=9, do_sample=False, eos_token_id=None, pad_token_id=0
        )
        for threshold in (-100.0, 0.0, -3.45):
            actual, stats = cast_generate(
                draft,
                target,
                ids,
                9,
                [],
                budget=7,
                branch_k=2,
                branch_filter=ConditionalBranchFilter(target, threshold),
                return_stats=True,
            )
            self.assertTrue(torch.equal(actual, expected), (actual, expected))
            self.assertTrue(stats["valuation"])
            if threshold == 0:
                self.assertTrue(
                    all(s["retained_tokens"] == 0 for s in stats["valuation"])
                )
        with self.assertRaisesRegex(ValueError, "greedy"):
            cast_generate(
                draft,
                target,
                ids,
                9,
                [],
                temperature=0.7,
                branch_filter=ConditionalBranchFilter(target, -3),
            )

    def test_valuspec_calibration_runner_and_split_guard(self):
        from draftfit.research.config import CastConfig
        from draftfit.research.runner import run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, draft = target_model(), draft_model(4).eval()
            target.save_pretrained(root / "target")
            draft.save_pretrained(root / "draft")
            save_tokenizer(root / "target")
            (root / "eval.jsonl").write_text(
                json.dumps({"input_ids": [1, 2, 3]}) + "\n"
            )
            (root / "cal.jsonl").write_text(json.dumps({"input_ids": [2, 4, 3]}) + "\n")
            cfg = CastConfig(
                method="cast",
                model={
                    "target_model_path": str(root / "target"),
                    "draft_checkpoint_path": str(root / "draft"),
                },
                data={"input_path": str(root / "eval.jsonl")},
                speculation={"budget": 7, "branch_k": 2},
                generation={"max_new_tokens": 7},
                valuation={
                    "scorer": {"target_model_path": str(root / "target")},
                    "calibration": {"input_path": str(root / "cal.jsonl")},
                },
                output_dir=str(root / "result"),
            )
            result = run(cfg)
            expected = target.generate(
                torch.tensor([[1, 2, 3]]),
                max_new_tokens=7,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=0,
            )[0, 3:].tolist()
            self.assertEqual(result["answers"][0]["token_ids"], expected)
            self.assertEqual(result["valuation"]["calibration_count"], 1)
            (root / "cal.jsonl").write_text((root / "eval.jsonl").read_text())
            cfg.output_dir = str(root / "bad")
            with self.assertRaisesRegex(ValueError, "disjoint"):
                run(cfg)
            self.assertFalse((root / "bad").exists())

    def test_rift_real_kv_gather_and_exact_target_output(self):
        from draftfit.research.config import RIFTOptions
        from draftfit.research.reference_decode import generate
        from draftfit.research.rift import (
            CompressedARProposer,
            attention_scores,
            select_positions,
        )

        target, draft = target_model(), target_model()
        prompt = torch.tensor([[1, 2, 3, 4, 5, 6]])
        expected = target.generate(
            prompt, max_new_tokens=9, do_sample=False, eos_token_id=None, pad_token_id=0
        )
        for budget in (2, 64):
            proposer = CompressedARProposer(
                draft, RIFTOptions(context_budget=budget, draft_tokens=4)
            )
            actual, stats = generate(target, prompt, proposer, max_new_tokens=9)
            self.assertTrue(torch.equal(actual, expected))
            self.assertGreater(stats["accepted_tokens"], 0)
            if budget == 2:
                for row in proposer.rounds:
                    self.assertLess(
                        row["active_context_kv_bytes_after_gather"],
                        row["dense_history_kv_bytes_before_gather"],
                    )
                    self.assertTrue(
                        all(
                            len(head) == 2
                            for layer in row["selected_absolute_positions"]
                            for head in layer
                        )
                    )
            else:
                self.assertEqual(stats["accepted_tokens"], stats["proposed_tokens"])
        scores = torch.tensor([[1.0, 7.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertEqual(select_positions(scores, 4, 0.5).tolist(), [[1, 2, 6, 7]])
        attention = torch.tensor([[[[0.2, 0.8]], [[0.6, 0.4]]]])
        torch.testing.assert_close(
            attention_scores(attention, 1, 64), torch.tensor([[0.4, 0.6]])
        )

    def test_rift_config_runner(self):
        from draftfit.research.config import RIFTConfig
        from draftfit.research.runner import run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = target_model()
            model.save_pretrained(root / "model")
            save_tokenizer(root / "model")
            (root / "input.jsonl").write_text(
                json.dumps({"input_ids": [1, 2, 3, 4]}) + "\n"
            )
            result = run(
                RIFTConfig(
                    method="rift",
                    model={
                        "target_model_path": str(root / "model"),
                        "draft_model_path": str(root / "model"),
                    },
                    data={"input_path": str(root / "input.jsonl")},
                    cache={"context_budget": 2, "draft_tokens": 3},
                    generation={"max_new_tokens": 5},
                    diagnostics={
                        "compare_ar": True,
                        "record_acceptance": True,
                        "warmup": 1,
                        "repetitions": 2,
                    },
                    output_dir=str(root / "result"),
                )
            )
            self.assertEqual(len(result["answers"][0]["token_ids"]), 5)
            self.assertFalse(result["production_engine_speed"])
            stats = result["answers"][0]["stats"]
            self.assertTrue(stats["measurement"]["paired_ar"]["token_ids_equal"])
            self.assertEqual(len(stats["measurement"]["generation_seconds"]), 2)
            self.assertTrue(stats["position_bins"])

    def test_treewy_matches_independent_recurrence_and_every_commit(self):
        from draftfit.research.treewy import solve_tree

        rng = torch.Generator().manual_seed(12)
        n, h, kd, vd = 6, 2, 3, 5
        q, k = (torch.randn(n, h, kd, generator=rng) for _ in range(2))
        v = torch.randn(n, h, vd, generator=rng)
        g, beta = -torch.rand(n, h, generator=rng), torch.rand(n, h, generator=rng)
        s0 = torch.randn(h, vd, kd, generator=rng)
        parent = torch.tensor([-1, 0, 0, 1, 2, 1])  # parent order, deliberately not DFS
        qn, kn = (x / x.norm(dim=-1, keepdim=True) for x in (q, k))
        states, outputs = [], []
        for t, p in enumerate(parent):
            state = s0 if p == -1 else states[p]
            decayed = state * g[t].exp()[:, None, None]
            prediction = torch.einsum("hvk,hk->hv", decayed, kn[t])
            state = decayed + beta[t, :, None, None] * torch.einsum(
                "hv,hk->hvk", v[t] - prediction, kn[t]
            )
            states.append(state)
            outputs.append(torch.einsum("hvk,hk->hv", state, qn[t]) / kd**0.5)
        for accepted in range(n):
            output, state = solve_tree(
                q, k, v, g, beta, parent, s0, accepted_node=accepted
            )
            torch.testing.assert_close(
                output, torch.stack(outputs), atol=2e-6, rtol=1e-5
            )
            torch.testing.assert_close(state, states[accepted], atol=2e-6, rtol=1e-5)
        _, unchanged = solve_tree(q, k, v, g, beta, parent, s0)
        torch.testing.assert_close(unchanged, s0, rtol=0, atol=0)
        for bad in (
            torch.tensor([-1, -1, 0, 1, 2, 1]),
            torch.tensor([-1, 2, 0, 1, 2, 1]),
        ):
            with self.assertRaisesRegex(ValueError, "connected root"):
                solve_tree(q, k, v, g, beta, bad, s0)
        with self.assertRaisesRegex(ValueError, "accepted_node"):
            solve_tree(q, k, v, g, beta, parent, s0, accepted_node=-1)


if __name__ == "__main__":
    unittest.main()
