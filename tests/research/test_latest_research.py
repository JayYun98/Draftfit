"""Behavioral checks for current-paper policies on actual tiny local targets."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from draftfit.research.config import ADAPTER
from draftfit.research.memory import NgramMemory, Proposal, SpecMemMemory
from draftfit.research.reference_decode import generate, verify_token
from tests.test_research_methods import target_model


class LatestResearchTest(unittest.TestCase):
    def test_ngram_snapshot_history_aging_and_reload(self):
        memory = NgramMemory(
            n_min=2, n_max=3, window_steps=2, max_prompts=2, max_rollouts_per_prompt=2
        )
        memory.add_verified("a", [1, 2, 3, 4], step=0)
        snapshot = memory.snapshot("a", step=0)
        memory.add_verified("a", [1, 2, 5, 6], step=1)
        self.assertEqual(snapshot.propose([1, 2], 2).tokens, (3, 4))
        self.assertEqual(memory.snapshot("a", step=1).successors([1, 2]), {3: 1, 5: 1})
        self.assertEqual(memory.snapshot("a", step=2).propose([1, 2], 2).tokens, (5, 6))
        restored = NgramMemory(
            n_min=2, n_max=3, window_steps=2, max_prompts=2, max_rollouts_per_prompt=2
        )
        restored.load_state_dict(json.loads(json.dumps(memory.state_dict())))
        self.assertEqual(restored.state_dict(), memory.state_dict())
        self.assertEqual(
            restored.snapshot("other", step=2).propose([1, 2], 2).tokens, ()
        )
        self.assertEqual(memory.snapshot("a", step=3).propose([1, 2], 2).tokens, ())
        with self.assertRaisesRegex(ValueError, "monotonic"):
            memory.snapshot("a", step=2)

    def test_memory_bounds_and_proposal_validation(self):
        memory = NgramMemory(
            max_prompts=1, max_rollouts_per_prompt=1, max_tokens_per_rollout=4
        )
        memory.add_verified("a", [1, 2], step=0)
        memory.add_verified("a", [1, 3], step=0)
        self.assertEqual(memory.snapshot("a", step=0).successors([1]), {3: 1})
        memory.add_verified("b", [1, 4], step=1)
        self.assertEqual(memory.snapshot("a", step=1).counts, {})
        with self.assertRaisesRegex(ValueError, "bound"):
            memory.add_verified("b", [1] * 5, step=1)
        with self.assertRaisesRegex(ValueError, "normalized"):
            Proposal((2,), ({2: 0.2},))

    def test_specmem_user_isolation_shared_fallback_lru_and_reload(self):
        memory = SpecMemMemory(capacity=2, threshold=0.8)
        memory.add_shared_seed([0, 0, 1], [9])
        memory.add_verified("alice", "one", [1, 0, 0], [1])
        memory.add_verified("alice", "two", [0, 1, 0], [2])
        self.assertEqual(memory.retrieve("bob", [1, 0, 0], schema_tokens=[8])[0], (8,))
        self.assertEqual(memory.retrieve("bob", [0, 0, 1])[1]["source"], "shared_seed")
        self.assertEqual(memory.retrieve("alice", [1, 0, 0])[0], (1,))
        memory.add_verified("alice", "three", [0, 0, 1], [3])
        self.assertEqual(list(memory.users["alice"]), ["one", "three"])
        restored = SpecMemMemory(capacity=2, threshold=0.8)
        restored.load_state_dict(json.loads(json.dumps(memory.state_dict())))
        self.assertEqual(restored.retrieve("alice", [1, 0, 0])[0], (1,))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            restored.retrieve("alice", [1, 0])

    def test_real_target_memory_verify_rejection_caps_and_eos(self):
        target = target_model()
        ids = torch.tensor([[1, 2, 3]])
        gold, _ = generate(target, ids, lambda *_: Proposal(), max_new_tokens=7)
        memory = NgramMemory()
        memory.add_verified("prompt", gold[0].tolist(), step=0)
        snapshot = memory.snapshot("prompt", step=1)
        output, stats = generate(
            target,
            ids,
            lambda context, remaining, _: snapshot.propose(context, min(4, remaining)),
            max_new_tokens=7,
        )
        self.assertTrue(torch.equal(output, gold))
        self.assertGreater(stats["accepted_tokens"], 0)
        self.assertLess(stats["target_forwards"], 7)
        wrong = (int(gold[0, 3]) + 1) % target.config.vocab_size
        output, stats = generate(
            target,
            ids,
            lambda *_: Proposal((wrong,), ({wrong: 1.0},)),
            max_new_tokens=1,
        )
        self.assertTrue(torch.equal(output, gold[:, :4]))
        self.assertEqual(stats["accepted_tokens"], 0)
        output, stats = generate(target, ids, lambda *_: Proposal(), max_new_tokens=0)
        self.assertTrue(torch.equal(output, ids))
        self.assertEqual(stats["target_forwards"], 0)
        output, _ = generate(
            target,
            ids,
            lambda *_: Proposal(),
            max_new_tokens=7,
            stop_token_ids=[int(gold[0, 3])],
        )
        self.assertEqual(output.shape[1], ids.shape[1] + 1)

    def test_frequency_proposals_and_residual_match_target_sampling(self):
        p = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
        q = {0: 0.7, 1: 0.2, 2: 0.1}
        generator = torch.Generator().manual_seed(719)
        counts = torch.zeros(3)
        rejections = 0
        for _ in range(6000):
            proposal = int(
                torch.multinomial(
                    torch.tensor(list(q.values())), 1, generator=generator
                )
            )
            emitted, accepted = verify_token(p, proposal, q, generator=generator)
            counts[emitted] += 1
            rejections += int(not accepted)
        self.assertGreater(rejections, 1000)
        self.assertTrue(torch.allclose(counts / counts.sum(), p.float(), atol=0.025))

    def test_real_cli_memory_writeback_and_state_continuation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target_model().save_pretrained(root / "target")
            rows = [
                {"id": i, "prompt_id": "repeat", "step": i, "input_ids": [1, 2, 3]}
                for i in range(2)
            ]
            data = root / "rows.jsonl"
            data.write_text("".join(json.dumps(row) + "\n" for row in rows))
            config = {
                "method": "ngramrl",
                "model": {"target_model_path": str(root / "target")},
                "data": {"input_path": str(data)},
                "generation": {"max_new_tokens": 7},
                "output_dir": str(root / "out"),
            }
            recipe = root / "recipe.json"
            recipe.write_text(json.dumps(config))
            env = dict(
                os.environ,
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false",
            )
            plan = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "draftfit.research",
                    "-c",
                    str(recipe),
                    "--plan",
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(plan.returncode, 0, plan.stderr)
            self.assertFalse((root / "out").exists())
            execution = subprocess.run(
                [sys.executable, "-m", "draftfit.research", "-c", str(recipe)],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(execution.returncode, 0, execution.stderr)
            result = json.loads((root / "out/result.json").read_text())
            self.assertEqual(
                result["answers"][0]["token_ids"], result["answers"][1]["token_ids"]
            )
            self.assertGreater(result["answers"][1]["stats"]["accepted_tokens"], 0)
            data.write_text(json.dumps(dict(rows[0], step=2)) + "\n")
            config["memory"] = {"state_path": str(root / "out/memory.json")}
            config["output_dir"] = str(root / "continued")
            recipe.write_text(json.dumps(config))
            continuation = subprocess.run(
                [sys.executable, "-m", "draftfit.research", "-c", str(recipe)],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(continuation.returncode, 0, continuation.stderr)
            continued = json.loads((root / "continued/result.json").read_text())
            self.assertGreater(continued["answers"][0]["stats"]["accepted_tokens"], 0)

    def test_specmem_runner_writes_only_verified_local_output(self):
        from draftfit.research.runner import run

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = target_model()
            target.save_pretrained(root / "target")
            with torch.no_grad():
                eos = int(target(torch.tensor([[1, 2, 3]])).logits[0, -1].argmax())
            data = root / "rows.jsonl"
            rows = [
                {"input_ids": [1, 2, 3], "user_id": user, "query_embedding": [1, 0]}
                for user in ("alice", "bob", "alice")
            ]
            data.write_text("".join(json.dumps(row) + "\n" for row in rows))
            config = ADAPTER.validate_python(
                {
                    "method": "specmem",
                    "model": {"target_model_path": str(root / "target")},
                    "data": {"input_path": str(data)},
                    "generation": {"max_new_tokens": 4, "stop_token_ids": [eos]},
                    "output_dir": str(root / "out"),
                }
            )
            result = run(config)
            self.assertEqual(
                [row["retrieval"]["source"] for row in result["answers"]],
                ["schema", "schema", "user"],
            )
            self.assertGreater(result["answers"][2]["stats"]["accepted_tokens"], 0)
            self.assertFalse(result["tool_execution_performed"])
            self.assertTrue(
                all(row["writeback"]["stored"] for row in result["answers"])
            )
            with self.assertRaises(FileExistsError):
                run(config)
            config.output_dir = str(root / "capped")
            config.generation.stop_token_ids = []
            capped = run(config)
            self.assertTrue(
                all(not row["writeback"]["stored"] for row in capped["answers"])
            )
            self.assertEqual(
                [row["retrieval"]["source"] for row in capped["answers"]],
                ["schema"] * 3,
            )


if __name__ == "__main__":
    unittest.main()
