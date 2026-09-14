"""CPU checks ensure the gate cannot pass without genuine round coverage."""

import contextlib
import importlib.util
import io
import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

spec = importlib.util.spec_from_file_location(
    "validator", Path(__file__).parents[2] / "scripts/validate_ling_replay.py"
)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class CoverageTest(unittest.TestCase):
    def test_port_probe_still_rejects_an_active_listener(self):
        with socket.socket() as listener, tempfile.TemporaryDirectory() as directory:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with self.assertRaises(OSError):
                validator.capture(
                    Path(directory) / "capture", listener.getsockname()[1], []
                )
            self.assertFalse((Path(directory) / "capture").exists())

    def test_default_hook_commits_before_any_state_inspection(self):
        for diagnostic in (False, True):
            with (
                self.subTest(diagnostic=diagnostic),
                tempfile.TemporaryDirectory() as directory,
            ):
                events = []

                class Worker:
                    _need_mamba_verify_commit = True
                    verify_num_draft_tokens = 8

                    def _commit_target_mamba_states_after_verify(self, **kwargs):
                        events.append("commit")
                        return "committed"

                state = SimpleNamespace(temporal=torch.ones(1, 1, 2), conv=[])

                def get_state():
                    events.append("inspect")
                    return state

                pool = SimpleNamespace(
                    mamba_pool=SimpleNamespace(
                        replayssm_spec_fold=True, replayssm_is_kda=True
                    ),
                    get_speculative_mamba2_params_all_layers=get_state,
                )
                backend = SimpleNamespace(
                    req_to_token_pool=pool,
                    forward_metadata=SimpleNamespace(
                        mamba_cache_indices=torch.tensor([0])
                    ),
                )
                worker = Worker()
                worker.target_worker = SimpleNamespace(
                    model_runner=SimpleNamespace(
                        attn_backend=SimpleNamespace(linear_attn_backend=backend)
                    )
                )
                module = SimpleNamespace(DSparkWorkerV2=Worker)
                found = SimpleNamespace(
                    loader=SimpleNamespace(exec_module=lambda module: None)
                )
                with (
                    mock.patch.object(
                        validator.sys, "meta_path", list(validator.sys.meta_path)
                    ),
                    mock.patch.dict(
                        validator.os.environ,
                        {
                            "LING_REPLAY_CAPTURE": directory,
                            "LING_REPLAY_CAPTURE_INITIAL": "1" if diagnostic else "0",
                        },
                    ),
                ):
                    validator.install_hook()
                    with mock.patch.object(
                        validator.importlib.machinery.PathFinder,
                        "find_spec",
                        return_value=found,
                    ):
                        validator.sys.meta_path[0].find_spec(
                            "sglang.srt.speculative.dspark_components.dspark_worker_v2"
                        ).loader.exec_module(module)
                    result = worker._commit_target_mamba_states_after_verify(
                        batch=SimpleNamespace(
                            reqs=[SimpleNamespace(rid=validator.CAPTURE_RID)]
                        ),
                        commit_lens=torch.tensor([2]),
                        seq_lens_pre_verify=torch.tensor([16]),
                        seq_lens_post_verify=torch.tensor([18]),
                    )
                self.assertEqual(result, "committed")
                self.assertEqual(
                    events,
                    ["inspect", "commit"] if diagnostic else ["commit", "inspect"],
                )

    def test_hook_selects_only_the_validation_request(self):
        def batch(*rids):
            return SimpleNamespace(reqs=[SimpleNamespace(rid=rid) for rid in rids])

        self.assertTrue(validator.is_capture_batch(batch(validator.CAPTURE_RID)))
        self.assertFalse(validator.is_capture_batch(batch("health")))
        self.assertFalse(validator.is_capture_batch(batch("warmup")))
        self.assertFalse(
            validator.is_capture_batch(batch(validator.CAPTURE_RID, "health"))
        )

    def test_accept_reject_and_terminal_trimming(self):
        def rounds(lengths):
            return [{"commit": [n], "window": 8} for n in lengths]

        validator.check_coverage(rounds([8, 3, 1]))
        for lengths in ([3], [8, 8, 1], [1, 1, 1]):
            with self.assertRaises(ValueError):
                validator.check_coverage(rounds(lengths))

    def test_real_file_comparison_rejects_divergence(self):
        with tempfile.TemporaryDirectory() as temporary:
            legacy, replay = [Path(temporary) / name for name in ("legacy", "replay")]
            args = ["--mamba-ssm-dtype", "float32", "--disable-radix-cache"]
            captures = []
            for mode, directory in enumerate((legacy, replay)):
                directory.mkdir()
                (directory / "args.json").write_text(
                    json.dumps(
                        args
                        + (
                            [
                                "--enable-linear-replayssm-spec",
                                "--linear-replayssm-cache-len",
                                "16",
                            ]
                            if mode
                            else []
                        )
                    )
                )
                (directory / "output.json").write_text(
                    json.dumps({"output_ids": [4, 5, 6]})
                )
                (directory / "request.json").write_text(
                    json.dumps(
                        {"text": "test", "sampling_params": {"max_new_tokens": 24}}
                    )
                )
                rounds = [
                    {
                        "rid": validator.CAPTURE_RID,
                        "pre": [i],
                        "post": [i + n],
                        "commit": [n],
                        "window": 8,
                        "replay": bool(mode),
                        "temporal": torch.ones(2, 1, 3),
                        "conv": [torch.zeros(2, 1, 3)],
                    }
                    for i, n in enumerate((8, 3, 1))
                ]
                for index, record in enumerate(rounds):
                    torch.save(record, directory / f"commits-1-{index:06d}.pt")
                captures.append(rounds)
            with contextlib.redirect_stdout(io.StringIO()):
                validator.compare(legacy, replay, 1e-5, 1e-4)
            captures[1][0]["temporal"][0, 0, 0] = 2
            torch.save(captures[1][0], replay / "commits-1-000000.pt")
            with self.assertRaisesRegex(AssertionError, "round 0: temporal"):
                validator.compare(legacy, replay, 1e-5, 1e-4)
            captures[1][0]["temporal"].fill_(1)
            for records, directory in zip(captures, (legacy, replay)):
                records[0]["initial_temporal"] = torch.ones(2, 1, 3)
                torch.save(records[0], directory / "commits-1-000000.pt")
            captures[1][0]["initial_temporal"][0, 0, 0] = 2
            torch.save(captures[1][0], replay / "commits-1-000000.pt")
            with self.assertRaisesRegex(AssertionError, "round 0: initial_temporal"):
                validator.compare(legacy, replay, 1e-5, 1e-4)
            captures[1][0]["initial_temporal"].fill_(1)
            captures[1][0]["replay"] = False
            torch.save(captures[1][0], replay / "commits-1-000000.pt")
            with self.assertRaisesRegex(ValueError, "actual KDA"):
                validator.compare(legacy, replay, 1e-5, 1e-4)
            captures[1][0]["replay"] = True
            torch.save(captures[1][0], replay / "commits-1-000000.pt")
            request = (replay / "request.json").read_text()
            (replay / "request.json").write_text('{"text": "different"}')
            with self.assertRaisesRegex(ValueError, "requests differ"):
                validator.compare(legacy, replay, 1e-5, 1e-4)
            (replay / "request.json").write_text(request)
            final = replay / "commits-1-000002.pt"
            gap = replay / "commits-1-000003.pt"
            final.rename(gap)
            with self.assertRaisesRegex(ValueError, "contiguous"):
                validator.compare(legacy, replay, 1e-5, 1e-4)
            gap.rename(final)
            other_worker = replay / "commits-2-000002.pt"
            final.rename(other_worker)
            with self.assertRaisesRegex(ValueError, "one hooked worker"):
                validator.compare(legacy, replay, 1e-5, 1e-4)
            other_worker.rename(final)
            (replay / "output.json").write_text(json.dumps({"output_ids": [4, 5, 9]}))
            with self.assertRaisesRegex(ValueError, "token IDs differ"):
                validator.compare(legacy, replay, 1e-5, 1e-4)
            (replay / "FAILED").write_text("capture exceeds budget")
            with self.assertRaisesRegex(ValueError, "exceeds budget"):
                validator.compare(legacy, replay, 1e-5, 1e-4)


if __name__ == "__main__":
    unittest.main()
