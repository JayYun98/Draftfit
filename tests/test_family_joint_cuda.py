"""Opt-in CUDA 13 gate; synthetic features, real training/CLI/export/resume.

Run ``python -m tests.test_family_joint_cuda`` in a provisioned CUDA runtime.
This does not certify pretrained target quality or serving throughput.
"""
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time
import unittest

import torch

from tests import test_family_joint_cpu


def require_cuda13():
    if not torch.cuda.is_available() or int((torch.version.cuda or "0").split(".")[0]) < 13:
        raise RuntimeError("This gate requires actual CUDA 13+ PyTorch and a visible GPU")


@unittest.skipUnless(__name__ == "__main__" or os.environ.get("DRAFTFIT_TEST_CUDA") == "1",
                     "opt in with python -m tests.test_family_joint_cuda")
class FamilyJointCUDA(test_family_joint_cpu.FamilyJointCPU):
    execution_device = "cuda"
    cli_module = "tests.test_family_joint_cuda"
    cli_prefix = ("--run-cli",)

    @classmethod
    def setUpClass(cls):
        require_cuda13()

    def test_long_bf16_accumulation_on_cuda(self):
        from types import SimpleNamespace
        from draftfit.training.joint_family import _backward_window
        from draftfit.training.strategies.base import StepOutput
        parameter = torch.nn.Parameter(torch.tensor(1.0, device="cuda", dtype=torch.bfloat16))
        model = torch.nn.ParameterDict({"shared": parameter})

        class Strategy:
            def forward_loss(self, batch):
                numerator = parameter.float() * batch
                return StepOutput(numerator, {}, loss_terms=(numerator, numerator.new_tensor(1.0)))

        _backward_window([SimpleNamespace(name="a"), SimpleNamespace(name="b")],
                         [Strategy(), Strategy()],
                         [SimpleNamespace(draft_model=model), SimpleNamespace(draft_model=model)],
                         [1.0] * 512)
        self.assertEqual(parameter.grad.item(), 1.0)

    def cli(self, root, *args):
        result = super().cli(root, *args)
        if evidence := os.environ.get("DRAFTFIT_CUDA_EVIDENCE"):
            destination = Path(evidence) / self.id().split(".")[-1]
            destination.mkdir(parents=True, exist_ok=True)
            with (destination / "commands.jsonl").open("a") as stream:
                stream.write(json.dumps({"args": args, "stdout": result.stdout,
                                         "stderr": result.stderr, "exit": result.returncode}) + "\n")
            shutil.copytree(root, destination / "artifacts", dirs_exist_ok=True)
        if "--resume-from" in args:
            reference = Path(args[args.index("--resume-from") + 1]).parent / "joint-step2.pt"
            manifest = json.loads((root / args[args.index("--manifest") + 1]).read_text())
            resumed = root / manifest["output_dir"] / "joint-step2.pt"

            def same(left, right):
                if isinstance(left, torch.Tensor):
                    self.assertTrue(torch.equal(left, right))
                elif isinstance(left, dict):
                    self.assertEqual(left.keys(), right.keys())
                    for key in left:
                        same(left[key], right[key])
                elif isinstance(left, (list, tuple)):
                    self.assertEqual(len(left), len(right))
                    for a, b in zip(left, right):
                        same(a, b)
                else:
                    self.assertEqual(left, right)

            same(torch.load(reference, map_location="cpu", weights_only=True),
                 torch.load(resumed, map_location="cpu", weights_only=True))
            print("CUDA_FULL_CHECKPOINT_EQUAL", flush=True)
        if args[0] == "family" and "--plan" not in args:
            records = [json.loads(line[len("CUDA_GATE "):]) for line in result.stdout.splitlines()
                       if line.startswith("CUDA_GATE ")]
            self.assertEqual(len(records), 1, result.stdout)
            self.assertGreater(records[0]["cuda_parameter_tensors"], 0)
            self.assertGreater(records[0]["peak_allocated_bytes"], 0)
            print("CUDA_GATE " + json.dumps(records[0]), flush=True)
        return result


def run_cli():
    require_cuda13()
    import draftfit.training.joint_family as joint
    original = joint._tie_backbones
    observed = {}

    def checked(bundles, **kwargs):
        count = original(bundles, **kwargs)
        params = [p for b in bundles for p in b.draft_model.parameters()]
        assert params and all(p.is_cuda for p in params), "CPU fallback in CUDA gate"
        observed["cuda_parameter_tensors"] = len(params)
        return count

    joint._tie_backbones = checked
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    sys.argv = ["draftfit", *sys.argv[2:]]
    try:
        runpy.run_module("draftfit.cli", run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    if observed:
        torch.cuda.synchronize()
        observed.update(torch=torch.__version__, cuda=torch.version.cuda,
                        gpu=torch.cuda.get_device_name(),
                        seconds=time.monotonic() - started,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
        print("CUDA_GATE " + json.dumps(observed), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--run-cli"]:
        run_cli()
    else:
        require_cuda13()
        unittest.main(verbosity=2)
