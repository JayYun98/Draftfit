"""Real child processes exercise scheduling, handoffs, and failure cleanup."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from draftfit.config import Config
from draftfit.family import FamilyStage, _run_concurrent, run_family

WORKER = r"""
import json, os, sys, time
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
p = Path(cfg['output_dir'])
root = p.parent
name = cfg['run_id']
(p/'started').write_text(str(time.time()))
if name in ('first', 'second'):
    until = time.monotonic() + 5
    while not all((root/n/'started').exists() for n in ('first', 'second')):
        if time.monotonic() > until: sys.exit(7)
        time.sleep(.01)
    if (root/'fail').exists() and name == 'first': sys.exit(9)
    if (root/'fail').exists() and name == 'second': time.sleep(30)
    time.sleep(.15)
if name == 'child':
    assert (root/'first'/'first-latest'/'training_state.pt').exists()
if not (root/'missing-eval').exists():
    (p/'metrics.jsonl').write_text(json.dumps({'step':2, 'eval/avg_loss':1.,'eval/avg_acc':.5})+'\n')
(p/'checkpoint').mkdir()
(p/'checkpoint'/'training_state.pt').write_text('test checkpoint')
(p/(name+'-latest')).symlink_to(p/'checkpoint', target_is_directory=True)
(p/'finished').write_text(str(time.time()))
"""


class DraftConfig:
    def to_json_file(self, path):
        Path(path).write_text("{}")


class FamilyConcurrencyTest(unittest.TestCase):
    def stages(self, root):
        return [
            FamilyStage(
                name,
                Config.model_validate(
                    {
                        "model": {"target_model_path": "target/model"},
                        "data": {"hidden_states_path": "/train"},
                        "training": {"strategy": "dflash"},
                        "run_id": name,
                        "output_dir": str(root / name),
                    }
                ),
                DraftConfig(),
                "first" if name == "child" else None,
            )
            for name in ("first", "second", "child")
        ]

    def run_workers(self, root, stages):
        real_popen = subprocess.Popen

        def start(command, **kwargs):
            return real_popen(
                [sys.executable, "-u", "-c", WORKER, command[-1]], **kwargs
            )

        with mock.patch("draftfit.family.subprocess.Popen", side_effect=start):
            return _run_concurrent(root, stages, 2)

    def test_independent_workers_overlap_and_handoff_waits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_workers(root, self.stages(root))
            self.assertEqual([r["status"] for r in result["runs"]], ["complete"] * 3)
            facts = {r["name"]: r for r in result["runs"]}
            self.assertLess(
                facts["second"]["started_at"], facts["first"]["finished_at"]
            )
            self.assertGreaterEqual(
                facts["child"]["started_at"], facts["first"]["finished_at"]
            )
            self.assertEqual(
                json.loads((root / "results.json").read_text()), result["runs"]
            )
            self.assertTrue(all((root / n / "worker.log").exists() for n in facts))

    def test_failure_stops_peer_and_blocks_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fail").touch()
            result = self.run_workers(root, self.stages(root))
            self.assertEqual(
                [r["status"] for r in result["runs"]],
                ["failed", "cancelled", "blocked"],
            )
            self.assertEqual(result["runs"][0]["exit_code"], 9)
            with self.assertRaises(ProcessLookupError):
                os.kill(result["runs"][1]["pid"], 0)
            self.assertFalse((root / "child" / "started").exists())

    def test_exit_zero_without_eval_is_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "missing-eval").touch()
            result = self.run_workers(root, self.stages(root))
            self.assertEqual(result["runs"][0]["status"], "failed")
            self.assertEqual(result["runs"][2]["status"], "blocked")

    def test_invalid_parallelism_fails_before_output(self):
        for jobs in (0, -1, True):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                run_family("absent.json", jobs=jobs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "new-output"
            with mock.patch(
                "draftfit.family.build_family_plan",
                return_value=(root, self.stages(root)),
            ):
                with mock.patch.dict(os.environ, {"RANK": "0"}):
                    with self.assertRaisesRegex(ValueError, "torchrun"):
                        run_family("placeholder", jobs=2)
            self.assertFalse(root.exists())

    def test_parallel_plan_validates_topology_and_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "new-output"
            stages = self.stages(root)
            with mock.patch(
                "draftfit.family.build_family_plan", return_value=(root, stages)
            ):
                result = run_family("placeholder", jobs=3, plan=True)
                self.assertEqual(result["jobs"], 3)
                self.assertEqual(result["execution"], "concurrent")
                self.assertEqual(result["runs"][2]["from"], "first")
                self.assertFalse(root.exists())
                stages[0].config.deployment.trainer.nproc_per_node = 2
                with self.assertRaisesRegex(ValueError, "single-rank"):
                    run_family("placeholder", jobs=2, plan=True)

    @unittest.skipUnless(os.name == "posix", "process-group signal test")
    def test_interrupt_reaps_owned_children(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = r"""
import json,sys,subprocess,time
from pathlib import Path
from draftfit.family import FamilyStage,_run_concurrent
from draftfit.config import Config
from draftfit.cli import _worker_signal_unwind,_WorkerTermination
from unittest.mock import patch
class Draft:
    def to_json_file(self,path):Path(path).write_text('{}')
root=Path(sys.argv[1]);stages=[]
for name in ('a','b'):
    cfg=Config.model_validate({'model':{'target_model_path':'target'},
        'training':{'strategy':'dflash'},'data':{'hidden_states_path':'/train'},
        'run_id':name,'output_dir':str(root/name)})
    stages.append(FamilyStage(name,cfg,Draft(),None))
real=subprocess.Popen
def spawn(command,**kwargs):
    return real([sys.executable,'-c','import time;time.sleep(30)'],**kwargs)
with patch('draftfit.family.subprocess.Popen',side_effect=spawn):
    try:
        with _worker_signal_unwind():_run_concurrent(root,stages,2)
    except _WorkerTermination as exc:sys.exit(128+exc.signum)
"""
            process = subprocess.Popen([sys.executable, "-c", script, str(root)])
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if (root / "results.json").exists():
                        rows = json.loads((root / "results.json").read_text())
                        if all(r["status"] == "running" for r in rows):
                            break
                    time.sleep(0.02)
                else:
                    self.fail("workers never started")
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=10), 143)
                rows = json.loads((root / "results.json").read_text())
                self.assertEqual(
                    [r["status"] for r in rows], ["cancelled", "cancelled"]
                )
                for row in rows:
                    with self.assertRaises(ProcessLookupError):
                        os.kill(row["pid"], 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    unittest.main()
