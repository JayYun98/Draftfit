from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from draftfit.campaign import decide, digest, evaluate_report, run, validate
from draftfit.campaign_recipe import build


def config(jobs):
    return dict(schema='draftfit-campaign-v1', resources={'gpu': 1, 'validation': 1},
                budget=dict(max_hours=1, max_usd=1, hourly_usd=0, fixed_usd=0, reserve_usd=0), jobs=jobs)


def job(name, delay=0, **kwargs):
    code = ('import os,json,time; from pathlib import Path; time.sleep(' + str(delay) + '); '
            'Path(os.environ["DRAFTFIT_JOB_RESULT"]).write_text(json.dumps({"status":"complete"}))')
    return dict(id=name, kind='train', resource='gpu', argv=[sys.executable, '-c', code],
                timeout_seconds=5, depends_on=[], **kwargs)


def evaluation(training, *, speed=1.2):
    return dict(status='complete', schema='draftfit-campaign-evaluation-v1', comparison_id='fixed', comparison_fingerprint='fixed-env', split='validation',
                training_result_sha256=digest(training), candidates=[dict(name='dspark', exact_token_parity=True,
                domains=[dict(name='code', requests=32, matched_requests=32, output_tokens=100, baseline_output_tokens=100,
                              seconds=1, baseline_seconds=speed)])])


class CampaignTests(unittest.TestCase):
    def test_overlap_resource_limit_restart_and_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b, c = job('a', .4), job('b', .4), job('c', .4)
            b['resource'] = 'validation'
            cfg = config([a, b, c]); path = root / 'config.json'; path.write_text(json.dumps(cfg))
            output = root / 'run'
            self.assertEqual(run(path, output), 0)
            state = json.loads((output / 'state.json').read_text())['jobs']
            self.assertLess(state['b']['started_at'], state['a']['finished_at'])
            self.assertGreaterEqual(state['c']['started_at'], state['a']['finished_at'])
            self.assertEqual(run(path, output), 0)
            again = json.loads((output / 'state.json').read_text())['jobs']
            self.assertEqual([x['attempts'] for x in again.values()], [1, 1, 1])
            self.assertIn('Draftfit campaign', (output / 'index.html').read_text())
            cfg['budget']['max_hours'] = 2; path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, 'changed'):
                run(path, output)

    def test_fail_fast_stops_independent_workers_and_retains_logs(self):
        for failure in ('exit', 'timeout', 'receipt'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp)
                a, b, c = job('a', .2), job('b', 20), job('c')
                b['resource'] = 'validation'
                if failure == 'exit': a['argv'] = [sys.executable, '-c', 'raise SystemExit(4)']
                if failure == 'timeout': a['timeout_seconds'] = .05; a['argv'] = [sys.executable, '-c', 'import time; time.sleep(20)']
                if failure == 'receipt': a['argv'] = [sys.executable, '-c', 'print("missing receipt", flush=True)']
                cfg = config([a, b, c]); cfg['fail_fast'] = True
                path = p / 'config.json'; path.write_text(json.dumps(cfg))
                self.assertEqual(run(path, p / 'out'), 1)
                state = json.loads((p / 'out/state.json').read_text())
                self.assertEqual(state['status'], 'failed')
                self.assertEqual(state['jobs']['b']['reason'], 'peer_failed')
                self.assertEqual(state['jobs']['c']['attempts'], 0)
                self.assertTrue((p / 'out/a/stdout.log').is_file())
                with self.assertRaises(ProcessLookupError):
                    os.kill(state['jobs']['b']['pid'], 0)

    def test_failed_child_cannot_unlock_next(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); a, b = job('a'), job('b'); a['argv'] = [sys.executable, '-c', 'raise SystemExit(4)']; b['depends_on'] = ['a']
            c = p / 'c.json'; c.write_text(json.dumps(config([a, b])))
            self.assertEqual(run(c, p / 'out'), 1)
            state = json.loads((p / 'out/state.json').read_text())
            self.assertEqual(state['jobs']['b']['status'], 'skipped')

    def test_timeout_budget_and_bad_receipt(self):
        for mode in ('timeout', 'budget', 'receipt'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp); a = job('a', 2); cfg = config([a])
                if mode == 'timeout': a['timeout_seconds'] = .15
                if mode == 'budget': cfg['budget']['max_hours'] = .00005
                if mode == 'receipt': a['argv'] = [sys.executable, '-c', 'pass']
                c = p / 'c.json'; c.write_text(json.dumps(cfg))
                self.assertNotEqual(run(c, p / 'out'), 0)
                state = json.loads((p / 'out/state.json').read_text())
                self.assertIn(state['jobs']['a']['status'], ('failed', 'stopped'))

    def test_validation_rejects_stale_nan_wrong_split_and_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); (p / 'train').mkdir(); receipt = p / 'train/result.json'; receipt.write_text('{"status":"complete"}')
            spec = dict(train_job='train', comparison_id='fixed', min_requests=32)
            good = evaluation(receipt)
            self.assertAlmostEqual(evaluate_report(spec, good, {}, p)['candidates'][0]['speedup'], 1.2)
            for field in ('parity', 'nan', 'split', 'stale', 'tokens', 'requests'):
                bad = copy.deepcopy(good)
                if field == 'parity': bad['candidates'][0]['exact_token_parity'] = False
                if field == 'nan': bad['candidates'][0]['domains'][0]['seconds'] = float('nan')
                if field == 'split': bad['split'] = 'test'
                if field == 'stale': bad['training_result_sha256'] = 'stale'
                if field == 'tokens': bad['candidates'][0]['domains'][0]['baseline_output_tokens'] = 101
                if field == 'requests': bad['candidates'][0]['domains'][0]['matched_requests'] = 31
                with self.subTest(field=field), self.assertRaises(ValueError):
                    evaluate_report(spec, bad, {}, p)

    def test_gates_rank_domains_plateau_and_changed_runtime(self):
        reports = []
        for speed in (1.2, 1.205, 1.21):
            reports.append(dict(comparison_id='same', comparison_fingerprint='same', candidates=[
                dict(name='dspark', speedup=speed, domains={'code': speed}),
                dict(name='dflash', speedup=1.1, domains={'code': .8})]))
        state = {'jobs': {str(i): {'evaluation': r} for i, r in enumerate(reports)}}
        gate = dict(evaluations=['0'], top_k=1, min_speedup=1.05, min_domain_speedup=.9, min_relative_gain=.01)
        self.assertTrue(decide(gate, state)[0])
        self.assertFalse(decide({**gate, 'family': 'dflash'}, state)[0])
        gate['evaluations'] = ['0', '1', '2']
        self.assertEqual(decide(gate, state)[1], 'plateau')
        reports[2]['comparison_fingerprint'] = 'new-hardware'
        with self.assertRaises(ValueError): decide(gate, state)

    def test_recipe_all_arms_fixed_horizon_and_branches(self):
        spec = dict(targets=[dict(name='a', manifest='/tmp/family.json', comparison_id='fixed', evaluate=['eval'],
                    branches=[dict(name='head', family='dspark', train=['train'], evaluate=['eval'])])], budget=config([])['budget'])
        plan = build(spec)
        self.assertEqual(len(plan['jobs']), 12)
        last = next(j for j in plan['jobs'] if j['id'] == 'a-train-30000')
        self.assertIn('30000', last['argv'])
        self.assertEqual(last['gate']['evaluations'], ['a-eval-3000', 'a-eval-10000', 'a-eval-20000'])
        self.assertEqual(next(j for j in plan['jobs'] if j['id'] == 'a-branch-head')['gate']['family'], 'dspark')

    def test_native_joint_milestones_match_uninterrupted(self):
        import torch
        from tests.test_family_joint_cpu import fixture
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); fixture(p, True)
            env = os.environ.copy()
            env.update(PYTHONPATH=str(Path(__file__).resolve().parents[1]), SPECFORGE_DEVICE='cpu', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1')
            def worker(name, step, previous=None):
                folder = p / name; folder.mkdir()
                env.update(DRAFTFIT_JOB_OUTPUT=str(folder), DRAFTFIT_JOB_RESULT=str(folder / 'result.json'), DRAFTFIT_JOB_ATTEMPT='1',
                           DRAFTFIT_INPUT_RESULTS=json.dumps({'first': str(p / 'first/result.json')} if previous else {}))
                cmd = [sys.executable, '-m', 'draftfit.campaign_joint', '--manifest', str(p / 'family.json'), '--stop-at', str(step)]
                if previous: cmd += ['--resume-job', 'first']
                result = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return json.loads((folder / 'result.json').read_text())
            first = worker('first', 1)
            resumed = worker('resumed', 2, first)
            full = worker('full', 2)
            a = torch.load(resumed['checkpoint'], weights_only=True)
            b = torch.load(full['checkpoint'], weights_only=True)
            self.assertEqual(a['step'], 2)
            self.assertEqual(a['fingerprint'], b['fingerprint'])
            for arm in a['drafts']:
                for key in a['drafts'][arm]:
                    self.assertTrue(torch.equal(a['drafts'][arm][key], b['drafts'][arm][key]), (arm, key))
            def equal(x, y):
                if isinstance(x, torch.Tensor): self.assertTrue(torch.equal(x, y))
                elif isinstance(x, dict):
                    self.assertEqual(x.keys(), y.keys())
                    for key in x: equal(x[key], y[key])
                elif isinstance(x, (list, tuple)):
                    self.assertEqual(len(x), len(y))
                    for left, right in zip(x, y): equal(left, right)
                else: self.assertEqual(x, y)
            equal(a['optimizer'], b['optimizer'])
            equal(a['rng'], b['rng'])
            self.assertEqual(len(resumed['arms']), 3)
            self.assertTrue(all(Path(x['export'], 'model.safetensors').exists() for x in resumed['arms']))



class AdapterTests(unittest.TestCase):
    def test_raw_mlx_aggregation_and_rejection(self):
        from draftfit.campaign_mlx import report_from_rows
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / 'train.json'; receipt.write_text('{}')
            rows = [dict(kind='provenance', manifest={'target': 'fixed', 'arms': [
                {'name': 'target', 'mode': 'baseline'}, {'name': 'dspark', 'mode': 'dspark', 'path': '/weights'}]},
                trials=3, selected_ids=['code:1'], max_new_tokens=4, data_sha256='data', runtime_commit='runtime')]
            for trial in range(3):
                for name, speed in [('target', 10), ('dspark', 20)]:
                    rows.append(dict(kind='measurement', arm=name, trial=trial, id='code:1', domain='code',
                                     parity=True, result={'token_ids': [1, 2]}, tok_s=speed))
            rows.append(dict(kind='summary', failed_arms=[]))
            report = report_from_rows(rows, receipt, 'fixed', ['dspark'])
            d = report['candidates'][0]['domains'][0]
            self.assertAlmostEqual(d['baseline_seconds'] / d['seconds'], 2)
            for mode in ('missing', 'duplicate', 'parity', 'tokens', 'trials'):
                broken = copy.deepcopy(rows)
                if mode == 'missing': del broken[1]
                if mode == 'duplicate': broken.insert(1, broken[1])
                if mode == 'parity': broken[2]['parity'] = False
                if mode == 'tokens': broken[2]['result']['token_ids'] = [3, 4]
                if mode == 'trials': broken[0]['trials'] = 1
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    report_from_rows(broken, receipt, 'fixed', ['dspark'])

    def test_actual_gate_chain_records_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            a = job('train')
            code = """import os,json,hashlib
from pathlib import Path
source=next(iter(json.loads(os.environ['DRAFTFIT_INPUT_RESULTS']).values()))
report={'status':'complete','schema':'draftfit-campaign-evaluation-v1','split':'validation',
'comparison_id':'fixed','comparison_fingerprint':'fixed','training_result_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
'candidates':[{'name':'dspark','exact_token_parity':True,'domains':[{'name':'code','requests':32,'matched_requests':32,
'output_tokens':100,'baseline_output_tokens':100,'seconds':1,'baseline_seconds':1.2}]}]}
Path(os.environ['DRAFTFIT_JOB_RESULT']).write_text(json.dumps(report))
"""
            e = dict(id='eval', kind='evaluate', resource='validation', argv=[sys.executable, '-c', code],
                     depends_on=['train'], train_job='train', comparison_id='fixed', timeout_seconds=5)
            gate = dict(evaluations=['eval'], top_k=1, min_speedup=1.1, min_domain_speedup=.9, min_relative_gain=.01)
            b = job('extend'); b.update(depends_on=['eval'], gate=gate)
            c = job('reject'); c.update(depends_on=['eval'], gate={**gate, 'min_speedup':1.3})
            path = p / 'config.json'; path.write_text(json.dumps(config([a, e, b, c])))
            self.assertEqual(run(path, p / 'out'), 0)
            state = json.loads((p / 'out/state.json').read_text())
            self.assertEqual(state['jobs']['extend']['status'], 'complete')
            self.assertEqual(state['jobs']['reject']['status'], 'skipped')
            self.assertEqual(state['jobs']['extend']['decision']['selected'], ['dspark'])
            (p / 'out/train/result.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'modified'):
                run(path, p / 'out')

    def test_sigterm_retry_is_explicit_and_bounded(self):
        import signal
        import time
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); path = p / 'c.json'
            a = job('long', 1.5); path.write_text(json.dumps(config([a])))
            cmd = [sys.executable, '-m', 'draftfit.campaign', 'run', '--config', str(path), '--output', str(p / 'out')]
            process = subprocess.Popen(cmd)
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    statepath = p / 'out/state.json'
                    if statepath.exists():
                        state = json.loads(statepath.read_text())
                        if state['jobs']['long'].get('pid'): break
                    time.sleep(.05)
                else: self.fail('child did not start')
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=6), 2)
                state = json.loads(statepath.read_text())
                self.assertEqual(state['jobs']['long']['status'], 'interrupted')
                with self.assertRaises(ProcessLookupError): os.kill(state['jobs']['long']['pid'], 0)
                self.assertEqual(run(path, p / 'out', retry_interrupted=True), 0)
                self.assertEqual(json.loads(statepath.read_text())['jobs']['long']['attempts'], 2)
            finally:
                if process.poll() is None: process.kill(); process.wait()


if __name__ == '__main__':
    unittest.main()
