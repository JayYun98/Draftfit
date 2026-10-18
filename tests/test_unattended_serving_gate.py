"""Local CPU/control tests only; no SGLang/GPU performance claim."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

spec = importlib.util.spec_from_file_location('serving_gate', Path(__file__).resolve().parents[1] / 'benchmarks/verify_unattended_serving.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class GateTests(unittest.TestCase):
    def test_real_subprocess_success_failure_and_timeout(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder)/'worker.log'
            gate.run_bounded([sys.executable, '-c', 'print("ok")'], log, time.time()+10, 2)
            self.assertEqual(log.read_text().strip(), 'ok')
            with self.assertRaisesRegex(RuntimeError, 'exited 7'):
                gate.run_bounded([sys.executable, '-c', 'raise SystemExit(7)'], log, time.time()+10, 2)
            start = time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                gate.run_bounded([sys.executable, '-c', 'import time;time.sleep(30)'], log, time.time()+10, .3)
            self.assertLess(time.monotonic()-start, 3)

    def test_orphan_descendant_killed_even_after_leader_exits(self):
        with tempfile.TemporaryDirectory() as folder:
            heartbeat = Path(folder)/'heartbeat'
            ready = Path(folder)/'ready'
            child = ('import signal,time,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);'
                     f'p=pathlib.Path({str(heartbeat)!r});pathlib.Path({str(ready)!r}).touch();'
                     '\nwhile True:p.write_text(str(time.time_ns()));time.sleep(.02)')
            parent = ('import subprocess,sys,time,pathlib;'
                      f'p=subprocess.Popen([sys.executable,"-c",{child!r}]);'
                      f'\nwhile not pathlib.Path({str(ready)!r}).exists():time.sleep(.01)')
            gate.run_bounded([sys.executable, '-c', parent], Path(folder)/'log', time.time()+10, 3)
            time.sleep(.1)
            before = heartbeat.read_text()
            time.sleep(.15)
            self.assertEqual(before, heartbeat.read_text())

    def test_real_http_health_sample_and_verification_rejection(self):
        response = dict(meta_info=dict(output_token_logprobs=[[0, i, None] for i in range(32)],
            completion_tokens=32, spec_verify_ct=31, spec_num_proposed_drafts=93, spec_num_correct_drafts=0))
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200); self.end_headers(); self.wfile.write(b'OK')
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(response).encode())
            def log_message(self, *args): pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}'
            gate.healthy(url, 1)
            prompt = dict(id='math-0', domain='math', input_ids=[1,2], input_sha256=gate.digest([1,2]))
            job = dict(max_new_tokens=32, seed=481, request_timeout_seconds=1, deadline_unix=time.time()+10)
            baseline = gate.sample(url, prompt, job, False)
            candidate = gate.sample(url, prompt, job, True)
            self.assertTrue(gate.compare(dict(samples=[baseline]), dict(samples=[candidate]))[0]['exact_token_parity'])
            self.assertEqual(requests[0], requests[1])
            self.assertTrue(requests[0]['sampling_params']['ignore_eos'])
            response['meta_info']['spec_verify_ct'] = 0
            with self.assertRaisesRegex(ValueError, 'verification'): gate.sample(url, prompt, job, True)
            response['meta_info'].pop('output_token_logprobs')
            with self.assertRaisesRegex(ValueError, 'missing output'): gate.sample(url, prompt, job, False)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)

    def test_pair_mismatch_duplicate_and_changed_tokens(self):
        row = dict(id='a', domain='chat', request_sha256='r', input_sha256='i', output_sha256='o', output_ids=[1])
        baseline = dict(samples=[row])
        wrong = dict(row, output_ids=[2], output_sha256='o2')
        self.assertFalse(gate.compare(baseline, dict(samples=[wrong]))[0]['exact_token_parity'])
        for rows in ([row,row], [], [dict(row,request_sha256='other')]):
            with self.assertRaises(ValueError):gate.compare(baseline, dict(samples=rows))

    def test_expired_actual_cli_creates_failed_receipt_without_workers(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'out'
            process = subprocess.run([sys.executable, '-B', str(Path(gate.__file__)),
                '--controller-state', str(Path(folder)/'nonexistent'), '--output-dir', str(out),
                '--deadline-unix', str(time.time()-1)], capture_output=True, text=True, timeout=5)
            self.assertEqual(process.returncode, 1, process.stderr)
            report = gate.read(out/'receipt.json')
            self.assertFalse(report['all_passed'])
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['stages']['target']['status'], 'not_run')
            self.assertFalse(list(out.glob('*worker.log')))

    def test_orchestration_continues_only_completed_token_mismatch(self):
        scenarios = ('parity', 'success', 'runtime', 'source', 'incomplete', 'pairing')
        for scenario in scenarios:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as folder:
                out = Path(folder)/'out'
                calls = []
                row = dict(id='chat-0', domain='chat', request_sha256='request',
                           input_sha256='input', output_sha256='base', output_ids=[1,2])
                def execute(command, *args):
                    stage = command[command.index('--_worker')+1]
                    calls.append(stage)
                    if scenario == 'runtime' and stage == 'dspark':
                        raise RuntimeError('worker exited 7')
                    if scenario == 'source' and stage == 'verify-sources':
                        raise ValueError('target changed during gate')
                    if stage == 'target' or stage in gate.FAMILIES:
                        sample = dict(row)
                        status = 'complete'
                        if stage == 'dspark' and scenario != 'success':
                            sample.update(output_ids=[1,3], output_sha256='different')
                            if scenario == 'incomplete': status = 'running'
                            if scenario == 'pairing': sample['request_sha256'] = 'wrong'
                        gate.write(out/f'{stage}-result.json',dict(status=status, samples=[sample]))
                    return .01
                with mock.patch.object(gate, 'run_bounded', side_effect=execute), mock.patch('builtins.print'):
                    code = gate.main(['--controller-state',str(Path(folder)/'state.json'),
                        '--output-dir',str(out),'--deadline-unix',str(time.time()+30)])
                receipt = gate.read(out/'receipt.json')
                self.assertEqual(code, 0 if scenario == 'success' else 1)
                self.assertEqual(receipt['all_passed'], scenario == 'success')
                if scenario in ('parity','success','source'):
                    self.assertEqual(calls,['prepare','target',*gate.FAMILIES,'verify-sources'])
                    self.assertEqual(receipt['stages']['dflash']['status'],'complete')
                    self.assertEqual(receipt['stages']['dflash2']['status'],'complete')
                else:
                    self.assertEqual(calls,['prepare','target','dspark'])
                    self.assertEqual(receipt['stages']['dflash']['status'],'not_run')
                if scenario == 'parity':
                    self.assertEqual(receipt['status'],'failed')
                    self.assertEqual(receipt['stages']['dspark']['status'],'failed')
                    self.assertFalse(receipt['stages']['dspark']['pairs'][0]['exact_token_parity'])
                    self.assertEqual(receipt['stages']['verify-sources']['status'],'complete')
                if scenario == 'source':
                    self.assertIn('target changed',receipt['error'])

    def test_actual_cpu_controller_exports_and_dflash2_conversion(self):
        import torch
        from safetensors.torch import load_file
        from tests.test_unattended_controller import UnattendedControllerTest
        from draftfit.unattended import run
        from draftfit.modeling.auto import AutoDraftModel
        fixture = UnattendedControllerTest(); fixture.setUp()
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from tokenizers.pre_tokenizers import Whitespace
            from transformers import PreTrainedTokenizerFast
            backend = Tokenizer(WordLevel({'[UNK]':0, 'user':1, 'assistant':2}, unk_token='[UNK]'))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]')
            tokenizer.chat_template = '{% for message in messages %}{{ message.role }} {{ message.content }} {% endfor %}{% if add_generation_prompt %}assistant{% endif %}'
            tokenizer.save_pretrained(fixture.root/'target')
            state = run(fixture.path)
            out = fixture.root/'serving';out.mkdir()
            job = dict(controller_state=str(fixture.root/'unattended/state.json'),
                target_model=str(fixture.root/'target'), output_dir=str(out), deadline_unix=time.time()+60,
                max_new_tokens=64, context_length=512)
            prepared = gate.prepare(job)
            self.assertEqual(set(prepared['arms']), set(gate.FAMILIES))
            self.assertEqual(len(prepared['prompts']), 6)
            self.assertEqual({p['domain'] for p in prepared['prompts']}, {'chat','code','math'})
            item = prepared['arms']['dflash2']
            model = AutoDraftModel.from_pretrained(item['source'], local_files_only=True, torch_dtype='auto')
            expected = model.state_dict_for_serving()
            actual = load_file(str(Path(item['serving'])/'model.safetensors'))
            self.assertEqual(set(expected), set(actual))
            for key in actual:
                self.assertEqual(expected[key].dtype, actual[key].dtype, key)
                self.assertTrue(torch.equal(expected[key], actual[key]), key)
            self.assertEqual(gate.read(Path(item['serving'])/'config.json')['architectures'], ['DFlash2DraftModel'])
            gate.worker(job, 'verify-sources')
            source = Path(item['source'])/'config.json'
            source.write_text(source.read_text()+'\n')
            with self.assertRaisesRegex(ValueError, 'changed'): gate.worker(job, 'verify-sources')
        finally: fixture.doCleanups()


if __name__ == '__main__': unittest.main(verbosity=2)
