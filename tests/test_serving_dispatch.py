"""Actual CPU subprocess boundary tests; synthetic bindings, no GPU claims."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import time
import unittest

from draftfit.training import serving_dispatch as dispatch
from draftfit.training.joint_policy import fingerprint
from tests.test_joint_policy import policy, preset
from tests.test_serving_report import fixture


class ServingDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw, self.expected = fixture()
        self.checkpoint = self.root/'joint.pt'; self.checkpoint.write_bytes(b'explicit CPU test checkpoint')
        checkpoint_sha = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.exports = {}
        for name, artifact in self.expected['artifacts'].items():
            path = self.root/name; path.mkdir()
            (path/'config.json').write_text('{}')
            (path/'weights.bin').write_bytes(b'synthetic test export')
            files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}
            artifact.update(artifact_sha256=checkpoint_sha, checkpoint_sha256=checkpoint_sha,
                            export_sha256=fingerprint(files))
            self.exports[name] = path
        for arm in self.raw['candidates']:
            arm['artifact'] = deepcopy(self.expected['artifacts'][arm['name']])
        self.raw_path = self.root/'raw.json'; self.raw_path.write_text(json.dumps(self.raw))
        self.native = dict(schema='draftfit-joint-generation-v1', production_engine_speed=False,
            artifact_sha256=checkpoint_sha, tokenparity=True,
            candidates=[dict(name=name, tokenparity=True) for name in self.exports], existing_drafts=[])
        self.events = []
        self.job = self.root/'job'
        self.admission = dict(approved=True, reservation_id='external-1',
                              deadline_unix=time.time()+30, max_seconds=10)
        self.command = [sys.executable, '-c',
            'import pathlib,sys;d=pathlib.Path(sys.argv[-1]);'
            f'(d/"report.json").write_bytes(pathlib.Path({str(self.raw_path)!r}).read_bytes())']

    @contextmanager
    def memory(self):
        rng = random.getstate()
        self.events.append('release')
        random.random()
        try:
            yield dict(mode='released_training_memory', target_resident=False, evidence='CPU fake handoff')
        finally:
            random.setstate(rng)
            self.events.append('restore')

    def run_job(self, **kwargs):
        args = dict(command=self.command, expected=self.expected, checkpoint=self.checkpoint,
                    exports=self.exports, native_report=self.native, memory_handoff=self.memory,
                    admission=self.admission)
        args.update(kwargs)
        return dispatch.dispatch(self.job, **args)

    def policies(self):
        return {name: policy(domains=self.expected['domains'], min_requests=3)
                for name in self.exports}

    def test_real_crash_during_attempt_initialization_is_retryable(self):
        import subprocess
        payload=dict(job_dir=str(self.job),command=self.command,expected=self.expected,
            checkpoint=str(self.checkpoint),exports={k:str(v) for k,v in self.exports.items()},
            native_report=self.native,admission=self.admission)
        request=self.root/'crash.json';request.write_text(json.dumps(payload))
        script="""import json,os,sys
from pathlib import Path
from contextlib import nullcontext
from draftfit.training import serving_dispatch as d
write=d._write
def crash(path,value):
 if Path(path).name=='admission.json':os._exit(23)
 return write(path,value)
d._write=crash
d.dispatch(memory_handoff=lambda:nullcontext(dict(target_resident=False,mode='trainer_process_exited')),**json.loads(Path(sys.argv[1]).read_text()))
"""
        result=subprocess.run([sys.executable,'-c',script,str(request)],capture_output=True,timeout=10)
        self.assertEqual(result.returncode,23,result.stderr)
        self.admission['reservation_id']='fresh-after-crash'
        self.assertEqual(len(self.run_job()['candidates']),len(self.exports))
        with self.assertRaisesRegex(ValueError,'reservation already'):self.run_job()

    def test_actual_worker_strict_validation_restore_and_replay_policy(self):
        before = random.getstate()
        result = self.run_job()
        self.assertEqual(random.getstate(), before)
        self.assertEqual(self.events, ['release','restore'])
        receipt = json.loads((self.job/'receipt.json').read_text())
        self.assertEqual(receipt['status'], 'verified')
        self.assertFalse(receipt['target_resident'])
        again = dispatch.read_result(self.job, expected=self.expected, deadline_unix=time.time()+5)
        self.assertEqual(again, result)
        inputs = dict(policies=self.policies(), losses={name:1. for name in self.exports},
                      histories={}, budget={})
        original = deepcopy(inputs)
        first = dispatch.policy_update(result, **inputs)
        self.assertEqual(inputs, original)
        replay = dispatch.policy_update(again, **{**inputs, 'histories':first['history']})
        self.assertEqual(first, replay)
        for name, decision in first['decisions'].items():
            self.assertTrue(decision['current_serving_validated'])
            self.assertNotIn('speed', decision['best'])
            self.assertEqual(len(first['history'][name]), 1)
            self.assertFalse(decision['executed'])
        with self.assertRaisesRegex(ValueError, 'reservation already'):
            self.run_job()
        self.assertEqual(self.events, ['release','restore'])

    def test_no_handoff_no_admission_or_native_failure_starts_nothing(self):
        failures = [dict(memory_handoff=None), dict(admission={**self.admission, 'approved':False}),
                    dict(admission={**self.admission, 'deadline_unix':time.time()-1}),
                    dict(native_report={**self.native, 'tokenparity':False}),
                    dict(native_report={**self.native, 'existing_drafts':[dict(tokenparity=False)]})]
        for change in failures:
            with self.subTest(change=change), self.assertRaises((ValueError, TimeoutError)):
                self.run_job(**change)
        self.assertFalse(self.job.exists())
        self.assertEqual(self.events, [])

    def test_fail_timeout_and_fresh_admission_retry(self):
        with self.assertRaisesRegex(RuntimeError, 'exited 9'):
            self.run_job(command=[sys.executable, '-c', 'raise SystemExit(9)'])
        self.assertEqual(self.events, ['release','restore'])
        self.assertEqual(json.loads((self.job/'receipt.json').read_text())['status'], 'failed')
        self.admission.update(reservation_id='external-2', max_seconds=2.3)
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            self.run_job(command=[sys.executable, '-c', 'import time;time.sleep(30)'])
        self.assertLess(time.monotonic()-start, 3)
        self.assertEqual(self.events, ['release','restore']*2)
        self.admission.update(reservation_id='external-3', max_seconds=10)
        result = self.run_job()
        self.assertEqual(len(result['candidates']), 2)
        self.assertEqual(len(list(self.job.glob('attempt-*'))), 3)

    def test_invalid_report_source_mutation_and_restore_failure_not_committed(self):
        self.raw['candidates'][0]['runs'][0]['token_ids'][0] += 1
        self.raw_path.write_text(json.dumps(self.raw))
        with self.assertRaisesRegex(ValueError, 'parity'):
            self.run_job()
        self.assertEqual(self.events, ['release','restore'])
        with self.assertRaisesRegex(ValueError, 'no matching'):
            dispatch.read_result(self.job, expected=self.expected, deadline_unix=time.time()+5)
        self.raw['candidates'][0]['runs'][0]['token_ids'][0] -= 1
        self.raw_path.write_text(json.dumps(self.raw))
        self.admission['reservation_id'] = 'external-2'
        @contextmanager
        def failed_restore():
            yield dict(mode='released_training_memory', target_resident=False)
            raise RuntimeError('restore failed')
        with self.assertRaisesRegex(RuntimeError, 'restore failed'):
            self.run_job(memory_handoff=failed_restore)
        self.assertEqual(json.loads((self.job/'receipt.json').read_text())['status'], 'failed')
        self.admission['reservation_id'] = 'external-3'
        command = [*self.command]
        command[-1] += f';pathlib.Path({str(self.exports["ce"]/"extra.json")!r}).write_text("changed")'
        with self.assertRaisesRegex(ValueError, 'source bytes'):
            self.run_job(command=command)
        self.assertEqual(json.loads((self.job/'receipt.json').read_text())['status'], 'failed')

    def test_full_export_hash_stale_expectations_and_corrupt_receipt(self):
        (self.exports['ce']/'config.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):self.run_job()
        self.assertFalse(self.job.exists())
        (self.exports['ce']/'config.json').write_text('{}')
        self.run_job()
        changed = deepcopy(self.expected);changed['bindings']['runtime']['version'] = 'different'
        with self.assertRaisesRegex(ValueError, 'no matching'):
            dispatch.read_result(self.job, expected=changed, deadline_unix=time.time()+5)
        report = self.job/'attempt-000000/report.json'; report.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'report changed'):
            dispatch.read_result(self.job, expected=self.expected, deadline_unix=time.time()+5)

    def test_policy_no_native_relabel_no_mutation_and_budget_stop(self):
        result = self.run_job()
        kwargs = dict(policies=self.policies(), losses={name:1. for name in self.exports}, histories={}, budget={})
        first = dispatch.policy_update(result, **kwargs)
        history = deepcopy(first['history']); history['ce'][0]['serving']['production_engine_speed'] = False
        with self.assertRaisesRegex(ValueError, 'native history'):
            dispatch.policy_update(result, **{**kwargs, 'histories':history})
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            dispatch.policy_update(result, **{**kwargs, 'histories':first['history'],
                                             'losses':{name:2. for name in self.exports}})
        stopped = dispatch.policy_update(result, **{**kwargs, 'budget':{'spent_usd':1.}})
        self.assertTrue(all(d['action']=='stop' for d in stopped['decisions'].values()))
        self.assertTrue(all(d['next_steps']==0 for d in stopped['decisions'].values()))

    def test_registered_plateau_branch_requires_new_weights_only_phase(self):
        from draftfit.training.serving_report import normalize_serving_report
        result = self.run_job()
        policies = self.policies()
        for name, cfg in policies.items():
            artifact = self.expected['artifacts'][name]
            cfg.update(patience=1, presets=[preset('lower-lr', family=artifact['family'],
                capture_fingerprint=artifact['capture_fingerprint'], optimizer_mode='weights_only')])
        losses = {name:1. for name in self.exports}
        first = dispatch.policy_update(result, policies=policies, losses=losses, histories={}, budget={})
        raw, expected = deepcopy(self.raw), deepcopy(self.expected)
        for name, artifact in expected['artifacts'].items():
            artifact.update(step=11, artifact_sha256='f'*64, checkpoint_sha256='f'*64)
        for arm in raw['candidates']: arm['artifact'] = deepcopy(expected['artifacts'][arm['name']])
        second = normalize_serving_report(raw, **expected)
        update = dispatch.policy_update(second, policies=policies, losses=losses,
                                        histories=first['history'], budget={})
        for decision in update['decisions'].values():
            self.assertEqual(decision['action'], 'branch')
            self.assertEqual(decision['next_preset'], 'lower-lr')
            self.assertEqual(decision['optimizer_mode'], 'weights_only')
            self.assertTrue(decision['new_phase_required'])
            self.assertFalse(decision['optimizer_migration_allowed'])
        blocked = dispatch.policy_update(second, policies=policies, losses=losses,
            histories=first['history'], budget={}, tried_presets=['lower-lr'])
        self.assertTrue(all(d['action']=='stop' for d in blocked['decisions'].values()))
        self.assertTrue(all(len(h)==1 for h in first['history'].values()))

    def test_orphan_termination_and_explicit_resident_handoff_rejected(self):
        @contextmanager
        def resident():
            try: yield dict(mode='released_training_memory', target_resident=True)
            finally: self.events.append('restore')
        with self.assertRaisesRegex(ValueError, 'release training target'):
            self.run_job(memory_handoff=resident)
        self.assertEqual(self.events, ['restore'])
        self.assertFalse((self.job/'attempt-000000/report.json').exists())
        self.admission['reservation_id'] = 'external-2'
        ready, heartbeat = self.root/'ready', self.root/'heartbeat'
        child = ('import signal,time,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);'
                 f'p=pathlib.Path({str(heartbeat)!r});pathlib.Path({str(ready)!r}).touch();'
                 '\nwhile True:p.write_text(str(time.time_ns()));time.sleep(.02)')
        leader = self.command[-1] + (';import subprocess,time;'
            f'subprocess.Popen([sys.executable,"-c",{child!r}]);'
            f'\nwhile not pathlib.Path({str(ready)!r}).exists():time.sleep(.01)')
        self.run_job(command=[sys.executable, '-c', leader])
        time.sleep(.05); before = heartbeat.read_text();time.sleep(.1)
        self.assertEqual(before, heartbeat.read_text())

    def test_policy_old_draft_stays_separate_smoke_cannot_authorize_speed(self):
        old = deepcopy(self.raw['candidates'][0]);old['name'] = 'old-draft'
        old['artifact'].update(step=1, artifact_sha256='1'*64, export_sha256='2'*64)
        self.raw['existing_draft'] = old
        self.expected['existing_draft'] = deepcopy(old['artifact'])
        self.raw['speed_decisions_allowed'] = True
        self.raw_path.write_text(json.dumps(self.raw))
        result = self.run_job()
        output = dispatch.policy_update(result, policies=self.policies(),
            losses={name:1. for name in self.exports}, histories={}, budget={})
        self.assertFalse(result['speed_decisions_allowed'])
        self.assertEqual(output['existing_draft']['name'], 'old-draft')
        self.assertEqual(set(output['decisions']), set(self.exports))
        for decision in output['decisions'].values(): self.assertNotIn('speed', decision['best'])


if __name__ == '__main__': unittest.main()
