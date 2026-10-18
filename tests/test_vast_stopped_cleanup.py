import os
import time
import subprocess
import shutil
import sys
import json
from unittest.mock import patch
import hashlib
import tempfile
import unittest
from pathlib import Path
from scripts.vast_stopped_cleanup import cleanup, verify, copy_verified, _host_endpoint, STDERR_LIMIT


class CleanupTests(unittest.TestCase):
    def run_case(self, recoveries, limit=4, confirm=True):
        events=[]; ticks=[0]; values=iter(recoveries)
        def recover():
            events.append('copy')
            value=next(values,False)
            if isinstance(value,Exception):raise value
            return value
        result=cleanup(stop=lambda:events.append('stop'),stopped=lambda:confirm,
            recover=recover,destroy=lambda reason:events.append(('delete',reason)),
            expired=lambda:ticks[0]>=limit,record=lambda *args:None,
            sleep=lambda _:ticks.__setitem__(0,ticks[0]+1))
        return result,events

    def test_order(self):
        result,events=self.run_case([True])
        self.assertEqual(events,['stop','copy',('delete','backup_verified')])
        self.assertEqual(result,'deleted_verified')

    def test_failed_backup_retries_before_delete(self):
        _,events=self.run_case([RuntimeError('network'),False,True])
        self.assertEqual(events.count('copy'),3)
        self.assertEqual(events[-1],('delete','backup_verified'))

    def test_unconfirmed_stop_never_copies(self):
        _,events=self.run_case([True],confirm=False)
        self.assertNotIn('copy',events)
        self.assertEqual(events[-1],('delete','absolute_limit_backup_incomplete'))

    def test_limit_is_only_failed_backup_delete(self):
        _,events=self.run_case([False],limit=2)
        self.assertEqual(events.count('copy'),2)
        self.assertEqual(events[-1],('delete','absolute_limit_backup_incomplete'))

    def rows(self, source, names):
        return [dict(path=name, size=(source/name).stat().st_size,
                     sha256=hashlib.sha256((source/name).read_bytes()).hexdigest())
                for name in names]

    def local_transport(self, source, *, slow=False, captured=None):
        # Keep the production rsync flags/stdin/process lifecycle; substitute only
        # the remote transport with a real local rsync source. No Vast request.
        popen = subprocess.Popen
        def launch(command, **kwargs):
            if captured is not None:
                captured.append(command[:])
            args = list(command)
            self.assertEqual(args[0], 'rsync')
            args[-2] = str(source) + '/'
            args.insert(1, '--no-whole-file')  # Match remote rsync delta mode.
            if slow:
                args.insert(1, '--bwlimit=128')
            return popen(args, **kwargs)
        return patch('scripts.vast_stopped_cleanup.subprocess.Popen', side_effect=launch)

    @unittest.skipUnless(shutil.which('rsync'), 'native rsync required')
    def test_explicit_identity_transfers_and_handles_spaces(self):
        import shlex
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source';target=root/'backup'
            source.mkdir();(source/'weights').write_bytes(b'good')
            identity=root/'ssh identity';identity.write_text('test fixture only')
            commands=[]
            with patch('scripts.vast_stopped_cleanup._host_endpoint',return_value=('host',22)), self.local_transport(source,captured=commands):
                self.assertTrue(copy_verified(123,'/source',target,self.rows(source,['weights']),time.time()+10,ssh_identity=identity))
            transport=shlex.split(commands[0][commands[0].index('-e')+1])
            self.assertEqual(transport[transport.index('-i')+1],str(identity.resolve()))
            self.assertIn('IdentitiesOnly=yes',transport)

    @unittest.skipUnless(shutil.which('rsync'), 'native rsync required')
    def test_native_exact_inventory_and_hash_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source'; target=root/'backup'
            (source/'model').mkdir(parents=True)
            (source/'model'/'weights').write_bytes(b'good')
            (source/'model'/'unselected').write_bytes(b'not requested')
            (source/'line\nbreak').write_bytes(b'newline name')
            rows=self.rows(source, ['model/weights', 'line\nbreak'])
            commands=[]
            with patch('scripts.vast_stopped_cleanup._host_endpoint', return_value=('host',22)), self.local_transport(source, captured=commands):
                self.assertTrue(copy_verified(123,'/workspace/exports',target,rows,time.time()+10))
                self.assertFalse((target/'model'/'unselected').exists())
                self.assertTrue(verify(target, rows))
                saved=(target/'model'/'weights').stat()
                (target/'model'/'weights').write_bytes(b'bad!')
                os.utime(target/'model'/'weights', ns=(saved.st_atime_ns,saved.st_mtime_ns))
                self.assertTrue(copy_verified(123,'/workspace/exports',target,rows,time.time()+10))
                self.assertEqual((target/'model'/'weights').read_bytes(), b'good')
                rows[0]['sha256']=hashlib.sha256(b'bad!').hexdigest()
                self.assertFalse(copy_verified(123,'/workspace/exports',target,rows,time.time()+10))
            self.assertIn('--partial', commands[0])
            self.assertIn('--files-from=-', commands[0])
            self.assertIn('vastai_kaalia@host::C.123//workspace/exports/', commands[0])
            self.assertEqual(commands[0], [
                'rsync', '-dt', '--relative', '--partial', '--checksum',
                '--from0', '--files-from=-', '-e',
                'ssh -p 22 -o BatchMode=yes -o StrictHostKeyChecking=accept-new '
                '-o ConnectTimeout=8 -o ServerAliveInterval=10 -o ServerAliveCountMax=2',
                '--', 'vastai_kaalia@host::C.123//workspace/exports/', str(target.resolve()) + '/'])

    @unittest.skipUnless(shutil.which('rsync'), 'native rsync required')
    def test_optional_compression_preserves_verified_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'
            source.mkdir(); (source / 'weights').write_bytes(b'redundant weights' * 4096)
            rows = self.rows(source, ['weights'])
            for compress in (False, True):
                with self.subTest(compress=compress):
                    target = root / str(compress); commands = []
                    with (patch('scripts.vast_stopped_cleanup._host_endpoint', return_value=('host', 22)),
                          self.local_transport(source, captured=commands)):
                        self.assertTrue(copy_verified(123, '/source', target, rows,
                            time.time() + 10, compress=compress))
                    self.assertEqual(commands[0].count('-z'), int(compress))
                    self.assertTrue(verify(target, rows))

    @unittest.skipUnless(shutil.which('rsync'), 'native rsync required')
    def test_explicit_owned_remote_config_is_shell_escaped(self):
        import shlex
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; target = root / 'backup'
            source.mkdir(); (source / 'weights').write_bytes(b'exact inventory')
            config = "/approved rsync/owner's configs/42_C.123_rsync.conf"
            commands = []
            with (patch('scripts.vast_stopped_cleanup._host_endpoint', return_value=('host', 22)),
                  self.local_transport(source, captured=commands)):
                self.assertTrue(copy_verified(123, '/source', target, self.rows(source, ['weights']),
                    time.time() + 10, rsync_remote_config=config))
            options = [arg for arg in commands[0] if arg.startswith('--rsync-path=')]
            self.assertEqual(len(options), 1)
            self.assertEqual(shlex.split(options[0].split('=', 1)[1]),
                             ['sudo', '-n', 'rsync', '--config=' + config])

    def test_remote_config_rejects_other_instance_and_unsafe_paths_before_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'weights').write_bytes(b'present')
            rows = self.rows(root, ['weights'])
            # Even an already verified local backup must not bypass validation.
            invalid = ['42_C.123_rsync.conf', '/configs/42_C.124_rsync.conf',
                       '/configs/C.123_rsync.conf', '/configs/42_C.123_rsync.conf/extra',
                       '/configs/../42_C.123_rsync.conf', '/configs/./42_C.123_rsync.conf',
                       '/configs//42_C.123_rsync.conf', '//configs/42_C.123_rsync.conf',
                       '/configs\n/42_C.123_rsync.conf', '/configs/42_C.123_rsync.conf\0']
            with (patch('scripts.vast_stopped_cleanup._host_endpoint') as endpoint,
                  patch('scripts.vast_stopped_cleanup.subprocess.Popen') as launch):
                for config in invalid:
                    with self.subTest(config=config), self.assertRaisesRegex(ValueError, 'config'):
                        copy_verified(123, '/source', root, rows, time.time() + 10,
                                      rsync_remote_config=config)
                endpoint.assert_not_called()
                launch.assert_not_called()

    @unittest.skipUnless(shutil.which('rsync'), 'native rsync required')
    def test_native_timeout_preserves_partial_then_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source'; target=root/'backup'
            source.mkdir(); (source/'weights').write_bytes(os.urandom(2*1024**2))
            rows=self.rows(source,['weights']); events=[]
            with patch('scripts.vast_stopped_cleanup._host_endpoint', return_value=('host',22)):
                with self.local_transport(source, slow=True):
                    self.assertFalse(copy_verified(123,'/source',target,rows,time.time()+15,
                        attempt_seconds=1, record=lambda state, details:events.append((state,details))))
                self.assertTrue((target/'weights').exists())
                self.assertGreater((target/'weights').stat().st_size, 0)
                self.assertLess((target/'weights').stat().st_size, rows[0]['size'])
                self.assertFalse(verify(target, rows))
                with self.local_transport(source):
                    self.assertTrue(copy_verified(123,'/source',target,rows,time.time()+10))
                self.assertTrue(verify(target, rows))
            self.assertIn('backup_limit',[e[0] for e in events])

    def test_sdk_host_endpoint_and_existing_auth(self):
        # Exercise the installed SDK routing and request construction, intercepting
        # HTTP before any network access. Never read real credentials.
        import requests
        seen=[]
        def send(session, request, **kwargs):
            seen.append((request,kwargs))
            response=requests.Response(); response.status_code=200
            response._content=b'{"success":true,"src_addr":"host.example","src_port":1234}'
            return response
        with patch.dict(os.environ, {'VAST_API_KEY':'test-only', 'VAST_URL':'https://console.vast.ai'}), patch('requests.Session.send',new=send):
            self.assertEqual(_host_endpoint(123,'/source/','/target',7),('host.example',1234))
        request, options=seen[0]
        self.assertTrue(request.url.endswith('/api/v0/commands/rsync/'))
        self.assertEqual(json.loads(request.body)['dst_id'],None)
        self.assertEqual(json.loads(request.body)['src_id'],'C.123')
        self.assertEqual(request.headers['Authorization'],'Bearer test-only')
        self.assertEqual(options['timeout'],7)

    def test_budget_and_deadline_during_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'weights').write_bytes(b'abcdefgh')
            rows=self.rows(root,['weights']); checks=[]
            def expired():
                checks.append(1)
                return len(checks)>=5
            with patch('scripts.vast_stopped_cleanup.CHUNK',2):
                with self.assertRaises(TimeoutError):
                    verify(root, rows, expired=expired)
                with patch('scripts.vast_stopped_cleanup.time.time',side_effect=range(100)):
                    with self.assertRaises(TimeoutError):
                        verify(root, rows, deadline=7)
            with patch('scripts.vast_stopped_cleanup._host_endpoint') as endpoint:
                self.assertFalse(copy_verified(123,'/source',root,rows,time.time()+10,expired=lambda:True))
                endpoint.assert_not_called()

    def test_budget_interrupts_transfer_and_bounds_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rows=[dict(path='weights',size=4,sha256=hashlib.sha256(b'good').hexdigest())]
            popen=subprocess.Popen; children=[]; events=[]
            def launch(command, **kwargs):
                child=popen([sys.executable,'-c',
                    "import sys,time;sys.stderr.write('x'*100000);sys.stderr.flush();time.sleep(30)"],**kwargs)
                children.append(child)
                return child
            began=time.monotonic()
            with patch('scripts.vast_stopped_cleanup._host_endpoint',return_value=('host',22)), patch('scripts.vast_stopped_cleanup.subprocess.Popen',side_effect=launch):
                self.assertFalse(copy_verified(123,'/source',root,rows,time.time()+20,
                    expired=lambda:time.monotonic()-began>1,
                    record=lambda state,details:events.append((state,details))))
            self.assertIsNotNone(children[0].poll())
            result=next(details for state,details in events if state=='rsync_result')
            self.assertEqual(len(result['stderr']),STDERR_LIMIT)
            self.assertLess(time.monotonic()-began,8)

    def test_unknown_module_is_preserved_and_never_authorizes_delete(self):
        # Native rsync normally exits nonzero; also cover the observed CLI-style
        # zero exit without data. Neither may count as a verified backup.
        for returncode in (5, 0):
            with self.subTest(returncode=returncode), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                rows=[dict(path='weights',size=4,sha256=hashlib.sha256(b'good').hexdigest())]
                popen=subprocess.Popen; events=[]; commands=[]; ticks=[0]; deletes=[]
                def launch(command, **kwargs):
                    commands.append(command)
                    self.assertEqual(command[0], 'rsync')
                    return popen([sys.executable,'-c',
                        "import sys;sys.stderr.write(\"@ERROR: Unknown module '123'\\n\");sys.exit(%d)" % returncode], **kwargs)
                def recover():
                    return copy_verified(123,'/source',root,rows,time.time()+10,
                        record=lambda state, details:events.append((state,details)))
                with patch('scripts.vast_stopped_cleanup._host_endpoint',return_value=('host',22)) as endpoint, patch('scripts.vast_stopped_cleanup.subprocess.Popen',side_effect=launch):
                    result=cleanup(stop=lambda:None,stopped=lambda:True,recover=recover,
                        destroy=deletes.append,expired=lambda:ticks[0]>=1,
                        record=lambda state,details:events.append((state,details)),
                        sleep=lambda _:ticks.__setitem__(0,ticks[0]+1))
                self.assertEqual(result, 'deleted_at_limit')
                self.assertEqual(deletes, ['absolute_limit_backup_incomplete'])
                self.assertNotIn('backup_verified', [state for state,_ in events])
                receipt=next(details for state,details in events if state=='rsync_result')
                self.assertEqual(receipt['returncode'],returncode)
                self.assertIn("@ERROR: Unknown module '123'", receipt['stderr'])
                self.assertEqual(len(commands),1)
                endpoint.assert_called_once()

    def test_unsafe_inventory_or_destination_never_contacts_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); target=root/'backup'; target.mkdir()
            (target/'link').symlink_to(root, target_is_directory=True)
            for name in ('../outside','/absolute','link/file','.', 'a\0b'):
                rows=[dict(path=name,size=4,sha256='a'*64)]
                with patch('scripts.vast_stopped_cleanup._host_endpoint') as endpoint:
                    with self.assertRaises(ValueError):
                        copy_verified(123,'/source',target,rows,time.time()+10)
                    endpoint.assert_not_called()

    def test_hash_verification_rejects_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'weights';p.write_bytes(b'good')
            rows=[dict(path='weights',size=4,sha256=hashlib.sha256(b'good').hexdigest())]
            self.assertTrue(verify(tmp,rows));p.write_bytes(b'zero')
            self.assertFalse(verify(tmp,rows));self.assertFalse(verify(tmp,[]))

if __name__=='__main__':unittest.main()
