"""Real tiny HTTP/process tests, explicitly MOCK engine/hardware; no GPU evidence."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from draftfit.training import sglang_producer as producer
from draftfit.training import serving_dispatch as dispatch
from draftfit.training.joint_policy import fingerprint
from draftfit.training.serving_report import normalize_serving_report
from tests.test_joint_policy import artifact

FAKE_SERVER = '''
import argparse,json
from http.server import BaseHTTPRequestHandler,HTTPServer
p=argparse.ArgumentParser();p.add_argument('--port',type=int);p.add_argument('--speculative-algorithm');a,_=p.parse_known_args()
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):self.send_response(200);self.end_headers();self.wfile.write(b'OK')
 def do_POST(self):
  r=json.loads(self.rfile.read(int(self.headers['Content-Length'])));n=r['sampling_params']['max_new_tokens']
  m={'completion_tokens':n,'output_token_logprobs':[[0,i,None] for i in range(n)]}
  if a.speculative_algorithm:m.update(spec_verify_ct=n-1,spec_num_proposed_drafts=(n-1)*3,spec_num_correct_drafts=0)
  self.send_response(200);self.end_headers();self.wfile.write(json.dumps({'meta_info':m}).encode())
 def log_message(self,*args):pass
HTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
'''


class SGLangProducerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.deadline=time.time()+30
        target=self.root/'target';target.mkdir();(target/'config.json').write_text('{}')
        (target/'model.safetensors').write_bytes(b'MOCK weights')
        self.holdout=self.root/'holdout.jsonl'
        self.rows=[dict(row_index=i,domain=d,input_ids=[1,2,i+3]) for i,d in enumerate(('chat','code','math'))]
        self.holdout.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        cp=self.root/'joint.pt';cp.write_bytes(b'MOCK joint checkpoint')
        sources={'checkpoint':dispatch._source(cp,self.deadline),
                 'source:target':dispatch._source(target,self.deadline),
                 'source:validation':dispatch._source(self.holdout,self.deadline)}
        artifacts={}
        for name,family in [('dspark','dspark'),('ce','dflash'),('selector','dflash2')]:
            path=self.root/name;path.mkdir()
            (path/'config.json').write_text(json.dumps(dict(architectures=[producer._ARCH[family]],
                                                          model_type='qwen3',block_size=4)))
            (path/'model.safetensors').write_bytes(b'MOCK '+name.encode())
            sources['export:'+name]=dispatch._source(path,self.deadline)
            artifacts[name]=artifact(4,family=family,artifact_sha256=sources['checkpoint']['sha256'],
                checkpoint_sha256=sources['checkpoint']['sha256'],export_sha256=fingerprint(sources['export:'+name]['files']))
        self.runtime=dict(engine='sglang',version='0.5.19',build_sha256='a'*64,device='MOCK CUDA',dtype='bfloat16')
        expected=producer.build_expectations(artifacts=artifacts,target_model=target,target_revision='local-immutable',
            validation_path=self.holdout,runtime=self.runtime,config={'max_new_tokens':32},min_requests=3,deadline_unix=self.deadline)
        self.request=dict(expected=expected,sources=sources,native_report=dict(schema='draftfit-joint-generation-v1',
            tokenparity=True,production_engine_speed=False,artifact_sha256=sources['checkpoint']['sha256'],
            candidates=[dict(name=n,tokenparity=True) for n in artifacts]))

    def test_actual_http_server_process_all_three_families_strict_report(self):
        prepared=producer.preflight(self.request,self.deadline,inspect=lambda d:self.runtime)
        fake=self.root/'mock-engine';fake.mkdir();package=fake/'sglang';package.mkdir()
        (package/'__init__.py').write_text('');(package/'launch_server.py').write_text(FAKE_SERVER)
        env={'PYTHONPATH':str(fake)+os.pathsep+os.environ.get('PYTHONPATH','')}
        results=[]
        with mock.patch.dict(os.environ,env):
            for arm in [None,*prepared['arms']]:results.append(producer.serve(self.root,prepared,arm,self.deadline))
        report=producer.assemble(self.request,results)
        output=normalize_serving_report(report,**self.request['expected'])
        self.assertEqual(len(output['candidates']),3)
        self.assertFalse(output['speed_decisions_allowed'])
        self.assertFalse(report['target_resident'])
        for arm in report['candidates']:
            self.assertEqual(len(arm['runs']),3)
            self.assertTrue(all(r['generated_tokens']==32 and r['proposed_tokens']>0 for r in arm['runs']))
        report['candidates'][2]['runs'][0]['token_ids'][0]+=1
        with self.assertRaisesRegex(ValueError,'parity'):normalize_serving_report(report,**self.request['expected'])

    def test_preflight_rejects_actual_runtime_prompt_export_and_native_mismatch(self):
        with self.assertRaisesRegex(ValueError,'runtime/device'):
            producer.preflight(self.request,self.deadline,inspect=lambda d:{**self.runtime,'version':'other'})
        for part in ('prompt','artifact','native','config'):
            request=deepcopy(self.request)
            if part=='prompt':request['expected']['bindings']['prompts'][0]['prompt_sha256']='f'*64
            if part=='artifact':request['expected']['artifacts']['selector']['export_sha256']='f'*64
            if part=='native':request['native_report']['tokenparity']=False
            if part=='config':request['expected']['bindings']['config']['cache_policy']='reused'
            with self.subTest(part=part),self.assertRaises(ValueError):
                producer.preflight(request,self.deadline,inspect=lambda d:self.runtime)
        self.holdout.write_text(self.holdout.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'source bytes'):
            producer.preflight(self.request,self.deadline,inspect=lambda d:self.runtime)

    def test_measure_rejects_missing_or_invalid_real_verification_counters(self):
        config=producer.DEFAULT_CONFIG.copy();config['max_new_tokens']=32
        prompt=dict(row_index=0,domain='chat',input_ids=[1],prompt_sha256=fingerprint([1]))
        meta=dict(output_token_ids=list(range(32)),completion_tokens=32)
        for fields in ({},{'spec_verify_ct':0,'spec_num_correct_drafts':0,'spec_num_proposed_drafts':10},
                       {'spec_verify_ct':2,'spec_num_correct_drafts':33,'spec_num_proposed_drafts':40}):
            with mock.patch.object(producer,'_http',return_value={'meta_info':{**meta,**fields}}):
                with self.assertRaisesRegex(ValueError,'counters'):
                    producer.measure('http://127.0.0.1',prompt,0,config,self.deadline,speculative=True)

    def test_real_cli_deadline_fails_without_gpu_or_server(self):
        job=self.root/'job';job.mkdir()
        (job/'request.json').write_text(json.dumps(self.request))
        (job/'admission.json').write_text(json.dumps(dict(approved=True,deadline_unix=time.time()-1,max_seconds=10)))
        p=subprocess.run([sys.executable,'-B','-m','draftfit.training.sglang_producer','--job-dir',str(job)],
            capture_output=True,text=True,timeout=5)
        self.assertNotEqual(p.returncode,0)
        self.assertEqual(json.loads((job/'producer-receipt.json').read_text())['status'],'failed')
        self.assertFalse((job/'report.json').exists())
        self.assertFalse(list(job.glob('*server.log')))

    def test_fixed_holdout_rejects_duplicate_rows_and_noninteger_tokens(self):
        for row in (dict(self.rows[0],input_ids=[True]),self.rows[0]):
            self.holdout.write_text(''.join(json.dumps(r)+'\n' for r in [*self.rows,row]))
            with self.assertRaises(ValueError):producer.load_holdout(self.holdout,producer.DEFAULT_CONFIG)


if __name__=='__main__':unittest.main()
