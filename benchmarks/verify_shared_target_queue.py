"""Persistent real-model CPU/CUDA queue → verify → resume → export gate.

Tiny random Qwen weights exercise execution, not pretrained quality or production
serving speed. Run from the repository with --output-dir a fresh durable directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    root=Path(args.output_dir).resolve()
    if root.exists():
        raise ValueError('gate requires a fresh output directory')
    root.mkdir(parents=True)
    os.environ.update(SPECFORGE_DEVICE=args.device,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import torch
    from tests.test_joint_online_cpu import online_fixture
    from draftfit.export.to_hf import export_to_hf
    from draftfit.modeling.auto import AutoDraftModel
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(1)
    started=time.time()
    online_fixture(root,seeded=False)
    target_sha=sha(root/'target/model.safetensors')
    train=[dict(input_ids=list(range(1+i,9+i)),loss_mask=[0,1,1,1,1,1,1,0],domain='chat') for i in range(20)]
    holdout=[dict(input_ids=list(range(50+i,58+i)),loss_mask=[0,1,1,1,1,1,1,0],domain=d) for i,d in enumerate(('chat','code'))]
    for name,rows in [('train',train),('eval',holdout)]:
        (root/f'{name}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    base=json.loads((root/'base.json').read_text())
    base['training'].update(max_steps=10,total_steps=10,eval_interval=5,save_interval=5,
        joint_feature_queue_bytes=32768,joint_feature_queue_batches=4,
        joint_verification=dict(interval=5,max_prompts=2,max_new_tokens=8,draft_tokens=2,trials=1,offload_optimizer=True))
    (root/'base.json').write_text(json.dumps(base))
    manifest=json.loads((root/'family.json').read_text())
    env=os.environ.copy()
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT'): env.pop(key,None)
    def run(name,*extra):
        manifest['output_dir']=name
        path=root/f'{name}.json';path.write_text(json.dumps(manifest))
        command=[sys.executable,'-m','draftfit.cli','family','--manifest',str(path),'--joint',*extra]
        with (root/f'{name}.log').open('w') as log:
            result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {root/name}.log')
    run('full')
    run('partial','--stop-at-step','5')
    run('resumed','--resume-from',str(root/'partial/joint-step5.pt'))
    full=torch.load(root/'full/joint-step10.pt',map_location='cpu',weights_only=True)
    resumed=torch.load(root/'resumed/joint-step10.pt',map_location='cpu',weights_only=True)
    assert full['step']==resumed['step']==10
    assert full['consumed_samples']==resumed['consumed_samples']==20
    exports={}
    for arm,weights in full['drafts'].items():
        assert all(torch.equal(value,resumed['drafts'][arm][key]) for key,value in weights.items()),arm
        out=root/'exports'/arm
        export_to_hf(str(root/'full'/arm/f'{arm}-latest'),str(root/'full'/arm/'config.json'),
                     str(out),embedding_source=str(root/'target'))
        loaded=AutoDraftModel.from_pretrained(str(out))
        assert all(torch.equal(value,loaded.state_dict()[key]) for key,value in weights.items()),arm
        exports[arm]={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.is_file()}
    assert sha(root/'target/model.safetensors')==target_sha
    report=json.loads((root/'full/verification-step10.json').read_text())
    assert report['tokenparity'] and not report['production_engine_speed']
    assert report['artifact_sha256']==sha(root/'full/joint-step10.pt')
    assert report['domains']==['chat','code']
    assert len(report['candidates'])==3
    curves=[json.loads(line) for line in (root/'full/learning-curves.jsonl').read_text().splitlines()]
    closed=[row for row in curves if row['kind']=='queue_closed'][-1]
    assert closed['resident_bytes']==closed['resident_items']==0
    assert closed['consumed']==20 and closed['peak_bytes']<=32768
    # Teacher training/eval capture count is independent of three consumers.
    capture=[row for row in curves if row['kind'] in ('capture','evaluation_capture')][-1]
    # 20 train microbatches plus 2 held-out batches at initial/step5/step10.
    # Reference decoder forwards are separate and intentionally not counted.
    assert capture['capture_calls']==26,capture
    assert not list((root/'full').rglob('*.ckpt'))
    receipt=dict(schema='draftfit-shared-queue-gate-v1',device=args.device,
        torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
        total_seconds=time.time()-started,global_step=10,consumed_samples=20,
        exact_resume_weights=True,teacher_sha_unchanged=True,
        queue=closed,capture_calls=capture['capture_calls'],exports=exports,
        verification=report)
    (root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('device','global_step','consumed_samples','exact_resume_weights','total_seconds')}))


if __name__=='__main__':
    main()
