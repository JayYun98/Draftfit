"""Paired MLX lookup vs DFlash2 benchmark; no training or downloads."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--trials', type=int, default=3)
    a = p.parse_args()
    if a.tokens < 2 or a.trials < 1:
        p.error('tokens >= 2 and trials >= 1 required')
    sys.path.insert(0, str(a.runtime / 'src'))
    import mlx.core as mx
    from mlx_dspark.generate import greedy_generate, dflash_generate, encode_messages
    from mlx_dspark.lookup import lookup_generate
    from mlx_dspark.load import load_target, load_dflash
    mx.set_cache_limit(128 * 1024 * 1024)
    cfg = json.loads(a.manifest.read_text())
    arm = next(x for x in cfg['arms'] if x['name'].startswith('dflash2'))
    target, tok = load_target(cfg['target'], require_tap=True)
    if cfg.get('target_float32'):
        target.model.set_dtype(mx.float32)
    draft, _ = load_dflash(arm['path'], bits=cfg.get('draft_bits', 4))
    arms = ['target', 'dflash2', 'lookup', 'hybrid']
    out = a.output.open('x')
    def emit(row):
        out.write(json.dumps(row) + '\n'); out.flush()
    emit(dict(kind='provenance', manifest=cfg, selected_draft=arm,
              runtime_commit=subprocess.check_output(['git','-C',str(a.runtime),'rev-parse','HEAD'],text=True).strip(),
              runtime_diff_sha256=hashlib.sha256(subprocess.check_output(['git','-C',str(a.runtime),'diff'])).hexdigest(),
              harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),
              device=mx.device_info(), mlx=mx.__version__, tokens=a.tokens, trials=a.trials,
              draft_cap=3, lookup_cap=6, lookup_long_cap=32, ngram_min=4, ngram_max=5,
              fresh_kv_every_request=True, synthetic_edit_cases=True, enable_thinking=False))
    def run(mode, ids, budget):
        rounds, prefill = [], []
        def pref(*_):
            mx.synchronize(); prefill.append(time.perf_counter())
        kwargs=dict(prompt_ids=ids, max_new_tokens=budget, temperature=0,
                    on_round=lambda **r: rounds.append(r), on_prefill=pref)
        mx.synchronize(); start=time.perf_counter()
        if mode == 'target':
            r=greedy_generate(target,tok,**kwargs)
        elif mode == 'lookup':
            r=lookup_generate(target,tok,max_draft_tokens=6,long_draft_tokens=32,
                              ngram_min=4,ngram_max=5,**kwargs)
        else:
            r=dflash_generate(target,tok,draft,max_draft_tokens=3,
                              lookup_drafts=mode=='hybrid',lookup_max_draft=6,
                              lookup_long_draft=32,**kwargs)
        mx.synchronize(); end=time.perf_counter()
        if len(r.token_ids)>budget:
            raise AssertionError('output exceeds budget')
        return dict(result=dataclasses.asdict(r),rounds=rounds,seconds=end-start,
                    prefill_callback_seconds=prefill[-1]-start,
                    decode_timing_valid=False, tok_s=len(r.token_ids)/(end-start))
    cases=[]; seen=set()
    for line in a.data.read_text().splitlines():
        row=json.loads(line); domain=row['id'].split(':')[0]
        if domain in seen: continue
        seen.add(domain)
        msgs=row.get('messages',row.get('conversations'))
        cases.append(dict(id=row['id'],group='general',domain=domain,messages=msgs[:-1]))
    for kind,code in [
        ('python', '\n'.join(f'def transform_{i}(value):\n    result = value * {i+2}\n    return result\n' for i in range(10))),
        ('javascript','\n'.join(f'function transform{i}(value) {{\n  const result = value * {i+2};\n  return result;\n}}' for i in range(10)))]:
        # Identical requested edit in cold and history-present contexts. No test output
        # is ever inserted into lookup; the prior document is known input in all arms.
        instruction='Return the entire code verbatim, changing only transform_9 to transform_final and transform9 to transformFinal. No explanation. Use a code fence.'
        for warm in (False,True):
            msgs=([dict(role='user',content='Show me the current code.'),
                   dict(role='assistant',content=f'```{kind}\n{code}\n```'),
                   dict(role='user',content=instruction)] if warm else
                  [dict(role='user',content=instruction+'\n```'+kind+'\n'+code+'\n```')])
            cases.append(dict(id=f'edit-{kind}-{warm}', group='edit-history' if warm else 'edit-cold',
                              domain=kind,messages=msgs))
    for c in cases:
        c['ids']=encode_messages(tok,c['messages'],enable_thinking=False)
    emit(dict(kind='cases',cases=cases))
    for mode in arms:
        run(mode,cases[0]['ids'],16)
    expected={}
    for c in cases:
        expected[c['id']]=run('target',c['ids'],a.tokens)['result']['token_ids']
    failures=0
    for trial in range(a.trials):
        for i,c in enumerate(cases):
            order=arms[(trial+i)%4:]+arms[:(trial+i)%4]
            for mode in order:
                r=run(mode,c['ids'],a.tokens)
                parity=r['result']['token_ids']==expected[c['id']]
                failures+=not parity
                emit(dict(kind='measurement',arm=mode,trial=trial,id=c['id'],group=c['group'],
                          domain=c['domain'],prompt_tokens=len(c['ids']),parity=parity,
                          peak_memory_bytes=mx.get_peak_memory(),**r))
                print(trial,c['id'],mode,round(r['decode_tok_s'],1),'parity',parity,flush=True)
    emit(dict(kind='complete',parity_failures=failures,measurements=a.trials*len(cases)*4))
    out.close()
    if failures: raise SystemExit(1)

if __name__ == '__main__': main()
