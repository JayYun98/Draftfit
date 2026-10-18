"""Run the existing MLX harness and turn raw matched requests into a gate report."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from draftfit.campaign import digest, number, read, write


def report_from_rows(rows, training_path, comparison_id, expected_arms):
    if not rows or rows[0].get('kind') != 'provenance' or rows[-1].get('kind') != 'summary' or rows[-1].get('failed_arms'):
        raise ValueError('incomplete/failed MLX benchmark')
    p = rows[0]
    arms = p['manifest']['arms']
    baselines = [a['name'] for a in arms if a['mode'] == 'baseline']
    if len(baselines) != 1 or {a['name'] for a in arms if a['mode'] != 'baseline'} != set(expected_arms):
        raise ValueError('benchmark arm set differs from trained artifacts')
    trials = p['trials']
    if type(trials) is not int or trials < 3:
        raise ValueError('at least three benchmark trials required')
    measurements = [r for r in rows if r['kind'] == 'measurement']
    expected = {(a['name'], i, t) for a in arms for i in p['selected_ids'] for t in range(trials)}
    keys = [(r['arm'], r['id'], r['trial']) for r in measurements]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError('missing/duplicate benchmark requests')
    baseline = {(r['id'], r['trial']): r for r in measurements if r['arm'] == baselines[0]}
    groups = {}
    for row in measurements:
        ref = baseline[(row['id'], row['trial'])]
        tokens = row['result']['token_ids']
        if row.get('parity') is not True or tokens != ref['result']['token_ids'] or row['domain'] != ref['domain'] or not 0 < len(tokens) <= p['max_new_tokens']:
            raise ValueError('MLX token parity/domain/output budget failed')
        number(row['tok_s'], positive=True)
        number(ref['tok_s'], positive=True)
        if row['arm'] == baselines[0]:
            continue
        group = groups.setdefault(row['arm'], {}).setdefault(row['domain'], dict(
            name=row['domain'], requests=0, matched_requests=0, output_tokens=0,
            baseline_output_tokens=0, seconds=0.0, baseline_seconds=0.0))
        group['requests'] += 1
        group['matched_requests'] += 1
        group['output_tokens'] += len(tokens)
        group['baseline_output_tokens'] += len(tokens)
        group['seconds'] += len(tokens) / row['tok_s']
        group['baseline_seconds'] += len(tokens) / ref['tok_s']
    # Exclude changing draft paths; retain workload/runtime/hardware/settings.
    identity = {k: v for k, v in p.items() if k not in {'manifest', 'kind'}}
    identity['manifest'] = {k: v for k, v in p['manifest'].items() if k not in {'arms', 'joint_monitor'}}
    identity['arms'] = [{k: v for k, v in a.items() if k != 'path'} for a in arms]
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return dict(status='complete', schema='draftfit-campaign-evaluation-v1', split='validation',
                comparison_id=comparison_id, comparison_fingerprint=fingerprint,
                training_result_sha256=digest(training_path),
                candidates=[dict(name=n, exact_token_parity=True, domains=list(d.values())) for n, d in groups.items()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    args = parser.parse_args()
    profile = read(args.profile)
    inputs = json.loads(os.environ['DRAFTFIT_INPUT_RESULTS'])
    if len(inputs) != 1:
        raise ValueError('MLX evaluation requires exactly one training dependency')
    training_path = next(iter(inputs.values()))
    receipt = read(training_path)
    folder = Path(os.environ['DRAFTFIT_JOB_OUTPUT'])
    manifest = dict(profile['manifest'])
    manifest['arms'] = [{'name': 'target', 'mode': 'baseline'}]
    for arm in receipt['arms']:
        if digest(Path(arm['export']) / 'model.safetensors') != arm['weights_sha256']:
            raise ValueError('export weights changed before benchmark')
        manifest['arms'].append(dict(name=arm['name'], mode=arm['strategy'], path=arm['export'], cap=profile.get('cap', 3)))
    write(folder / 'mlx-manifest.json', manifest)
    raw = folder / ('benchmark-' + os.environ.get('DRAFTFIT_JOB_ATTEMPT', '1') + '.jsonl')
    command = [profile.get('python', sys.executable), str(Path(profile['benchmark']).resolve()),
               '--runtime', profile['runtime'], '--manifest', str(folder / 'mlx-manifest.json'),
               '--data', profile['data'], '--output', str(raw),
               '--per-domain', str(profile.get('per_domain', 16)), '--trials', str(profile.get('trials', 3)),
               '--max-tokens', str(profile.get('max_tokens', 128))]
    subprocess.run(command, check=True)
    rows = [json.loads(l) for l in raw.read_text().splitlines() if l.strip()]
    report = report_from_rows(rows, training_path, profile['comparison_id'], [a['name'] for a in receipt['arms']])
    report['raw_benchmark_sha256'] = digest(raw)
    write(os.environ['DRAFTFIT_JOB_RESULT'], report)


if __name__ == '__main__':
    main()
