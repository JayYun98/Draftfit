"""Durable controller for the native shared-target joint trainer.

Each phase keeps its target/optimizer resident. Only an explicitly registered
weights-only phase may change training settings. Publication snapshots are
immutable and durable; a failed publication can be retried without retraining.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

from draftfit.training.joint_policy import decide, fingerprint, _config, _validate

SCHEMA = 'draftfit-unattended-v1'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _path(base, value):
    return (base / value).expanduser().resolve()


def source_binding(stages, *, full_target=False):
    """Bind data and model bytes across weights-only phases, not just resumes."""
    result, digests = {}, {}
    for stage in stages:
        cfg = stage.config
        files = [Path(cfg.data.prompts_path), Path(cfg.data.eval_data_path)]
        if cfg.model.draft_checkpoint_path:
            seed = Path(cfg.model.draft_checkpoint_path)
            if seed.is_dir():
                files.extend(p for p in seed.rglob('*') if p.is_file())
            elif seed.is_file():
                files.append(seed)
            else:
                raise ValueError('continual seed must be an existing local checkpoint')
        if cfg.training.joint_verification is not None:
            for source in cfg.training.joint_verification.existing_drafts.values():
                files.extend(p for p in Path(source).rglob('*') if p.is_file())
        target = Path(cfg.model.target_model_path)
        if target.is_dir():
            files.extend(p for p in target.rglob('*') if p.is_file() and
                         (full_target or p.suffix in {'.json', '.safetensors', '.bin', '.py', '.model'}))
        elif target.is_file():
            files.append(target)
        elif not re.fullmatch(r'[a-fA-F0-9]{40}', cfg.model.target_revision or ''):
            raise ValueError('remote target needs an immutable 40-character commit revision')
        for p in sorted(set(files)):
            name = str(p.resolve())
            if name not in digests: digests[name] = sha(p)
        result[stage.name] = dict(files={str(p.resolve()): digests[str(p.resolve())] for p in sorted(set(files))},
            target=cfg.model.target_model_path, revision=cfg.model.target_revision,
            draft_config=stage.draft_config.to_dict())
    return result


def resource_config(config, stages):
    from draftfit.config.schema import apply_overrides
    from draftfit.training.resource_policy import _config as validate_resource_config
    settings = config['resources']
    if set(settings) != {'policy', 'pilot_seconds'}:
        raise ValueError('resources requires policy and per-preset pilot_seconds')
    expected = {'initial', *(p['id'] for p in config['presets'])}
    if set(settings['pilot_seconds']) != expected or 'presets' in settings['policy']:
        raise ValueError('resource presets must match registered controller presets')
    candidates = []
    for preset in [None, *config['presets']]:
        name = preset['id'] if preset else 'initial'
        training = (apply_overrides(stages[0].config, preset['overrides']) if preset else stages[0].config).training
        candidates.append(dict(id=name, preflight_passed=True,
            training={k: getattr(training, k) for k in ('batch_size', 'accumulation_steps', 'learning_rate')},
            pilot_steps=preset['pilot_steps'] if preset else training.max_steps,
            pilot_cost_usd=preset['pilot_cost_usd'] if preset else config['policy']['next_cost_usd'],
            pilot_seconds=settings['pilot_seconds'][name]))
    return validate_resource_config({**settings['policy'], 'presets': candidates})


def serving_config(config, stages):
    from draftfit.training.sglang_producer import DEFAULT_CONFIG, validate_config
    settings = config['serving']
    if not isinstance(settings, dict) or set(settings) - {'max_seconds', 'cost_usd', 'config'}:
        raise ValueError('serving supports max_seconds/cost_usd/config only')
    for name in ('max_seconds', 'cost_usd'):
        value = settings.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('serving needs positive bounded duration and explicit admitted cost')
    if settings['max_seconds'] <= 5:
        raise ValueError('serving needs time for bounded process cleanup')
    validate_config({**DEFAULT_CONFIG, **settings.get('config', {})})
    if config['selection'] != 'acceptance':
        raise ValueError('optional production serving supports acceptance selection only')
    for stage in stages:
        if stage.config.model.target_backend != 'transformers' or stage.config.training.strategy not in ('dspark','dflash','dflash2'):
            raise ValueError('serving handoff supports ordinary Torch HF DSpark/DFlash/DFlash2 only')
        if stage.config.training.joint_verification.existing_drafts:
            raise ValueError('controller production old-draft mapping is not implemented; keep native mode')


def load_plan(path):
    from draftfit.family import build_family_plan
    from draftfit.training.joint_family import validate_joint_family
    path = Path(path).resolve()
    config = read(path)
    allowed = {'schema', 'manifest', 'output_dir', 'max_steps', 'deadline_unix',
               'budget_path', 'policy', 'presets', 'selection', 'publication', 'resources', 'serving'}
    if set(config) - allowed or config.get('schema') != SCHEMA:
        raise ValueError('invalid unattended schema or unknown fields')
    for name in ('manifest', 'output_dir', 'budget_path'):
        config[name] = str(_path(path.parent, config[name]))
    if type(config['max_steps']) is not int or config['max_steps'] < 1:
        raise ValueError('max_steps must be positive')
    deadline = config['deadline_unix']
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError('deadline_unix must be finite and explicit')
    config.setdefault('selection', 'acceptance')
    if config['selection'] not in ('loss', 'acceptance'):
        raise ValueError('native reference selection supports loss or acceptance; not production speed')
    config.setdefault('presets', [])
    ids = set()
    for preset in config['presets']:
        if set(preset) != {'id', 'when', 'reason', 'overrides', 'pilot_steps', 'pilot_cost_usd'}:
            raise ValueError('preset requires id/when/reason/overrides/pilot_steps/pilot_cost_usd')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', preset['id']) or preset['id'] in ids or preset['id'] == 'initial':
            raise ValueError('invalid/duplicate preset ID')
        ids.add(preset['id'])
        if not isinstance(preset['overrides'], list) or not preset['overrides']:
            raise ValueError('preset needs explicit training overrides')
        # Geometry and dataset changes need a different explicit source contract.
        allowed_overrides = {'training.batch_size', 'training.accumulation_steps',
            'training.learning_rate', 'training.lr_scheduler', 'training.warmup_ratio'}
        if any(not isinstance(v, str) or v.split('=', 1)[0] not in allowed_overrides for v in preset['overrides']):
            raise ValueError('preset may only change registered batch/LR schedule fields')
    from draftfit.config.schema import apply_overrides
    from draftfit.training.joint_online import JointOnlineSource
    from draftfit.training.schedule import validate_fixed_accumulation_plan
    _, stages = build_family_plan(config['manifest'])
    validate_joint_family(stages)
    for stage in stages:
        t = stage.config.training
        if not t.joint_online_capture or t.joint_verification is None or t.joint_control is not None:
            raise ValueError('unattended requires online capture + verification and owns joint_control')
        if not t.joint_feature_queue_bytes or not getattr(t, 'joint_emit_arm_checkpoints', True):
            raise ValueError('unattended requires bounded feature queue and arm checkpoints')
        if not t.max_steps or t.max_steps > config['max_steps']:
            raise ValueError('declare a phase max_steps within the controller total')
        if t.joint_verification.interval > config['policy']['next_steps']:
            raise ValueError('verification interval exceeds policy admitted interval')
    # Pure token/data validation and scheduler capacity checks, no model loading.
    source = JointOnlineSource(stages[0], needs_dspark=any(s.config.training.strategy == 'dspark' for s in stages))
    try:
        _, count = source.loader(stages[0], epoch=0, shuffle=True, drop_last=True)
        verification = stages[0].config.training.joint_verification
        selected_rows = source._rows[True][:verification.max_prompts]
        domains = [row.get('domain', 'unlabeled') for row in selected_rows]
        minimum = _config(config['policy'])
        if (set(domains) != set(minimum['domains'])
                or len(selected_rows) * verification.trials < minimum['min_requests']
                or any(domains.count(d) * verification.trials < minimum['min_domain_requests'] for d in minimum['domains'])):
            raise ValueError('verification workload does not satisfy policy domains/request counts')
    finally:
        source.close_queues()
    for preset in [None, *config['presets']]:
        adjusted = []
        for stage in stages:
            cfg = apply_overrides(stage.config, preset['overrides']) if preset else stage.config
            if preset:
                cfg.training.max_steps = cfg.training.total_steps = preset['pilot_steps']
            t = cfg.training
            validate_fixed_accumulation_plan(num_samples=count, batch_size=t.batch_size,
                accumulation_steps=t.accumulation_steps, num_epochs=t.num_epochs, max_steps=t.max_steps)
            capacity = (count // t.batch_size * t.num_epochs) // t.accumulation_steps
            if t.max_steps > capacity:
                raise ValueError('phase exceeds available optimizer steps')
            adjusted.append(replace(stage, config=cfg))
        validate_joint_family(adjusted)
    pub = config.get('publication')
    if pub is not None:
        if set(pub) != {'repos', 'private'} or type(pub['private']) is not bool:
            raise ValueError('publication requires repos and explicit private boolean')
        if set(pub['repos']) != {s.name for s in stages} or len(set(pub['repos'].values())) != len(stages):
            raise ValueError('publication needs a distinct repository for every arm')
    # Validate policy shape before creating any GPU model or output.
    policy = _config({**config['policy'], 'comparison_fingerprint': 'preflight'})
    if policy['presets']:
        raise ValueError('register executable presets at the controller, not in policy')
    if policy['limits']['max_steps'] != config['max_steps']:
        raise ValueError('policy/controller max_steps must agree')
    validation_policy = deepcopy(policy)
    validation_policy['presets'] = [dict(p, family='preflight', training_fingerprint='preflight',
        objective_fingerprint='preflight', capture_fingerprint='preflight', preflight_passed=True)
        for p in config['presets']]
    _validate(validation_policy, read(config['budget_path']))
    if config.get('resources') is not None:
        resource_config(config, stages)
    if config.get('serving') is not None:
        serving_config(config, stages)
    manifest = read(config['manifest'])
    binding = {'config': config, 'manifest_sha': sha(config['manifest']),
               'base_sha': sha(_path(Path(config['manifest']).parent, manifest['base_config'])),
               'sources': source_binding(stages, full_target=config.get('serving') is not None)}
    return config, stages, fingerprint(binding)


class Controller:
    def __init__(self, config, stages, binding, *, resume=False, publication_command=None):
        self.config, self.stages = config, stages
        self.root = Path(config['output_dir'])
        if self.root.exists() and not resume:
            raise ValueError('output exists; use --resume')
        if resume and not (self.root / 'state.json').is_file():
            raise ValueError('no durable state to resume')
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / '.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume:
            self.state = read(self.root / 'state.json')
            if self.state['binding'] != binding:
                self.lock.close()
                raise ValueError('unattended config/manifest/base changed')
        else:
            self.state = dict(schema=SCHEMA, binding=binding, status='ready',
                total_steps=0, cost_floor=read(config['budget_path']).get('spent_usd', 0),
                phases=[], best={}, publication_jobs=[], history={}, tried=[],
                exploration_floor=0, external_step_floor=read(config['budget_path']).get('steps_used',0), sources=source_binding(stages, full_target=config.get('serving') is not None),
                resource_state=dict(transitions=0, last_transition_step=None), resource_history=[], resource_reports=[])
            self.save()
        from draftfit.export.publication_worker import PublicationWorker
        self.publisher = PublicationWorker(command=publication_command)
        self.active_publication = None
        self.sampler = None
        self.observation_committed = False

    def save(self):
        write(self.root / 'state.json', self.state)

    def budget(self):
        result = read(self.config['budget_path'])
        timestamp = result.get('timestamp_unix')
        if type(timestamp) not in (int, float) or not 0 <= time.time() - timestamp <= 120:
            raise ValueError('fresh external budget observation required (max age 120s)')
        result['spent_usd'] = max(result.get('spent_usd', 0), self.state['cost_floor'])
        if self.config.get('serving') is not None:
            # Actual serving admissions survive observation rollback/process failure.
            for path in (self.root/'serving').glob('*/admission-*.json'):
                result['spent_usd'] = max(result['spent_usd'], read(path)['cost_floor'])
        result['steps_used'] = max(result.get('steps_used', 0), self.state['external_step_floor'] + self.state['total_steps'])
        result['exploration_spent_usd'] = max(result.get('exploration_spent_usd', 0), self.state['exploration_floor'])
        _validate(_config({**self.config['policy'], 'comparison_fingerprint': 'admission'}), result)
        result['pilots_used'] = max(result.get('pilots_used', 0), len(self.state['tried']))
        return result

    def reserve(self, cost, *, persist=True):
        budget = self.budget()
        limits = self.config['policy']['limits']
        if (time.time() >= self.config['deadline_unix'] or
                cost > limits['max_usd'] - limits['reserve_usd'] - budget['spent_usd'] - budget.get('reserved_usd', 0)):
            return False
        # Conservative admission floor, not a claim about settled provider costs.
        self.state['cost_floor'] = budget['spent_usd'] + cost
        if persist:
            self.save()
        return True

    def publications(self, *, wait=False):
        while True:
            if self.active_publication is not None:
                status = (self.publisher.wait(self.config['deadline_unix']) if wait
                          else self.publisher.poll())
                if status['status'] == 'running':
                    return
                if status['status'] != 'succeeded':
                    raise RuntimeError('publication failed or timed out; durable job preserved')
                receipt = Path(self.active_publication) / 'receipt.json'
                if not receipt.exists() or read(receipt).get('status') != 'verified':
                    raise RuntimeError('publication worker exited without a verified receipt')
                self.active_publication = None
            pending = self.pending_publications()
            if not pending:
                return
            if time.time() >= self.config['deadline_unix']:
                raise TimeoutError('publication deadline reached; durable job preserved')
            self.active_publication = pending[0]
            self.publisher.start(self.active_publication)
            if not wait:
                return

    def pending_publications(self):
        return [p for p in self.state['publication_jobs']
                if not (Path(p) / 'receipt.json').exists()
                or read(Path(p) / 'receipt.json').get('status') != 'verified']

    def prune_artifacts(self):
        # Only controller-owned immutable folders; no training or source files.
        protected = {str(Path(row['export']).parent) for row in self.state['best'].values()}
        for phase in self.state['phases']:
            for key in ('source', 'latest_export'):
                receipt = phase.get(key)
                if receipt:
                    protected.update(str(Path(row['export']).parent) for row in receipt['arms'].values())
        if self.config.get('serving') is not None:
            # Durable production receipts must retain the exact files they bind.
            for path in (self.root/'serving').glob('*/job/request.json'):
                for key, source in read(path)['sources'].items():
                    if key.startswith('export:'):
                        protected.add(str(Path(source['path']).parent))
        artifacts = self.root / 'artifacts'
        if artifacts.exists():
            for folder in artifacts.iterdir():
                if folder.is_dir() and not folder.is_symlink() and str(folder) not in protected and (folder / 'receipt.json').is_file():
                    shutil.rmtree(folder)
        # Verified jobs keep their manifest, remote SHA receipt and resume link;
        # their duplicate payload bytes are no longer needed for retry.
        for job in self.state['publication_jobs']:
            path = Path(job)
            if (path / 'receipt.json').exists() and read(path / 'receipt.json').get('status') == 'verified':
                if (path / 'payload').is_dir(): shutil.rmtree(path / 'payload')

    def export(self, checkpoint, stages, directory):
        # Exporters construct temporary models; their random initialization must
        # never alter the next training update, including after a resumed skip.
        from draftfit.training.joint_online import _preserve_rng
        from draftfit.utils import get_local_device
        with _preserve_rng(get_local_device()):
            return self._export(checkpoint, stages, directory)

    def _export(self, checkpoint, stages, directory):
        import torch
        from draftfit.export.to_hf import export_to_hf
        directory = Path(directory)
        if (directory / 'receipt.json').exists():
            receipt = read(directory / 'receipt.json')
            if receipt['checkpoint_sha256'] != sha(checkpoint):
                raise ValueError('existing export belongs to another checkpoint')
            for row in receipt['arms'].values():
                self.verify_export(row)
            return receipt
        # Only a private unpublished temporary directory may be replaced.
        temporary = directory.with_name(directory.name + '.building')
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        arms = {}
        with tempfile.TemporaryDirectory(dir=self.root, prefix='.materialize-') as tmp:
            tmp = Path(tmp)
            for stage in stages:
                training = tmp / stage.name
                training.mkdir()
                weights = state['drafts'][stage.name]
                torch.save({'draft_state_dict': weights}, training / 'training_state.pt')
                stage.draft_config.to_json_file(training / 'config.json')
                output = temporary / stage.name
                export_to_hf(str(training), str(training / 'config.json'), str(output),
                             embedding_source=stage.config.model.target_model_path)
                arms[stage.name] = dict(export=str(directory / stage.name),
                    sha256=sha(output / 'model.safetensors'), strategy=stage.config.training.strategy,
                    files={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
        receipt = dict(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=sha(checkpoint), arms=arms)
        write(temporary / 'receipt.json', receipt)
        temporary.rename(directory)
        return receipt

    @staticmethod
    def verify_export(row):
        folder = Path(row['export'])
        files = {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
        if files != row['files']:
            raise ValueError('immutable export changed')

    def verify_sources(self):
        current = source_binding(self.stages, full_target=self.config.get('serving') is not None)
        if fingerprint(current) != fingerprint(self.state['sources']):
            raise ValueError('target/dataset/draft source changed before policy commit or reuse')

    def verify_retained(self):
        """Use controller-committed expectations, never expectations from a job."""
        for row in self.state['best'].values():
            self.verify_export(row)
        if self.config.get('serving') is None:
            return
        self.verify_sources()
        from draftfit.training.serving_dispatch import read_result
        phases = {p['id']: p for p in self.state['phases']}
        keys = {(b['phase'], b['checkpoint_sha256']) for b in self.state['best'].values()}
        keys.update((p['id'], p['checkpoint_sha256']) for p in phases.values() if p.get('production_report'))
        deadline = time.time() + 60  # bounded local reads after compute admission expiry
        for phase_id, checkpoint_sha in sorted(keys):
            phase = phases[phase_id]
            evidence = phase.get('production_evidence', {}).get(checkpoint_sha)
            if evidence is None:
                raise ValueError('missing authoritative production binding')
            expected = evidence['expected']
            if any(a['artifact_sha256'] != checkpoint_sha or a['checkpoint_sha256'] != checkpoint_sha
                   for a in expected['artifacts'].values()):
                raise ValueError('committed production artifact binding changed')
            for name, best in self.state['best'].items():
                if (best['phase'], best['checkpoint_sha256']) != (phase_id, checkpoint_sha):
                    continue
                artifact = expected['artifacts'][name]
                if (artifact['step'] != best['step'] or artifact['family'] != best['strategy']
                        or evidence['exports'][name]['files'] != best['files']
                        or evidence['exports'][name]['export'] != best['export']):
                    raise ValueError('selected best production binding changed')
            job = self.root/'serving'/(phase_id+'-'+checkpoint_sha)/'job'
            result = read_result(job, expected=expected, deadline_unix=deadline)
            if evidence['source_report_sha256'] != result['source_report_sha256']:
                raise ValueError('committed production report binding changed')
            latest = phase.get('production_report', {})
            if checkpoint_sha == phase.get('checkpoint_sha256') and (
                    latest.get('job') != str(job) or latest.get('source_report_sha256') != result['source_report_sha256']):
                raise ValueError('latest production report binding changed')

    def start_resources(self):
        if self.config.get('resources') is None:
            return
        from draftfit.training.resource_monitor import ResourceSampler
        from draftfit.utils import get_local_device
        if self.sampler is not None: self.sampler.stop()
        self.resource_started = time.monotonic()
        self.resource_step = self.state['total_steps']
        self.sampler = ResourceSampler(device=get_local_device()).start()

    def observe_resources(self, step, metrics, report):
        if self.sampler is None:
            return None
        from draftfit.training.resource_policy import decide as resource_decide
        phase = self.state['phases'][-1]
        self.sampler.stop()
        measurement = self.sampler.summarize(phase['id'], phase['preset'], step,
            max(0, step - self.resource_step) / max(1e-9, time.monotonic() - self.resource_started),
            sum(v['eval/avg_loss'] for v in metrics.values()) / len(metrics))
        measurement['throughput_scope'] = 'campaign_updates_per_wall_second_including_verification'
        self.state['resource_reports'].append(measurement)
        if measurement['observation'] is not None:
            self.state['resource_history'].append(measurement['observation'])
        settings = resource_config(self.config, self.stages)
        budget = self.budget(); limits = self.config['policy']['limits']
        eligible = [p['id'] for p in self.config['presets'] if p['id'] not in self.state['tried']
            and p['pilot_cost_usd'] <= min(limits['max_pilot_usd'], limits['max_exploration_usd'] - budget['exploration_spent_usd'])
            and p['pilot_steps'] <= limits['max_pilot_steps']]
        admission = dict(allowed=budget['pilots_used'] < limits['max_pilots'], phase_id=phase['id'], step=step,
            preset_ids=eligible, max_steps=max(0,self.config['max_steps'] - budget['steps_used']),
            max_cost_usd=max(0,limits['max_usd'] - limits['reserve_usd'] - budget['spent_usd'] - budget.get('reserved_usd',0)),
            max_seconds=max(0,self.config['deadline_unix'] - time.time()))
        rows = [r for r in self.state['resource_history'] if r['phase_id'] == phase['id']
                and 0 <= step - r['step'] <= settings['max_lag_steps']
                and r['step'] > (self.state['resource_state']['last_transition_step'] or 0)][-settings['window_size']:]
        decision = resource_decide(settings, rows,
            current=dict(phase_id=phase['id'],preset=phase['preset'],step=step),
            admission=admission,state=self.state['resource_state'])
        phase['resource_decision'] = decision
        if measurement['observation'] is None or decision['action'] != 'new_phase':
            return None
        return dict(action='branch', reason=decision['reason'], next_preset=decision['next_preset'],
            source_artifact=dict(artifact_sha256=report['artifact_sha256']),
            resource_transition=decision['transition_state'])

    def observe_serving(self, checkpoint, report, metrics, stages, memory_handoff,
                        exported, artifacts, policies):
        import sys
        from draftfit.training import sglang_producer as producer
        from draftfit.training import serving_dispatch as dispatcher
        from draftfit.training.joint_online import _preserve_rng
        from draftfit.utils import get_local_device
        if not callable(memory_handoff):
            raise ValueError('production observer requires explicit model-memory handoff')
        settings = self.config['serving']; deadline = self.config['deadline_unix']
        phase = self.state['phases'][-1]
        root = self.root/'serving'
        root.mkdir(exist_ok=True)
        holdout = root/'holdout.jsonl'
        text = ''.join(json.dumps(dict(row_index=p['row_index'], domain=p['domain'],
            input_ids=p['prompt_token_ids']), sort_keys=True)+'\n' for p in report['bindings']['prompts'])
        if holdout.exists() and holdout.read_text() != text:
            raise ValueError('fixed production holdout changed between observations')
        if not holdout.exists(): holdout.write_text(text)
        folder = root/(phase['id']+'-'+report['artifact_sha256'])
        folder.mkdir(exist_ok=True)
        with _preserve_rng(get_local_device()):
            exports = producer.prepare_exports(exported, folder/'exports', deadline)
        for name, artifact in artifacts.items():
            artifact.update(checkpoint_sha256=report['artifact_sha256'],
                            export_sha256=fingerprint(exports[name]['files']))
        runtime = producer.runtime_identity(deadline)
        model = stages[0].config.model
        expected = producer.build_expectations(artifacts=artifacts, target_model=model.target_model_path,
            target_revision=model.target_revision or 'local-immutable', validation_path=holdout,
            runtime=runtime, config=settings.get('config'),
            min_requests=_config(self.config['policy'])['min_requests'], deadline_unix=deadline)
        expected['min_domain_requests'] = _config(self.config['policy'])['min_domain_requests']
        self.verify_sources()  # cannot rebind a changed target after native verification
        job = folder/'job'
        if (job/'receipt.json').exists() and read(job/'receipt.json').get('status') == 'verified':
            normalized = dispatcher.read_result(job, expected=expected, deadline_unix=deadline)
        else:
            budget = self.budget(); limits = self.config['policy']['limits']
            cost = settings['cost_usd']
            if time.time() >= deadline or cost > limits['max_usd']-limits['reserve_usd']-budget['spent_usd']-budget.get('reserved_usd',0):
                raise ValueError('external serving admission exceeds remaining budget/deadline')
            attempt = len(list(folder.glob('admission-*.json')))
            admission = dict(approved=True, reservation_id=folder.name+':'+str(attempt),
                deadline_unix=deadline, max_seconds=min(settings['max_seconds'],deadline-time.time()),
                cost_floor=budget['spent_usd']+cost)
            write(folder/f'admission-{attempt:06d}.json',admission)
            normalized = dispatcher.dispatch(job, command=[sys.executable,'-m','draftfit.training.sglang_producer'],
                expected=expected, checkpoint=checkpoint, exports={n:e['path'] for n,e in exports.items()},
                native_report=report, memory_handoff=memory_handoff, admission=admission,
                additional_sources={'target':model.target_model_path,'validation':holdout})
        self.verify_sources()
        history = self.state.setdefault('production_history', {}).setdefault(phase['id'], {})
        update = dispatcher.policy_update(normalized, policies=policies,
            losses={s.name:metrics[s.name]['eval/avg_loss'] for s in stages}, histories=history,
            budget=self.budget(), tried_presets=self.state['tried'])
        self.verify_sources()
        self.state['production_history'][phase['id']] = update['history']
        phase.setdefault('production_evidence', {})[report['artifact_sha256']] = dict(
            expected=deepcopy(expected), source_report_sha256=normalized['source_report_sha256'],
            exports=deepcopy(exported['arms']))
        phase['production_report'] = dict(job=str(job), source_report_sha256=normalized['source_report_sha256'],
                                         target_resident=False, speed_decisions_allowed=False)
        update['current_acceptance'] = {row['name']: sum(d['accepted_tokens'] for d in row['domains']) /
            sum(d['proposed_tokens'] for d in row['domains']) for row in normalized['candidates']}
        return update

    def observe(self, *, step, checkpoint, report, metrics, stages, memory_handoff=None):
        previous_state = deepcopy(self.state)
        self.observation_committed = False
        try:
            return self._observe(step=step, checkpoint=checkpoint, report=report, metrics=metrics, stages=stages, memory_handoff=memory_handoff)
        except BaseException:
            # Partial export/snapshot folders are idempotent; the decision,
            # reservation and observation history commit together or not at all.
            if not self.observation_committed:
                self.state = previous_state
            raise

    def _observe(self, *, step, checkpoint, report, metrics, stages, memory_handoff=None):
        if step == 0:
            # A seed checkpoint replay is not a completed optimizer interval.
            # Match a fresh run: no policy history, resource transition or best.
            return dict(stop=False, reason='seed_checkpoint', protect_steps=[0])
        if self.config.get('serving') is not None: self.verify_sources()
        self.publications()
        if len(self.pending_publications()) >= len(stages):
            self.publications(wait=True)  # bounded backpressure, before another immutable snapshot
        phase = self.state['phases'][-1]
        global_step = phase['offset'] + step
        key = f"{phase['id']}:{step}:{report['artifact_sha256']}"
        previous = phase.get('last_observation')
        if previous and previous['key'] == key:
            self.verify_retained()
            budget = self.budget()
            limits = self.config['policy']['limits']
            if (time.time() >= self.config['deadline_unix'] or
                    budget['spent_usd'] + budget.get('reserved_usd', 0) >= limits['max_usd'] - limits['reserve_usd']):
                return dict(stop=True, reason='budget_or_deadline', protect_steps=[step])
            return previous['response']
        if previous and previous['step'] > step:
            raise ValueError('observer cannot silently roll back committed policy history')
        if sha(checkpoint) != report['artifact_sha256'] or report['tokenparity'] is not True:
            raise ValueError('verified checkpoint/parity required')
        self.state['total_steps'] = global_step
        export_dir = self.root / 'artifacts' / f"{phase['id']}-step{step}"
        updates = {}
        candidates = {c['name']: c for c in report['candidates']}
        decisions = {}
        production_policies, production_artifacts = {}, {}
        exported = None
        bindings = deepcopy(report['bindings'])
        bindings['config'].pop('deadline_unix', None)
        comparison = fingerprint(bindings)
        capture = fingerprint({k: v for k, v in bindings.items() if k != 'config'})
        budget = self.budget()
        for stage in stages:
            name = stage.name
            loss = metrics[name]['eval/avg_loss']
            artifact = dict(artifact_sha256=report['artifact_sha256'], family=stage.config.training.strategy,
                preset=phase['preset'], step=global_step,
                training_fingerprint=fingerprint(stage.config.training.model_dump(mode='json')),
                objective_fingerprint=fingerprint(stage.draft_config.to_dict()), capture_fingerprint=capture)
            # Reuse the native observer's request matching/acceptance normalization.
            policy = deepcopy(self.config['policy'])
            registered = []
            for p in self.config['presets']:
                registered.append({**{k: v for k, v in p.items() if k != 'overrides'},
                    'family': artifact['family'], 'training_fingerprint': fingerprint(p['overrides']),
                    'objective_fingerprint': artifact['objective_fingerprint'], 'capture_fingerprint': capture,
                    'optimizer_mode': 'weights_only', 'preflight_passed': True})
            policy.update(presets=registered, comparison_fingerprint=comparison)
            production_policies[name] = deepcopy(policy)
            production_artifacts[name] = deepcopy(artifact)
            # Normalize native paired requests without pretending they are a production engine.
            paired = {(r['trial'], r['row_index']): r for r in report['baseline']['runs']}
            candidate = candidates[name]
            domains = []
            for domain in report['domains']:
                rows = [r for r in candidate['runs'] if bindings['prompts'][r['row_index']]['domain'] == domain]
                bases = [paired[(r['trial'], r['row_index'])] for r in rows]
                domains.append(dict(name=domain, requests=len(rows), matched_requests=len(rows),
                    exact_token_parity=all(r['token_ids'] == b['token_ids'] for r, b in zip(rows, bases)),
                    output_tokens=sum(r['generated_tokens'] for r in rows),
                    baseline_output_tokens=sum(r['generated_tokens'] for r in bases),
                    seconds=sum(r['seconds'] for r in rows), baseline_seconds=sum(r['seconds'] for r in bases),
                    accepted_tokens=sum(r['accepted_tokens'] for r in rows), proposed_tokens=sum(r['proposed_tokens'] for r in rows)))
            serving = {**{k: artifact[k] for k in ('artifact_sha256', 'family', 'preset', 'step')},
                'comparison_fingerprint': comparison, 'split': 'validation', 'exact_token_parity': candidate['tokenparity'],
                'domains': domains, 'speed_decisions_allowed': False}
            history = self.state['history'].setdefault(phase['id'] + ':' + name, [])
            history.append(dict(artifact=artifact, loss=loss, serving=serving))
            decisions[name] = decide(policy, history, current=artifact, budget=budget, tried_presets=self.state['tried'])
            if (self.config.get('serving') is None and self.config['selection'] == 'acceptance'
                    and not decisions[name]['current_serving_validated']):
                raise ValueError('acceptance selection requires a current validated generation report')
            if self.config.get('serving') is None:
                score = -loss if self.config['selection'] == 'loss' else candidate['acceptance_rate']
                best = self.state['best'].get(name)
                if best is None or score > best['score']:
                    updates[name] = score
        if self.config.get('serving') is not None:
            exported = self.export(checkpoint, stages, export_dir)
            production = self.observe_serving(checkpoint, report, metrics, stages, memory_handoff,
                exported, production_artifacts, production_policies)
            decisions = production['decisions']
            updates = {}
            for name, decision in decisions.items():
                score = production['current_acceptance'][name]
                best = self.state['best'].get(name)
                if best is None or score > best['score']:
                    updates[name] = score
        branch = next((d for d in decisions.values() if d['action'] == 'branch'), None)
        resource_branch = self.observe_resources(global_step, metrics, report)
        if branch is None and all(d['action'] == 'continue' for d in decisions.values()):
            branch = resource_branch
        if branch and branch['source_artifact']['artifact_sha256'] != report['artifact_sha256']:
            raise ValueError('phase source is not the current verified checkpoint')
        exported = exported or (self.export(checkpoint, stages, export_dir) if updates or branch else phase.get('latest_export'))
        for name, score in updates.items():
            self.state['best'][name] = dict(score=score, criterion=self.config['selection'], step=global_step,
                phase=phase['id'], checkpoint=exported['checkpoint'], checkpoint_sha256=exported['checkpoint_sha256'],
                **exported['arms'][name])
            pub = self.config.get('publication')
            if pub:
                from draftfit.export.hf_publish import prepare_publication
                job = self.root / 'publications' / f"{phase['id']}-step{step}-{name}"
                if not job.exists():
                    prepare_publication(exported['arms'][name]['export'], pub['repos'][name], job,
                                        key, private=pub['private'], resume_from=checkpoint)
                if str(job) not in self.state['publication_jobs']:
                    self.state['publication_jobs'].append(str(job))
        stop = branch is not None or any(d['action'] != 'continue' for d in decisions.values())
        reason = branch['reason'] if branch else next((d['reason'] for d in decisions.values() if d['action'] == 'stop'), 'continue')
        if not stop and step < phase['steps'] and step >= phase['admitted_until']:
            if not self.reserve(self.config['policy']['next_cost_usd'], persist=False):
                stop, reason = True, 'budget_or_deadline'
            else:
                phase['admitted_until'] = min(phase['steps'], step + self.config['policy']['next_steps'])
        # Keep the current checkpoint and every best selected within this phase.
        protected = [step] + [b['step'] - phase['offset'] for b in self.state['best'].values() if b['phase'] == phase['id']]
        response = dict(stop=stop, reason=reason, protect_steps=sorted(set(protected)))
        phase.update(checkpoint=str(checkpoint), checkpoint_sha256=report['artifact_sha256'],
                     decisions=decisions, branch=branch, latest_export=exported,
                     last_observation=dict(key=key, step=step, response=response))
        if self.config.get('serving') is not None: self.verify_sources()
        self.save()
        self.observation_committed = True
        self.prune_artifacts()
        self.publications()
        if not stop: self.start_resources()
        return response

    def execute(self):
        from draftfit.training.joint_family import run_joint_family, validate_joint_family
        from draftfit.config.schema import apply_overrides
        try:
            self.verify_retained()
            while self.state['status'] != 'complete':
                self.verify_sources()
                remaining_steps = self.config['max_steps'] - self.budget()['steps_used']
                if remaining_steps <= 0:
                    self.state.update(status='complete', reason='step_budget_exhausted'); self.save(); break
                if not self.reserve(0):
                    self.state.update(status='complete', reason='budget_or_deadline'); self.save(); break
                if not self.state['phases'] or self.state['phases'][-1].get('finished'):
                    if self.state['phases']:
                        prior = self.state['phases'][-1]
                        decision = prior.get('branch')
                        if not decision or self.state['total_steps'] >= self.config['max_steps']:
                            self.state['status'] = 'complete'; self.save(); break
                        preset = next(p for p in self.config['presets'] if p['id'] == decision['next_preset'])
                        if preset['id'] in self.state['tried']:
                            raise ValueError('preset already executed')
                        steps = min(preset['pilot_steps'], remaining_steps)
                        cost, selected = preset['pilot_cost_usd'], preset['id']
                        source = prior['latest_export']
                        if source['checkpoint_sha256'] != decision['source_artifact']['artifact_sha256']:
                            raise ValueError('branch checkpoint differs from selected source')
                        for row in source['arms'].values(): self.verify_export(row)
                        b = self.budget(); limits = self.config['policy']['limits']
                        if b['pilots_used'] >= limits['max_pilots'] or b['exploration_spent_usd'] + cost > limits['max_exploration_usd']:
                            self.state.update(status='complete', reason='exploration_budget_bound'); self.save(); break
                    else:
                        steps = min(self.stages[0].config.training.max_steps, remaining_steps)
                        cost, selected, source = self.config['policy']['next_cost_usd'], 'initial', None
                    if not self.reserve(cost):
                        self.state.update(status='complete', reason='budget_or_deadline'); self.save(); break
                    phase = dict(id=f"phase{len(self.state['phases'])}", preset=selected,
                                 offset=self.state['total_steps'], steps=steps, source=source, attempts=[],
                                 admitted_until=steps if selected != 'initial' else min(steps,self.config['policy']['next_steps']))
                    self.state['phases'].append(phase)
                    if selected != 'initial':
                        self.state['tried'].append(selected)
                        self.state['exploration_floor'] = self.budget()['exploration_spent_usd'] + cost
                        if decision.get('resource_transition'):
                            self.state['resource_state'] = decision['resource_transition']
                    self.state['status'] = 'training'; self.save()
                phase = self.state['phases'][-1]
                output = self.root / phase['id'] / f"attempt{len(phase['attempts'])}"
                output.mkdir(parents=True, exist_ok=False)
                phase['attempts'].append(str(output)); self.save()
                adjusted = []
                for stage in self.stages:
                    cfg = stage.config.model_copy(deep=True)
                    if phase['preset'] != 'initial':
                        preset = next(p for p in self.config['presets'] if p['id'] == phase['preset'])
                        cfg = apply_overrides(cfg, preset['overrides'])
                        cfg.model.draft_checkpoint_path = phase['source']['arms'][stage.name]['export']
                        cfg.training.max_steps = cfg.training.total_steps = phase['steps']
                    cfg.output_dir = str(output / stage.name)
                    Path(cfg.output_dir).mkdir()
                    cfg.model.draft_model_config = str(Path(cfg.output_dir) / 'draft.json')
                    stage.draft_config.to_json_file(cfg.model.draft_model_config)
                    write(Path(cfg.output_dir) / 'resolved_config.json', cfg.model_dump(mode='json'))
                    adjusted.append(replace(stage, config=cfg))
                validate_joint_family(adjusted)
                checkpoints = [p for attempt in phase['attempts'][:-1] for p in Path(attempt).glob('joint-step*.pt')]
                committed = [p for p in checkpoints if p.with_suffix('.sha256').is_file()]
                if checkpoints and not committed:
                    raise ValueError('no checkpoint has a committed integrity receipt')
                phase['uncommitted_checkpoints'] = [str(p) for p in checkpoints if p not in committed]
                resume = max(committed, key=lambda p: (int(p.stem.removeprefix('joint-step')), p.stat().st_mtime_ns)) if committed else None
                if resume is not None:
                    checksum = resume.with_suffix('.sha256')
                    if not checksum.is_file() or checksum.read_text().strip() != sha(resume):
                        raise ValueError('resume checkpoint integrity receipt missing or mismatched')
                local_start = int(resume.stem.removeprefix('joint-step')) if resume else 0
                operational_stop = min(phase['steps'], local_start + remaining_steps)
                self.start_resources()
                result = run_joint_family(output, adjusted, resume_from=resume,
                    stop_at_step=operational_stop, deadline_unix=self.config['deadline_unix'], milestone_observer=self.observe,
                    **({'milestone_runtime': True} if self.config.get('serving') is not None else {}))
                completed = {r['global_step'] for r in result['runs']}
                if len(completed) != 1: raise ValueError('joint arms disagree on committed step')
                self.state['total_steps'] = phase['offset'] + completed.pop()
                phase['resume_checkpoint'] = result['runs'][0]['joint_checkpoint']
                phase['resume_sha256'] = sha(phase['resume_checkpoint'])
                phase['result'] = result
                phase['finished'] = True
                if result.get('reason') == 'deadline': phase['branch'] = None
                self.save()
            self.publications(wait=True)
            self.state['publication_complete'] = True
            self.save()
            return self.state
        except BaseException:
            self.state['status'] = 'failed'
            self.save()
            raise
        finally:
            if self.sampler is not None: self.sampler.stop()
            self.publisher.close()
            self.lock.close()


def run(config_path, *, resume=False, plan=False):
    config, stages, binding = load_plan(config_path)
    if plan:
        return dict(schema=SCHEMA, binding=binding, arms=[s.name for s in stages], config=config)
    return Controller(config, stages, binding, resume=resume).execute()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config, resume=args.resume, plan=args.plan), indent=2))


if __name__ == '__main__':
    main()
