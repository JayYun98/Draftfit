"""Opt-in actual tiny CPU engine execution through the optional web adapter."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from draftfit.web.server import create_app
from .test_workspace import ROOT, await_job


@pytest.mark.skipif(not os.getenv("DRAFTFIT_TRAINER_PYTHON"), reason="Set a provisioned CPU DRAFTFIT_TRAINER_PYTHON to run actual engine integration.")
def test_actual_cpu_training_and_unattended_plan_run_resume(tmp_path, monkeypatch):
    python = os.environ["DRAFTFIT_TRAINER_PYTHON"]
    for key, value in {"SPECFORGE_DEVICE": "cpu", "OMP_NUM_THREADS": "1",
                       "MKL_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}.items():
        monkeypatch.setenv(key, value)
    # Reuse the current engine's proven tiny fixtures. No downloaded weights,
    # tokenizer network access or fake controller implementation is involved.
    setup = '''
import json,sys,time,torch
from pathlib import Path
from tests.test_family_joint_cpu import fixture
from tests.test_joint_online_cpu import online_fixture
from tests.test_joint_policy import policy
root=Path(sys.argv[1])
torch.set_num_threads(1)
offline=root/'offline'; offline.mkdir(); fixture(offline,False)
raw=json.loads((offline/'base.json').read_text())
raw['training'].update(max_steps=1,save_interval=1,eval_interval=1)
raw['output_dir']=str(offline/'trained')
(offline/'train-config.json').write_text(json.dumps(raw))
online=root/'online'; online.mkdir(); online_fixture(online,False)
base=json.loads((online/'base.json').read_text())
base['training'].update(max_steps=2,total_steps=2,num_epochs=2,joint_feature_queue_bytes=32768,joint_feature_queue_batches=4,max_checkpoints=1,joint_verification=dict(interval=1,max_prompts=2,max_new_tokens=4,draft_tokens=2,trials=1))
(online/'base.json').write_text(json.dumps(base))
rows=[json.loads(x) for x in (online/'eval.jsonl').read_text().splitlines()]
for row in rows: row['domain']='general'
(online/'eval.jsonl').write_text(''.join(json.dumps(r)+'\\n' for r in rows))
(online/'budget.json').write_text(json.dumps(dict(timestamp_unix=time.time(),spent_usd=0)))
settings=dict(schema='draftfit-unattended-v1',manifest='family.json',output_dir='controller-output',max_steps=2,deadline_unix=time.time()+180,budget_path='budget.json',selection='acceptance',policy=policy(domains=['general'],patience=1,min_delta=1e6,min_requests=2,next_steps=1))
settings['policy']['limits']['max_steps']=2
(online/'controller.json').write_text(json.dumps(settings))
'''
    prepared = subprocess.run([python, "-c", setup, str(tmp_path)], cwd=ROOT,
                              env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True, timeout=60)
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    with TestClient(create_app(tmp_path, tmp_path / "web", allow_local_execution=True)) as client:
        client.headers["x-draftfit-token"] = client.get("/api/bootstrap").json()["token"]
        raw = json.loads((tmp_path / "offline/train-config.json").read_text())
        saved = client.post("/api/runs", json={"name": "Actual CPU", "config": raw})
        assert saved.status_code == 200, saved.text
        run = saved.json()
        started = client.post(f"/api/runs/{run['id']}/execute", json={"action": "train", "timeout": 90})
        assert started.status_code == 200, started.text
        trained = await_job(client, started.json()["id"], 90)
        log = client.get(f"/api/jobs/{trained['id']}/log").text
        assert trained["status"] == "succeeded", log
        checkpoints = list((tmp_path / "offline/trained").rglob("training_state.pt"))
        assert checkpoints
        verified = subprocess.run([python, "-c",
            "import sys,torch; state=torch.load(sys.argv[1],map_location='cpu',weights_only=True); assert state['global_step']==1",
            str(checkpoints[0])], capture_output=True, text=True, timeout=20)
        assert verified.returncode == 0, verified.stdout + verified.stderr

        controller = tmp_path / "online/controller.json"
        state = controller.parent / "controller-output/state.json"
        def execute(action):
            response = client.post("/api/unattended/execute", json={"action": action,
                "params": {"config_path": str(controller)}, "timeout": 90})
            assert response.status_code == 200, response.text
            job = await_job(client, response.json()["id"], 90)
            assert job["status"] == "succeeded", client.get(f"/api/jobs/{job['id']}/log").text
        execute("unattended_plan")
        assert not state.exists()
        execute("unattended")
        observation = client.get("/api/unattended/status", params={"config_path": str(controller)}).json()
        assert observation["state"]["status"] == "complete"
        assert observation["state"]["total_steps"] == 2
        assert observation["queue"] is not None
        assert any(row["kind"] == "generation_verification" for row in observation["events"])
        before = hashlib.sha256(state.read_bytes()).hexdigest()
        execute("unattended_resume")
        assert hashlib.sha256(state.read_bytes()).hexdigest() == before
