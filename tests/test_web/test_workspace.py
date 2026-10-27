"""Current engine boundaries and real local subprocesses through the HTTP adapter."""

import json
import sys
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from draftfit.target_project import algorithm_catalog
from draftfit.web.server import create_app
from draftfit.web.store import Store

ROOT = Path(__file__).resolve().parents[2]


def config():
    return yaml.safe_load(
        (ROOT / "examples/configs/qwen3-8b-dspark-offline.yaml").read_text()
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DRAFTFIT_TRAINER_PYTHON", sys.executable)
    with TestClient(create_app(tmp_path, tmp_path / "web")) as client:
        client.headers["x-draftfit-token"] = client.get("/api/bootstrap").json()[
            "token"
        ]
        yield client


def save(client, cfg=None):
    result = client.post(
        "/api/runs", json={"name": "Adapter test", "config": cfg or config()}
    )
    assert result.status_code == 200, result.text
    return result.json()


def await_job(client, ident, seconds=15):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        job = next(j for j in client.get("/api/jobs").json() if j["id"] == ident)
        if job["status"] not in ("queued", "running", "cancelling"):
            return job
        time.sleep(0.03)
    pytest.fail("Job did not complete within the test bound.")


def test_bootstrap_uses_current_catalog_and_does_not_load_a_trainer(client):
    value = client.get("/api/bootstrap").json()
    assert value["engine"]["package"] == "draftfit"
    assert value["algorithms"] == algorithm_catalog()
    assert not value["allow_local_execution"]
    assert "research" not in value["engine"]["actions"]
    assert "research_schema" not in value
    assert value["recipes"]


def test_validation_uses_algorithm_and_topology_contracts(client):
    assert (
        client.post("/api/config/validate", json=config()).json()["gpu_verified"]
        is False
    )
    raw = config()
    raw["training"]["tp_size"] = 2
    result = client.post("/api/config/validate", json=raw)
    assert result.status_code == 400
    assert "tensor parallelism" in result.text
    raw["training"].update(strategy="unregistered", tp_size=1)
    assert client.post("/api/config/validate", json=raw).status_code == 400


@pytest.mark.parametrize(
    "payload", ["[broken", "true", "[]", "null", "x" * (1024 * 1024 + 1)]
)
def test_config_import_rejects_invalid_content(client, payload):
    assert (
        client.post(
            "/api/config/import", files={"file": ("bad.yaml", payload)}
        ).status_code
        == 400
    )


def test_yaml_import_and_config_secrets(client):
    response = client.post(
        "/api/config/import", files={"file": ("train.yaml", yaml.safe_dump(config()))}
    )
    assert response.status_code == 200
    raw = config()
    raw["tracking"] = {"wandb_key": "not-a-real-key"}
    result = client.post("/api/config/validate", json=raw)
    assert result.status_code == 400
    assert "not-a-real-key" not in result.text
    assert "environment variables" in result.text


def test_upload_preparation_holdout_and_failed_upload_cleanup(client):
    payload = "\n".join(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": f"question {i}"},
                    {"role": "assistant", "content": f"answer {i}"},
                ]
            }
        )
        for i in range(25)
    )
    result = client.post("/api/datasets", files={"file": ("data.jsonl", payload)})
    assert result.status_code == 200, result.text
    dataset = result.json()
    assert dataset["train_rows"] + dataset["eval_rows"] == 25
    assert dataset["eval_rows"] > 0
    assert Path(dataset["train_path"]).is_file()
    assert (
        client.post(
            "/api/datasets", files={"file": ("bad.jsonl", "broken")}
        ).status_code
        == 400
    )
    assert len(client.get("/api/bootstrap").json()["datasets"]) == 1
    assert len(list(client.app.state.store.path("datasets").iterdir())) == 1
    bad_offline = client.post(
        "/api/runs",
        json={
            "name": "Wrong data mode",
            "config": config(),
            "dataset_id": dataset["id"],
        },
    )
    assert bad_offline.status_code == 400
    assert "offline feature cache" in bad_offline.text
    raw = yaml.safe_load(
        (ROOT / "examples/configs/qwen3-4b-dspark-disaggregated.yaml").read_text()
    )
    saved = client.post(
        "/api/runs",
        json={"name": "Online data", "config": raw, "dataset_id": dataset["id"]},
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["config"]["data"]["train_data_path"] == dataset["train_path"]
    assert not saved.json()["config"]["data"]["eval_data_path"]
    assert Path(dataset["eval_path"]).is_file()


@pytest.mark.parametrize(
    "action",
    [
        "research",
        "research_plan",
        "pretrain",
        "joint_pipeline",
        "capture",
        "serve",
        "publish",
        "shell",
    ],
)
def test_unsupported_actions_do_not_create_jobs(client, action):
    run = save(client)
    result = client.post(f"/api/runs/{run['id']}/execute", json={"action": action})
    assert result.status_code == 400
    assert "Unsupported web action" in result.text
    assert client.get("/api/jobs").json() == []


def test_preview_is_side_effect_free_plan_is_real_and_config_change_rejected(client):
    run = save(client)
    route = f"/api/runs/{run['id']}"
    preview = client.post(route + "/preview", json={"action": "plan"})
    assert preview.status_code == 200
    assert preview.json()["commands"][0][1:4] == ["-m", "draftfit.cli", "train"]
    assert preview.json()["commands"][0][-1] == "--plan"
    assert client.get("/api/jobs").json() == []
    submitted = client.post(route + "/execute", json={"action": "plan"})
    assert submitted.status_code == 200, submitted.text
    job = await_job(client, submitted.json()["id"])
    output = client.get(f"/api/jobs/{job['id']}/log").text
    assert job["status"] == "succeeded", output
    assert '"kind": "worker"' in output
    assert not (client.app.state.engine.project / "outputs").exists()
    Path(run["config_path"]).write_text("{}")
    assert client.post(route + "/execute", json={"action": "plan"}).status_code == 400


def test_real_token_comparison_keeps_mismatch_as_failed(client):
    run = save(client)
    expected = client.app.state.engine.project / "expected.json"
    actual = expected.with_name("actual.json")
    expected.write_text("[1,2,3]")
    actual.write_text("[1,2,3]")
    body = {
        "action": "validate_tokens",
        "params": {"expected": str(expected), "actual": str(actual)},
    }

    def run_gate():
        result = client.post(f"/api/runs/{run['id']}/execute", json=body)
        assert result.status_code == 200, result.text
        job = await_job(client, result.json()["id"])
        return job, client.get(f"/api/jobs/{job['id']}/log").text

    passed, output = run_gate()
    assert passed["status"] == "succeeded", output
    assert '"passed": true' in output
    actual.write_text("[1,9,3]")
    failed, output = run_gate()
    assert failed["status"] == "failed"
    assert failed["exit_code"] != 0
    assert '"passed": false' in output


def test_loopback_origin_token_and_remote_fields(client):
    assert (
        client.get("/api/bootstrap", headers={"host": "evil.example"}).status_code
        == 403
    )
    route = "/api/config/validate"
    assert (
        client.post(
            route, json=config(), headers={"x-draftfit-token": "wrong"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            route, json=config(), headers={"origin": "https://evil.example"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            route, json=config(), headers={"origin": "http://127.0.0.1:5173"}
        ).status_code
        == 200
    )
    run = save(client)
    assert (
        client.post(
            f"/api/runs/{run['id']}/execute", json={"action": "train"}
        ).status_code
        == 409
    )
    assert (
        client.post(
            f"/api/runs/{run['id']}/execute",
            json={"action": "plan", "remote_instance": 1},
        ).status_code
        == 422
    )


def controller_file(client):
    root = client.app.state.engine.project
    for name in ("family.json", "budget.json"):
        (root / name).write_text("{}")
    path = root / "controller.json"
    path.write_text(
        json.dumps(
            {
                "schema": "draftfit-unattended-v1",
                "manifest": "family.json",
                "budget_path": "budget.json",
                "output_dir": "controller-output",
            }
        )
    )
    return path


def test_unattended_reads_only_latest_attempt_without_editing_controller(client):
    path = controller_file(client)
    output = path.parent / "controller-output"
    old, current = output / "phase0/attempt0", output / "phase1/attempt0"
    for attempt in (old, current):
        attempt.mkdir(parents=True)
    (old / "learning-curves.jsonl").write_text(
        '{"kind":"generation_verification","step":1}\n'
    )
    (current / "learning-curves.jsonl").write_text(
        '{"kind":"capture","step":2,"queue":{"backpressure":3}}\n{"kind":'
    )
    state = {
        "status": "failed",
        "total_steps": 2,
        "phases": [
            {"attempts": [str(old)]},
            {"attempts": [str(current)]},
        ],
    }
    state_path = output / "state.json"
    state_path.write_text(json.dumps(state))
    before = state_path.read_bytes(), path.read_bytes()
    response = client.get("/api/unattended/status", params={"config_path": str(path)})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["state"]["status"] == "failed"
    assert [r["kind"] for r in result["events"]] == ["capture"]
    assert result["queue"]["backpressure"] == 3
    assert before == (state_path.read_bytes(), path.read_bytes())
    body = {"action": "unattended_resume", "params": {"config_path": str(path)}}
    command = client.post("/api/unattended/preview", json=body).json()["commands"][0]
    assert command[-1] == "--resume"
    assert client.post("/api/unattended/execute", json=body).status_code == 409


def test_paths_symlinks_publication_and_arbitrary_params_rejected(client, tmp_path):
    path = controller_file(client)
    raw = json.loads(path.read_text())
    raw["publication"] = {"owner": "example"}
    path.write_text(json.dumps(raw))
    body = {"action": "unattended_plan", "params": {"config_path": str(path)}}
    result = client.post("/api/unattended/preview", json=body)
    assert result.status_code == 400
    assert "Publication" in result.text
    run = save(client)
    outside = tmp_path.parent / (tmp_path.name + "-outside.json")
    outside.write_text("[]")
    link = tmp_path / "link.json"
    link.symlink_to(outside)
    try:
        result = client.post(
            f"/api/runs/{run['id']}/preview",
            json={
                "action": "validate_tokens",
                "params": {"expected": str(link), "actual": str(link)},
            },
        )
        assert result.status_code == 400
        assert (
            client.post(
                f"/api/runs/{run['id']}/preview",
                json={"action": "train", "params": {"python": "arbitrary"}},
            ).status_code
            == 400
        )
    finally:
        outside.unlink()


def test_store_restart_marks_only_ui_jobs_interrupted(tmp_path):
    store = Store(tmp_path)
    record = store.create("job", status="running")
    with pytest.raises(ValueError, match="Another workspace"):
        Store(tmp_path)
    store.close()
    reopened = Store(tmp_path)
    assert reopened.get("job", record["id"])["status"] == "interrupted"
    reopened.close()


def test_active_process_conflict_and_actual_cancellation(client):
    job = client.app.state.jobs.start(
        "bounded process",
        [[sys.executable, "-c", "import time; time.sleep(30)"]],
        client.app.state.engine.project,
        timeout=10,
    )
    run = save(client)
    blocked = client.post(f"/api/runs/{run['id']}/execute", json={"action": "plan"})
    assert blocked.status_code == 409
    assert client.post(f"/api/jobs/{job['id']}/cancel", json={}).status_code == 200
    assert await_job(client, job["id"])["status"] == "cancelled"


def test_actual_process_timeout_is_failed(client):
    job = client.app.state.jobs.start(
        "timeout",
        [[sys.executable, "-c", "import time; time.sleep(30)"]],
        client.app.state.engine.project,
        timeout=0.2,
    )
    result = await_job(client, job["id"])
    assert result["status"] == "failed"
    assert "timed out" in result["error"]
