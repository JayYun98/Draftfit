"""Real process and HTTP regressions for engine ownership and teardown."""

import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from draftfit.web.server import create_app

from .test_workspace import ROOT, await_job, save


def wait_until(predicate, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    pytest.fail("The isolated process did not reach the expected state.")


def alive(pid):
    status = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=2
    ).stdout.strip()
    return bool(status) and not status.startswith("Z")


def kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_selected_project_cannot_shadow_engine_in_cli_or_nested_workers(
    tmp_path, monkeypatch
):
    project = tmp_path / "selected-project"
    shadow = project / "draftfit"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("")
    (shadow / "cli.py").write_text("print('SHADOW_ENGINE'); raise SystemExit(7)\n")
    (project / "expected.json").write_text("[1,2,3]")
    (project / "actual.json").write_text("[1,2,3]")
    runtime = tmp_path / "runtime-startup"
    runtime.mkdir()
    startup = tmp_path / "startup-pids.txt"
    (runtime / "sitecustomize.py").write_text(
        "import os\nfrom pathlib import Path\n"
        f"with Path({str(startup)!r}).open('a') as stream: stream.write(str(os.getpid())+'\\n')\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(runtime))
    monkeypatch.setenv("DRAFTFIT_TRAINER_PYTHON", sys.executable)
    # Exercise ordinary startup path behavior, including on Python 3.10.
    monkeypatch.delenv("PYTHONSAFEPATH", raising=False)
    with TestClient(create_app(project, tmp_path / "web")) as client:
        bootstrap = client.get("/api/bootstrap").json()
        assert Path(bootstrap["engine"]["root"]) == ROOT
        client.headers["x-draftfit-token"] = bootstrap["token"]
        run = save(client)
        response = client.post(
            f"/api/runs/{run['id']}/execute", json={"action": "plan"}
        )
        assert response.status_code == 200, response.text
        plan = await_job(client, response.json()["id"])
        output = client.get(f"/api/jobs/{plan['id']}/log").text
        assert plan["status"] == "succeeded", output
        assert '"kind": "worker"' in output
        assert "SHADOW_ENGINE" not in output
        probe = """
import json,os,subprocess,sys,draftfit
print(json.dumps(dict(engine=draftfit.__file__,cwd=os.getcwd(),pid=os.getpid())),flush=True)
subprocess.run([sys.executable,'-m','draftfit.cli','validate','tokens',
               '--expected','expected.json','--actual','actual.json'],check=True)
"""
        job = client.app.state.jobs.start(
            "import/cwd probe", [[sys.executable, "-c", probe]], project
        )
        finished = await_job(client, job["id"])
        output = client.get(f"/api/jobs/{job['id']}/log").text
        assert finished["status"] == "succeeded", output
        observation = next(
            json.loads(line)
            for line in output.splitlines()
            if line.startswith('{"engine"')
        )
        assert Path(observation["engine"]).resolve() == ROOT / "draftfit/__init__.py"
        assert Path(observation["cwd"]).resolve() == project
        assert '"passed": true' in output and "SHADOW_ENGINE" not in output
        # The private guard must retain provisioned runtime startup hooks.
        pids = {int(line) for line in startup.read_text().splitlines()}
        assert observation["pid"] in pids
        assert len(pids) >= 3  # plan, probe, nested engine CLI


SLEEPER = """
import os,signal,sys,time
from pathlib import Path
signal.signal(signal.SIGTERM,signal.SIG_IGN)
Path(sys.argv[1]).write_text(str(os.getpid()))
print('OWNED_PROCESS_READY',flush=True)
time.sleep(180)
"""


def start_server(tmp_path, mode):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    worker = tmp_path / "worker.pid"
    service_paths = (
        [tmp_path / f"service-{i}.pid" for i in range(2)] if mode == "shutdown" else []
    )
    supervisor = f"""
import sys,time
from pathlib import Path
from draftfit.launch_plan import CommandSpec,LaunchPlan,ReadinessSpec,ServiceSpec,run_commands
root=Path({str(tmp_path)!r})
command=CommandSpec('cpu-only-worker',(sys.executable,'-c',{SLEEPER!r},str(root/'worker.pid')))
services=tuple(ServiceSpec(CommandSpec('cpu-only-service-'+str(i),
    (sys.executable,'-c',{SLEEPER!r},str(root/('service-'+str(i)+'.pid')))),
    ReadinessSpec('http','http://127.0.0.1:1',5),str(root/('service-'+str(i)+'.log')),i)
    for i in range({len(service_paths)}))
def ready(service,process,started):
    deadline=time.monotonic()+5
    marker=Path(service.command.argv[-1])
    while not marker.exists():
        if process.poll() is not None or time.monotonic()>deadline: raise RuntimeError('fixture service failed')
        time.sleep(.02)
options={{'shutdown_grace_s':5}} if services else ({{'shutdown_grace_s':11}} if {mode!r}=='timeout' else {{}})
plan=LaunchPlan('managed_supervisor' if services else 'supervisor','all',
    commands=(command,),services=services,managed_root=str(root/'managed'),**options)
print('ENGINE_GRACE',plan.shutdown_grace_s,flush=True)
raise SystemExit(run_commands(plan,managed_preflight=lambda _:None,readiness_waiter=ready))
"""
    server = f"""
import sys,uvicorn
from draftfit.web.server import create_app
app=create_app({str(tmp_path)!r},{str(tmp_path / 'web')!r})
app.state.jobs.start('engine lifecycle fixture',[[sys.executable,'-c',{supervisor!r}]],
                     {str(tmp_path)!r},timeout={3 if mode == 'timeout' else 120})
uvicorn.run(app,host='127.0.0.1',port={port},log_level='warning')
"""
    log = (tmp_path / "server.log").open("w")
    process = subprocess.Popen(
        [sys.executable, "-c", server],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "SPECFORGE_DEVICE": "cpu",
        },
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log, port, [worker, *service_paths]


@pytest.mark.parametrize("mode", ["cancel", "shutdown", "timeout"])
def test_real_http_engine_cleanup_finishes_before_store_shutdown(tmp_path, mode):
    server = unrelated = None
    stream = None
    paths = []
    try:
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(180)"],
            start_new_session=True,
        )
        server, stream, port, paths = start_server(tmp_path, mode)
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=3) as client:

            def ready():
                try:
                    return client.get("/api/bootstrap").status_code == 200 and all(
                        p.exists() for p in paths
                    )
                except httpx.TransportError:
                    return False

            wait_until(ready)
            bootstrap = client.get("/api/bootstrap").json()
            client.headers["x-draftfit-token"] = bootstrap["token"]
            ident = bootstrap["jobs"][0]["id"]
            owned = [int(path.read_text()) for path in paths]
            assert all(os.getpgid(pid) == pid for pid in owned)
            if mode == "shutdown":
                server.send_signal(signal.SIGTERM)
                server.wait(timeout=45)
                # Uvicorn can re-raise the received signal after graceful exit.
                assert server.returncode in (0, -signal.SIGTERM), (
                    tmp_path / "server.log"
                ).read_text()
                with sqlite3.connect(tmp_path / "web/workspace.sqlite") as database:
                    job = json.loads(
                        database.execute(
                            "SELECT body FROM records WHERE id=?", (ident,)
                        ).fetchone()[0]
                    )
            else:
                if mode == "cancel":
                    assert (
                        client.post(f"/api/jobs/{ident}/cancel", json={}).status_code
                        == 200
                    )
                    assert (
                        client.post(f"/api/jobs/{ident}/cancel", json={}).status_code
                        == 200
                    )

                def finished():
                    return client.get("/api/jobs").json()[0]["status"] not in (
                        "queued",
                        "running",
                        "cancelling",
                    )

                wait_until(finished, seconds=40)
                job = client.get("/api/jobs").json()[0]
            assert job["status"] == (
                "failed" if mode == "timeout" else "cancelled"
            ), job
            if mode == "timeout":
                assert "timed out" in job["error"]
            assert job["exit_code"] != -signal.SIGKILL
            wait_until(lambda: not any(alive(pid) for pid in owned), seconds=3)
            assert unrelated.poll() is None
            if server.poll() is None:
                server.send_signal(signal.SIGTERM)
                server.wait(timeout=10)
            assert server.returncode in (0, -signal.SIGTERM)
            assert unrelated.poll() is None
    finally:
        # Always clean up this test's exact groups, including on the old failure.
        for path in paths:
            if path.exists():
                kill_group(int(path.read_text()))
        if server is not None:
            kill_group(server.pid)
            server.wait(timeout=5)
        if unrelated is not None:
            kill_group(unrelated.pid)
            unrelated.wait(timeout=5)
        if stream is not None:
            stream.close()
