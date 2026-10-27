"""Loopback-only optional workspace; all training remains in the Draftfit CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from draftfit.data.prepare import prepare_dataset
from .engine import Engine
from .jobs import Jobs
from .store import Store
from .workflow import build_commands, read_tail, unattended_status


class RunInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    config: dict
    dataset_id: str | None = None


class Execute(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: str
    params: dict = Field(default_factory=dict)
    timeout: int = Field(default=3600, ge=1, le=86400)


def create_app(project_root=None, state_root=None, *, load_env=False,
               allow_local_execution=False):
    project = Path(project_root or Path.cwd()).resolve()
    if load_env:
        from dotenv import load_dotenv
        env_file = project / ".env"
        fallback = os.getenv("DRAFTFIT_FALLBACK_ENV")
        if env_file.is_file():
            load_dotenv(env_file, override=False)
        elif fallback and Path(fallback).is_file():
            load_dotenv(fallback, override=False)
    engine = Engine(project)
    store = Store(Path(state_root or project / "outputs/web-workspace"))
    jobs = Jobs(store, env={"PYTHONPATH": str(engine.root) + os.pathsep + os.environ.get("PYTHONPATH", "")})
    token = uuid.uuid4().hex
    execution_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app):
        yield
        jobs.close()
        store.close()

    app = FastAPI(title="Draftfit Workspace", docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)
    app.state.store, app.state.jobs, app.state.engine = store, jobs, engine

    @app.middleware("http")
    async def personal_boundary(request: Request, call_next):
        if request.url.hostname not in ("localhost", "127.0.0.1", "testserver"):
            return JSONResponse({"detail": "The workspace is available on loopback only."}, status_code=403)
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and origin not in {
                str(request.base_url).rstrip("/"),
                "http://localhost:5173", "http://127.0.0.1:5173",
            }:
                return JSONResponse({"detail": "Request origin is not allowed."}, status_code=403)
            if request.headers.get("x-draftfit-token") != token:
                return JSONResponse({"detail": "Refresh the workspace and try again."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api") else "no-cache"
        return response

    @app.exception_handler(ValidationError)
    async def invalid_config(_request, error):
        return JSONResponse({"detail": error.errors(include_input=False, include_context=False)}, status_code=400)

    @app.exception_handler(ValueError)
    async def bad_input(_request, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.exception_handler(FileNotFoundError)
    async def missing_file(_request, _error):
        return JSONResponse({"detail": "File not found. Check the project path and runtime."}, status_code=404)

    @app.get("/api/bootstrap")
    def bootstrap():
        recipes = []
        for path in sorted((engine.root / "examples/configs").glob("*.yaml")):
            try:
                raw = yaml.safe_load(path.read_text())
                engine.validate(raw)
            except (ValueError, TypeError):
                continue
            recipes.append({"name": path.stem, "config": raw})
        return {"token": token, "engine": engine.info(), "algorithms": engine.catalog(),
                "schema": engine.Config.model_json_schema(), "recipes": recipes,
                "runs": store.list("run"), "datasets": store.list("dataset"),
                "jobs": store.list("job"), "allow_local_execution": allow_local_execution}

    @app.post("/api/config/validate")
    def validate_config(body: dict):
        return engine.validate(body)

    @app.post("/api/config/import")
    async def import_config(file: UploadFile = File(...)):
        try:
            content = await file.read(1024 * 1024 + 1)
            if len(content) > 1024 * 1024:
                raise ValueError("Config uploads are limited to 1 MiB.")
            try:
                raw = yaml.safe_load(content)
            except yaml.YAMLError as exc:
                raise ValueError("Check the JSON/YAML syntax.") from exc
            if not isinstance(raw, dict):
                raise ValueError("The config must be a JSON/YAML object.")
            return engine.validate(raw)
        finally:
            await file.close()

    @app.post("/api/datasets")
    async def upload(file: UploadFile = File(...), data_format: str = Form("auto"),
                     split_eval: bool = Form(True)):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in (".json", ".jsonl"):
            raise ValueError("Choose a JSON or JSONL conversation file.")
        ident = uuid.uuid4().hex[:16]
        directory = store.path("datasets/" + ident)
        directory.mkdir(parents=True)
        source = directory / ("source" + suffix)
        total = 0
        digest = hashlib.sha256()
        try:
            with source.open("xb") as stream:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > 64 * 1024 * 1024:
                        raise ValueError("Dataset uploads are limited to 64 MiB.")
                    digest.update(chunk)
                    stream.write(chunk)
            summary = prepare_dataset(source, directory / "train.jsonl", data_format=data_format,
                                      split_eval=split_eval, eval_output_path=directory / "eval.jsonl",
                                      seed=42).as_dict()
            record = {"id": ident, "created_at": time.time(), "name": Path(file.filename).name,
                      "bytes": total, "sha256": digest.hexdigest(), **summary}
            return store.put("dataset", record)
        except Exception:
            shutil.rmtree(directory)
            raise
        finally:
            await file.close()

    @app.post("/api/runs")
    def save_run(body: RunInput):
        raw = json.loads(json.dumps(body.config))
        if body.dataset_id:
            dataset = store.get("dataset", body.dataset_id)
            if raw.get("data", {}).get("hidden_states_path"):
                raise ValueError("Raw conversations cannot replace an offline feature cache. Capture features with the engine first.")
            # The current single-run server capture path rejects online eval.
            # Keep the prepared holdout in the dataset record for later capture
            # or verification; do not inject an unsupported eval_data_path.
            raw.setdefault("data", {}).update(train_data_path=dataset["train_path"],
                                               prompts_path="")
        validated = engine.validate(raw)["config"]
        ident = uuid.uuid4().hex[:16]
        directory = store.path("runs/" + ident)
        directory.mkdir(parents=True)
        config_file = directory / "config.json"
        config_file.write_text(json.dumps(validated, indent=2) + "\n")
        return store.put("run", {"id": ident, "created_at": time.time(), "name": body.name,
                                 "config": validated, "dataset_id": body.dataset_id,
                                 "config_path": str(config_file),
                                 "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest()})

    def commands_for(ident, body):
        run = store.get("run", ident) if ident else None
        if run:
            config_path = store.path("runs/" + ident + "/config.json")
            if hashlib.sha256(config_path.read_bytes()).hexdigest() != run["config_sha256"]:
                raise ValueError("The saved config changed. Save a new experiment before executing.")
            engine.validate(run["config"])
        else:
            config_path = None
        if ident and body.action.startswith("unattended") or not ident and not body.action.startswith("unattended"):
            raise ValueError("Choose the matching experiment or unattended action endpoint.")
        return build_commands(engine, body.action, config_path=config_path, params=body.params)

    def start(ident, body):
        commands = commands_for(ident, body)
        if body.action in ("train", "unattended", "unattended_resume") and not allow_local_execution:
            raise HTTPException(409, "Local training is disabled. Start the server with --allow-local-execution in a provisioned runtime.")
        with execution_lock:
            if any(job["status"] in ("queued", "running", "cancelling") for job in store.list("job")):
                raise HTTPException(409, "A workspace job is active. Wait for it or stop it first.")
            return jobs.start(body.action, commands, project,
                              metadata={"run_id": ident, "action": body.action}, timeout=body.timeout)

    @app.post("/api/runs/{ident}/preview")
    def preview(ident: str, body: Execute):
        return {"commands": commands_for(ident, body), "cwd": str(project),
                "executes_training": body.action == "train"}

    @app.post("/api/runs/{ident}/execute")
    def execute(ident: str, body: Execute):
        return start(ident, body)

    @app.post("/api/unattended/preview")
    def preview_unattended(body: Execute):
        return {"commands": commands_for(None, body), "cwd": str(project),
                "executes_training": body.action != "unattended_plan"}

    @app.post("/api/unattended/execute")
    def execute_unattended(body: Execute):
        return start(None, body)

    @app.get("/api/unattended/status")
    def status(config_path: str):
        return unattended_status(engine, config_path)

    @app.get("/api/jobs")
    def list_jobs():
        return store.list("job")

    @app.get("/api/jobs/{ident}/log", response_class=PlainTextResponse)
    def job_log(ident: str):
        job = store.get("job", ident)
        path = store.path(job["log_path"])
        return read_tail(path) if path.exists() else ""

    @app.post("/api/jobs/{ident}/cancel")
    def cancel(ident: str):
        return jobs.cancel(ident)

    dist = engine.root / "web/dist"
    if dist.is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/")
        def index():
            return FileResponse(dist / "index.html")
    else:
        @app.get("/")
        def missing_frontend():
            return PlainTextResponse("Build the frontend with npm ci --prefix web && npm run build --prefix web.", status_code=503)
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path.cwd()))
    parser.add_argument("--state-dir")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--allow-local-execution", action="store_true")
    args = parser.parse_args(argv)
    import uvicorn
    uvicorn.run(create_app(args.project, args.state_dir, load_env=True,
                           allow_local_execution=args.allow_local_execution),
                host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
