"""Use the imported Draftfit engine's public composition and CLI boundaries."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import draftfit
from draftfit.application.composition import resolve_run
from draftfit.config import Config
from draftfit.target_project import algorithm_catalog


def reject_secrets(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                str(key).lower()
                in {
                    "hf_token",
                    "huggingface_token",
                    "access_token",
                    "api_key",
                    "wandb_key",
                    "swanlab_key",
                    "password",
                    "secret",
                }
                and item
            ):
                raise ValueError("Use process environment variables for credentials.")
            reject_secrets(item)
    elif isinstance(value, list):
        for item in value:
            reject_secrets(item)


class Engine:
    """No engine discovery, namespace fallback, or model construction."""

    Config = Config
    catalog = staticmethod(algorithm_catalog)

    def __init__(self, project):
        self.project = Path(project).resolve()
        self.root = Path(draftfit.__file__).resolve().parents[1]
        self.python = os.path.abspath(
            os.path.expanduser(os.getenv("DRAFTFIT_TRAINER_PYTHON", sys.executable))
        )

    def info(self):
        return {
            "root": str(self.root),
            "package": "draftfit",
            "python": self.python,
            "actions": [
                "plan",
                "train",
                "validate_tokens",
                "unattended_plan",
                "unattended",
                "unattended_resume",
            ],
            "support_scope": "Registered engine actions; GPU and serving validation are separate.",
        }

    def validate(self, raw):
        reject_secrets(raw)
        cfg = Config.model_validate(raw)
        resolve_run(cfg)
        return {
            "valid": True,
            "config": cfg.model_dump(mode="json"),
            "gpu_verified": False,
            "serving_verified": False,
        }

    def file(self, value, *, required=True):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("An explicit project path is required.")
        path = (self.project / Path(value).expanduser()).resolve()
        if not path.is_relative_to(self.project):
            raise ValueError("The path must be inside the selected project.")
        if required and not path.is_file():
            raise FileNotFoundError(path)
        return path

    def command(self, *args):
        if not Path(self.python).is_file():
            raise ValueError(
                "DRAFTFIT_TRAINER_PYTHON must name an existing Python interpreter."
            )
        return [self.python, "-m", "draftfit.cli", *map(str, args)]
