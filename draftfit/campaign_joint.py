"""Native fixed-horizon joint worker for the campaign subprocess protocol."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from draftfit.campaign import digest, read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stop-at", type=int, required=True)
    parser.add_argument("--resume-job")
    args = parser.parse_args()
    from draftfit.export.to_hf import export_to_hf
    from draftfit.family import build_family_plan
    from draftfit.training.joint_family import run_joint_family, validate_joint_family

    folder = Path(os.environ["DRAFTFIT_JOB_OUTPUT"])
    manifest = Path(args.manifest).resolve()
    _, stages = build_family_plan(str(manifest))
    validate_joint_family(stages)
    source = dict(
        manifest_sha256=digest(manifest),
        base_sha256=digest(manifest.parent / read(manifest)["base_config"]),
    )
    resume = None
    if args.resume_job:
        receipt = read(read_inputs()[args.resume_job])
        if receipt.get("source") != source:
            raise ValueError("family manifest/base changed between milestones")
        resume = receipt["checkpoint"]
        if digest(resume) != receipt["checkpoint_sha256"]:
            raise ValueError("resume checkpoint hash mismatch")
    # Checkpoints are atomically written by the trainer. Recover an interrupted
    # attempt without resetting the optimizer/schedule or overwriting its logs.
    previous = list(folder.glob("attempt-*/joint-step*.pt"))
    if previous:
        resume = max(
            previous,
            key=lambda p: (int(p.stem.removeprefix("joint-step")), p.stat().st_mtime),
        )
    run_dir = folder / f"attempt-{os.environ.get('DRAFTFIT_JOB_ATTEMPT', '1')}"
    run_dir.mkdir(exist_ok=False)
    adjusted = []
    for stage in stages:
        cfg = stage.config.model_copy(deep=True)
        cfg.output_dir = str(run_dir / stage.name)
        Path(cfg.output_dir).mkdir()
        cfg.model.draft_model_config = str(Path(cfg.output_dir) / "draft.json")
        stage.draft_config.to_json_file(cfg.model.draft_model_config)
        write(
            Path(cfg.output_dir) / "resolved_config.json", cfg.model_dump(mode="json")
        )
        adjusted.append(replace(stage, config=cfg))
    result = run_joint_family(
        run_dir, adjusted, resume_from=resume, stop_at_step=args.stop_at
    )
    checkpoint = run_dir / f"joint-step{args.stop_at}.pt"
    arms = []
    for stage, row in zip(adjusted, result["runs"]):
        latest = Path(row["latest"]).resolve()
        exported = run_dir / "exports" / stage.name
        export_to_hf(
            str(latest),
            str(latest / "config.json"),
            str(exported),
            embedding_source=stage.config.model.target_model_path,
        )
        arms.append(
            dict(
                name=stage.name,
                strategy=stage.config.training.strategy,
                checkpoint=str(latest),
                export=str(exported),
                weights_sha256=digest(exported / "model.safetensors"),
            )
        )
    write(
        os.environ["DRAFTFIT_JOB_RESULT"],
        dict(
            status="complete",
            schema="draftfit-campaign-joint-v1",
            source=source,
            step=args.stop_at,
            checkpoint=str(checkpoint),
            checkpoint_sha256=digest(checkpoint),
            monitor_dir=str(run_dir),
            arms=arms,
        ),
    )


def read_inputs():
    return json.loads(os.environ.get("DRAFTFIT_INPUT_RESULTS", "{}"))


if __name__ == "__main__":
    main()
