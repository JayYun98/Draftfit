"""Prepare fixed-revision Humanizer family configs; does not rent or train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from draftfit.campaign import digest
from draftfit.data.humanizer import MODEL_ID, REVISION
from draftfit.family import build_family_plan
from draftfit.training.joint_family import validate_joint_family

SEED_ID = "z-lab/gemma4-12B-it-DFlash"
SEED_REVISION = "541d351de6dc9cd698fc348612bd5cdf03c636d0"
SEED_SHA256 = "8b32507dcff3ca4c1bbbb6c1cca7d17749bbf394480024e505228b6ed40afa02"


def prepare(
    output,
    seed,
    train_features=None,
    validation_features=None,
    *,
    max_length=512,
    train_tokens=None,
    validation_tokens=None,
):
    output, seed = Path(output).resolve(), Path(seed).resolve()
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    if type(max_length) is not int or max_length < 32:
        raise ValueError("max_length must be an integer >=32")
    # Require an immutable Hub snapshot; the training API has no draft_revision.
    if (
        seed.name != SEED_REVISION
        or seed.parent.name != "snapshots"
        or seed.parent.parent.name != "models--z-lab--gemma4-12B-it-DFlash"
    ):
        raise ValueError("seed must be the pinned Gemma DFlash Hub snapshot directory")
    for name in ("config.json", "model.safetensors"):
        if not (seed / name).is_file():
            raise FileNotFoundError(f"missing seed file: {name}")
    if digest(seed / "model.safetensors") != SEED_SHA256:
        raise ValueError("seed weight SHA256 differs from pinned Hub LFS object")
    online = train_tokens is not None or validation_tokens is not None
    if online:
        if (
            not train_tokens
            or not validation_tokens
            or train_features
            or validation_features
        ):
            raise ValueError(
                "choose tokenized train/validation files OR offline feature directories"
            )
        train_features, validation_features = train_tokens, validation_tokens
    elif not train_features or not validation_features:
        raise ValueError("provide both train and validation sources")
    train, validation = (
        Path(train_features).resolve(),
        Path(validation_features).resolve(),
    )
    if (
        train == validation
        or train in validation.parents
        or validation in train.parents
    ):
        raise ValueError("train and validation feature directories must be disjoint")
    # Data/capture may be prepared later. Missing features are not marked verified.
    base = dict(
        model=dict(
            target_model_path=MODEL_ID,
            target_revision=REVISION,
            draft_model_config=str(seed / "config.json"),
            draft_checkpoint_path=str(seed),
            target_backend="transformers",
            embedding_key="model.language_model.embed_tokens.weight",
            mask_token_id=4,
            torch_dtype="bfloat16",
        ),
        data=dict(
            hidden_states_path=str(train),
            eval_hidden_states_path=str(validation),
            max_length=max_length,
            dataloader_num_workers=0,
        ),
        training=dict(
            strategy="dflash",
            num_epochs=32,
            max_steps=30000,
            total_steps=30000,
            batch_size=1,
            accumulation_steps=4,
            num_anchors=8,
            learning_rate=2e-5,
            warmup_ratio=0.01,
            save_interval=250,
            eval_interval=1000,
            log_interval=10,
            max_checkpoints=2,
            fsdp_sharding="NO_SHARD",
            attention_backend="sdpa",
        ),
    )
    if online:
        base["data"].pop("hidden_states_path")
        base["data"].pop("eval_hidden_states_path")
        base["data"].update(prompts_path=str(train), eval_data_path=str(validation))
        base["training"]["joint_online_capture"] = True
    manifest = dict(
        base_config="base.json",
        output_dir="joint",
        runs=[
            dict(name="dflash", strategy="dflash"),
            dict(name="dspark", strategy="dspark"),
            dict(name="dflash2", strategy="dflash2"),
        ],
    )
    output.mkdir(parents=True)
    for name, value in [("base.json", base), ("family.json", manifest)]:
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    try:
        _, stages = build_family_plan(str(output / "family.json"))
        validate_joint_family(stages)
    except Exception:
        # Keep diagnostic inputs, but never create a ready receipt on failure.
        (output / "FAILED").write_text("config preflight failed; see command error\n")
        raise
    next(
        s.draft_config for s in stages if s.config.training.strategy == "dspark"
    ).to_json_file(str(output / "capture-dspark.json"))
    receipt = dict(
        status="config_prepared",
        target_revision=REVISION,
        seed_revision=SEED_REVISION,
        families=[s.config.training.strategy for s in stages],
        max_steps=30000,
        seed_weights_sha256=SEED_SHA256,
        trained=False,
        rented=False,
        authorized_total_usd_exclusive=3,
        fail_fast_required=True,
        remaining=[
            "data-and-capture",
            "GPU-smoke-and-budget-watchdog",
            "actual-MLX-export-validation",
        ],
        feature_mode="online" if online else "offline",
        persisted_feature_bytes=0 if online else None,
        approximate_feature_window_bytes=(
            4 * max_length * 3840 * 7 * 2 if online else None
        ),
    )
    (output / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--train-features")
    parser.add_argument("--train-tokens")
    parser.add_argument("--validation-features")
    parser.add_argument("--validation-tokens")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.output,
                args.seed,
                args.train_features,
                args.validation_features,
                max_length=args.max_length,
                train_tokens=args.train_tokens,
                validation_tokens=args.validation_tokens,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
