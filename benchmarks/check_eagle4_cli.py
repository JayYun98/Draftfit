"""Opt-in CUDA gate for actual EAGLE4 train/resume/export CLI entry points.

Uses a tiny random target to test lifecycle wiring. Pretrained MiniCPM training
and serving quality require the separate comparison workload.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output.resolve()
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from draftfit.algorithms.builtin import builtin_algorithm_registry
    from draftfit.modeling.auto import AutoDraftModel
    from draftfit.offline_capture.transformers import OfflineTransformersCapture

    if not torch.cuda.is_available():
        raise RuntimeError("This gate requires CUDA")
    torch.set_num_threads(4)
    torch.manual_seed(29)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
    )
    target_path, features = root / "target", root / "features"
    features.mkdir()
    target = LlamaForCausalLM(config).cuda().bfloat16().eval().requires_grad_(False)
    target.save_pretrained(target_path, max_shard_size="2KB")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({str(i): i for i in range(32)}, unk_token="0")
        ),
        unk_token="0",
        bos_token="1",
        eos_token="2",
        pad_token="3",
    )
    tokenizer.save_pretrained(target_path)
    capture = OfflineTransformersCapture(target)
    capture.set_capture_layers([0, 1, 2])
    layout = (
        builtin_algorithm_registry()
        .resolve("eagle4")
        .providers.offline_for("text")
        .capture_layout
    )
    for index, length in enumerate([8, 11, 9, 10]):
        ids = torch.arange(1, length + 1, device="cuda").reshape(1, -1)
        values = capture.capture(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            loss_mask=torch.ones_like(ids),
        )
        row = layout.materialize(
            dict(
                input_ids=ids[0],
                loss_mask=values.loss_mask[0],
                aux_hidden_states=values.hidden_states,
                last_hidden_states=values.last_hidden_states,
            )
        )
        torch.save(row, features / f"{index}.ckpt")
    del capture, target
    torch.cuda.empty_cache()
    draft = LlamaConfig(
        **{
            **config.to_dict(),
            "architectures": ["LlamaForCausalLMEagle4"],
            "num_hidden_layers": 1,
            "layer_types": ["full_attention"],
            "draft_vocab_size": 32,
            "fc_norm": True,
            "norm_output": True,
            "eagle_config": {"eagle_aux_hidden_state_layer_ids": [0, 1, 2]},
            "efficient_lm_head_config": {
                "type": "low_rank",
                "rank": 8,
                "num_candidates": 8,
            },
        }
    )
    draft.to_json_file(root / "draft.json")
    env = {
        **os.environ,
        "SPECFORGE_DEVICE": "cuda",
        "DRAFTFIT_CUDA_DIST_BACKEND": "gloo",
        "OMP_NUM_THREADS": "4",
    }

    def cli(name, *arguments):
        command = [sys.executable, "-m", "draftfit.cli", *map(str, arguments)]
        with (root / f"{name}.log").open("w") as log:
            result = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise RuntimeError(
                f"{name} failed with exit {result.returncode}; see {root/name}.log"
            )

    def train(name, stage, steps, *, checkpoint=None, resume=None):
        output = root / name
        run = dict(
            model=dict(
                target_model_path=str(target_path),
                draft_model_config=str(root / "draft.json"),
                torch_dtype="bfloat16",
                draft_checkpoint_path=checkpoint,
            ),
            data=dict(hidden_states_path=str(features), max_length=16),
            training=dict(
                strategy="eagle4",
                attention_backend="sdpa",
                ttt_length=2,
                max_steps=steps,
                total_steps=4,
                num_epochs=4,
                batch_size=1,
                save_interval=1,
                fsdp_sharding="NO_SHARD",
                eagle4_stage=stage,
                eagle4_initialize_candidates=stage == "candidates" and resume is None,
                optimizer_cpu_offload=True,
                resume_from=resume,
                seed=29,
            ),
            run_id=name,
            output_dir=str(output),
            deployment=dict(mode="local_colocated", trainer=dict(nproc_per_node=1)),
        )
        path = root / f"{name}.json"
        path.write_text(json.dumps(run, indent=2) + "\n")
        cli(name, "train", "-c", path)
        latest = output / f"{name}-latest"
        state = torch.load(
            latest / "training_state.pt", map_location="cpu", weights_only=True
        )
        return latest, state

    dense, dense_state = train("dense", "draft", 2)
    cli(
        "dense-export",
        "export",
        "--checkpoint",
        dense,
        "--draft-config",
        root / "draft.json",
        "--to",
        "hf",
        "--output-dir",
        root / "dense-export",
        "--embedding-source",
        target_path,
    )
    _, reference = train(
        "candidate-reference", "candidates", 4, checkpoint=str(root / "dense-export")
    )
    partial, _ = train(
        "candidate-partial", "candidates", 2, checkpoint=str(root / "dense-export")
    )
    resumed, resumed_state = train(
        "candidate-resumed", "candidates", 4, resume=str(partial.resolve())
    )
    expected = reference["draft_state_dict"]
    actual = resumed_state["draft_state_dict"]
    equal = expected.keys() == actual.keys() and all(
        torch.equal(v, actual[k]) for k, v in expected.items()
    )
    frozen = all(
        torch.equal(v, actual[k])
        for k, v in dense_state["draft_state_dict"].items()
        if not k.startswith("efficient_lm_head")
    )
    if not equal or not frozen:
        raise RuntimeError(f"resume_equal={equal}, frozen_dense_equal={frozen}")
    cli(
        "candidate-export",
        "export",
        "--checkpoint",
        resumed,
        "--draft-config",
        root / "draft.json",
        "--to",
        "hf",
        "--output-dir",
        root / "candidate-export",
        "--embedding-source",
        target_path,
    )
    reloaded = AutoDraftModel.from_pretrained(root / "candidate-export")
    reload_equal = all(
        torch.equal(v, reloaded.state_dict()[k]) for k, v in actual.items()
    )
    if not reload_equal:
        raise RuntimeError("CLI export weights changed")
    receipt = dict(
        scope="tiny random CUDA target; production train/resume/export CLI",
        dense_steps=2,
        candidate_steps=4,
        resumed_from_step=2,
        resume_weights_equal=equal,
        candidate_stage_froze_dense=frozen,
        export_reload_equal=reload_equal,
    )
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
