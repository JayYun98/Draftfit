"""Evaluate actual comparison exports with common target-verified generation.

This deliberately slow full-prefix reference decoder checks greedy token
equality and proposal acceptance. It does not implement the external EAGLE-4
tree/cost policy and cannot establish optimized engine throughput.
"""

import argparse
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from draftfit.algorithms.common.dflash_family_model import OnlineDSparkModel, OnlineDFlash2Model
from draftfit.modeling.auto import AutoDraftModel
from draftfit.offline_capture.transformers import OfflineTransformersCapture
from draftfit.training.joint_verification import _generate
from train_eagle4_comparison import LAYERS, NAMES, digest, memory_guard, write_json


def eagle4_propose(bundle, features, anchor, count):
    """Greedy chain using the candidate head and recurrent normalized features.

    Recompute the single layer over the committed context and appended draft
    ancestors. The committed window stays fixed during this proposal round.
    This avoids depending on engine-specific tree/cache implementations.
    """
    draft = bundle.draft_model
    hidden = draft.project_hidden_states(features.hidden_states)
    ids = torch.cat((features.input_ids[:, 1:], anchor[:, None]), dim=1)
    committed = ids.shape[1]
    selected = []
    for _ in range(count):
        length = ids.shape[1]
        positions = torch.arange(length, device=ids.device)
        allowed = positions[None, :] <= positions[:, None]
        if draft.sliding_window is not None:
            # Ancestors append to the last committed query's window.
            lower = positions.clamp_max(committed - 1) - draft.sliding_window + 1
            allowed &= positions[None, :] >= lower[:, None]
        mask = torch.zeros((1, 1, length, length), device=hidden.device, dtype=hidden.dtype)
        mask.masked_fill_(~allowed[None, None], torch.finfo(hidden.dtype).min)
        output = draft.backbone(
            input_embeds=draft.embed_input_ids(ids), hidden_states=hidden,
            cache_hidden=None, attention_mask=mask, position_ids=positions[None],
        )[:, -1:]
        scores, candidates = draft.candidate_logits(output)
        token = candidates.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
        selected.append(token)
        hidden = torch.cat((hidden, output), dim=1)
        ids = torch.cat((ids, token), dim=1)
    return torch.cat(selected, dim=1)


@torch.inference_mode()
def evaluate(args):
    torch.set_num_threads(4)
    memory_guard(args)
    metadata = json.loads((args.output / "capture.json").read_text())
    rows = sorted((args.output / "test_sft").glob("*.pt"))[:args.samples]
    if len(rows) != args.samples:
        raise ValueError("Not enough held-out captured samples")
    for path, identity in zip(rows, metadata["splits"]["test_sft"]):
        if digest(path) != identity["sha256"]:
            raise ValueError("Held-out capture identity changed")
    prompts = [torch.load(path, weights_only=True)["prompt_ids"][0] for path in rows]
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).eval().requires_grad_(False).cuda()
    teacher = OfflineTransformersCapture(target)
    teacher.set_capture_layers(LAYERS, capture_method="eagle3")
    # The reference verifier always uses the frozen TARGET head, never a draft head.
    projection = SimpleNamespace(parameters=target.parameters,
                                 _compute_logits=target.lm_head)
    baseline_bundle = SimpleNamespace(model=projection)
    eos = tokenizer.eos_token_id
    stops = [] if eos is None else eos if isinstance(eos, list) else [eos]
    config = dict(stop_token_ids=stops, max_new_tokens=args.max_new_tokens,
                  draft_tokens=6, deadline=None)
    destination = args.output / args.result_dir
    destination.mkdir(exist_ok=True)
    baseline = []
    for index, prompt in enumerate(prompts):
        result = _generate(teacher, baseline_bundle, prompt, config, speculative=False)
        baseline.append(result)
        print(json.dumps({"algorithm":"target", "sample":index,
                          "tokens":result["generated_tokens"]}), flush=True)
    write_json(destination / "target.json", baseline)
    summaries = {}
    for name in NAMES:
        export = args.output / name / "export"
        draft = AutoDraftModel.from_pretrained(export, torch_dtype=torch.bfloat16).cuda().eval()
        if name == "eagle4":
            bundle = SimpleNamespace(draft_model=draft, model=projection)
        else:
            cls = OnlineDSparkModel if name == "dspark" else OnlineDFlash2Model
            model = cls(draft_model=draft, target_lm_head=target.lm_head,
                        target_embed_tokens=target.get_input_embeddings(),
                        mask_token_id=draft.mask_token_id, block_size=draft.block_size,
                        attention_backend="sdpa")
            bundle = SimpleNamespace(draft_model=draft, model=model)
        records = []
        for index, prompt in enumerate(prompts):
            result = _generate(teacher, bundle, prompt, config, speculative=True,
                               proposal_fn=eagle4_propose if name == "eagle4" else None)
            result["greedy_equal"] = result["token_ids"] == baseline[index]["token_ids"]
            records.append(result)
            print(json.dumps({"algorithm":name, "sample":index,
                  "accepted":result["accepted_tokens"], "proposed":result["proposed_tokens"],
                  "greedy_equal":result["greedy_equal"]}), flush=True)
        write_json(destination / f"{name}.json", records)
        accepted = sum(r["accepted_tokens"] for r in records)
        proposed = sum(r["proposed_tokens"] for r in records)
        tokens = sum(r["generated_tokens"] for r in records)
        rounds = sum(r["rounds"] for r in records)
        seconds = sum(r["seconds"] for r in records)
        summaries[name] = dict(
            samples=len(records), greedy_equal_samples=sum(r["greedy_equal"] for r in records),
            accepted_tokens=accepted, proposed_tokens=proposed,
            acceptance_rate=accepted / proposed if proposed else 0,
            committed_tokens_per_round=tokens / rounds, generated_tokens=tokens,
            rounds=rounds, reference_seconds=seconds, reference_tok_s=tokens / seconds,
            export_sha256={p.name:digest(p) for p in export.glob("*.safetensors")},
        )
        print(json.dumps(summaries[name]), flush=True)
        del bundle, draft
        if name != "eagle4":
            del model
        gc.collect(); torch.cuda.empty_cache()
    write_json(destination / "summary.json", dict(
        decoder="full-prefix greedy chain reference; no optimized engine/tree policy",
        max_new_tokens=args.max_new_tokens, draft_tokens=6, arms=summaries,
        target_reference_seconds=sum(r["seconds"] for r in baseline),
        heldout_capture_sha256=[digest(p) for p in rows],
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
    ))
    if any(r["greedy_equal_samples"] != len(prompts) for r in summaries.values()):
        raise RuntimeError("At least one exported arm diverged from target greedy tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--memory-mib", type=int, default=5600)
    parser.add_argument("--result-dir", default="evaluation")
    evaluate(parser.parse_args())
