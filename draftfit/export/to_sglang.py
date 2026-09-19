# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Export a DataFlow training checkpoint to an SGLang spec-decoder draft directory.

Weight-name compatibility is the silent-failure risk here: a key the serving
loader does not expect is skipped (or zero-filled) without an error. The
per-architecture trainer-key -> serving-key map therefore lives in
``WEIGHT_MAPS`` below, and this exporter validates the produced state against it.

For ``LlamaForCausalLMEagle3`` the map is the identity: SpecForge's draft module
names (``midlayer.*`` / ``fc`` / ``norm`` / ``lm_head`` + the ``t2d``/``d2t``
buffers) are exactly what SGLang's EAGLE3 draft loader reads.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional

from draftfit.export.checkpoint_io import (
    apply_legacy_rope_scaling,
    materialize_draft,
    resolve_training_state,
)

#: per-architecture trainer-key -> serving-key renames ({} = identity).
WEIGHT_MAPS: Dict[str, Dict[str, str]] = {
    "LlamaForCausalLMEagle3": {},
}

#: keys the sglang EAGLE3 spec-decoder loader requires in a draft checkpoint.
_REQUIRED_SERVING_KEYS = ("fc.weight", "norm.weight", "lm_head.weight", "t2d", "d2t")
_REQUIRED_DFLASH2_SERVING_KEYS = (
    "fc.weight",
    "hidden_norm.weight",
    "norm.weight",
    "candidate_selector.predecessor_codebook",
    "candidate_selector.successor_codebook",
    "candidate_selector.hidden_projection.weight",
)


def _serving_state(
    state_dict,
    weight_map: Dict[str, str],
    required_keys=_REQUIRED_SERVING_KEYS,
):
    out = {weight_map.get(k, k): v for k, v in state_dict.items()}
    bad_prefix = [k for k in out if k.startswith("draft_model.")]
    if bad_prefix:
        raise ValueError(
            f"serving state still carries trainer prefixes: {sorted(bad_prefix)}"
        )
    missing = [k for k in required_keys if weight_map.get(k, k) not in out]
    if missing:
        raise ValueError(
            f"serving state is missing required keys {missing}; the sglang loader "
            f"would silently produce a broken draft. Present: {sorted(out)[:8]}..."
        )
    return out


def export_to_sglang(
    checkpoint_path: str,
    draft_config_path: str,
    output_dir: str,
    *,
    vocab_mapping_path: Optional[str] = None,
    weight_map: Optional[Dict[str, str]] = None,
) -> str:
    """Write an sglang-loadable draft directory; returns ``output_dir``.

    ``weight_map`` overrides the per-architecture entry in :data:`WEIGHT_MAPS`.
    ``vocab_mapping_path`` refreshes the ``t2d``/``d2t`` buffers when the
    checkpoint predates them.
    """
    state = resolve_training_state(checkpoint_path)
    strategy = state.get("strategy")
    if strategy not in {"eagle3", "dflash2"}:
        raise ValueError(
            "the specialized SGLang exporter supports EAGLE3 and DFlash2 "
            f"checkpoints only, got strategy={strategy!r}; use --to hf for "
            "other draft model directories"
        )
    model = materialize_draft(
        state, draft_config_path, vocab_mapping_path=vocab_mapping_path
    )
    model_name = type(model).__name__
    if strategy == "dflash2" and model_name != "DFlash2DraftModel":
        raise ValueError(
            "dflash2 checkpoint does not contain a DFlash2DraftModel; "
            f"draft config resolved to {model_name}"
        )
    if strategy == "eagle3" and model_name == "DFlash2DraftModel":
        raise ValueError("eagle3 checkpoint cannot use a DFlash2DraftModel config")
    if model_name == "DFlash2DraftModel":
        if vocab_mapping_path:
            raise ValueError("DFlash2 serving does not support vocabulary pruning")
        from draftfit.modeling.draft.dflash2 import (
            dflash2_config_for_serving,
            dflash2_state_dict_for_serving,
        )

        config = dflash2_config_for_serving(model.config.to_dict())
        full = dflash2_state_dict_for_serving(model.state_dict())
        model.save_pretrained(
            output_dir,
            state_dict=_serving_state(
                full,
                {},
                required_keys=_REQUIRED_DFLASH2_SERVING_KEYS,
            ),
        )
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
            handle.write("\n")
        apply_legacy_rope_scaling(output_dir)
        return output_dir
    if weight_map is None:
        weight_map = WEIGHT_MAPS.get(type(model).__name__, {})
    # the model's state dict includes any refreshed t2d/d2t buffers; drop the
    # embeddings exactly as the trainer-side checkpoint filter does.
    full = {k: v for k, v in model.state_dict().items() if "embed" not in k.lower()}
    model.save_pretrained(output_dir, state_dict=_serving_state(full, weight_map))
    apply_legacy_rope_scaling(output_dir)
    return output_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=export_to_sglang.__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--draft-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-mapping", default=None)
    args = parser.parse_args(argv)
    out = export_to_sglang(
        args.checkpoint,
        args.draft_config,
        args.output_dir,
        vocab_mapping_path=args.vocab_mapping,
    )
    print(f"exported sglang draft to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
