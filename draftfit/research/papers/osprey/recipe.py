"""Convert the draft and publish it to a fresh output directory."""

import json
import shutil
import tempfile
from pathlib import Path


def run_osprey(config):
    from .method import convert

    m = config.model
    destination = Path(config.output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".osprey-", dir=destination.parent))
    try:
        converted = scratch / "draft"
        result = convert(
            m.source_checkpoint,
            str(converted),
            m.num_layers,
            m.target_hidden_size,
            m.target_layer_ids,
        )
        if m.target_tokenizer:
            from .tokenizer_alignment import align

            result["alignment"] = align(
                converted,
                m.target_tokenizer,
                converted,
                target_revision=m.target_revision,
            )
        (converted / "adaptation.json").write_text(json.dumps(result, indent=2) + "\n")
        converted.rename(destination)
        return result
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
