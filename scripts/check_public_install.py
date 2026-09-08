"""CPU/offline onboarding against an INSTALLED wheel, never the source checkout.

Run with the provisioned environment: python -I scripts/check_public_install.py.
This checks metadata/configuration, not real target training or GPU support.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    import specforge

    if not sys.flags.isolated:
        raise RuntimeError("run with python -I to exclude the source checkout")
    installed = Path(specforge.__file__).resolve()
    if "site-packages" not in installed.parts:
        raise RuntimeError(f"expected installed wheel, not checkout: {installed}")
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
                   "HF_DATASETS_OFFLINE": "1", "WANDB_MODE": "disabled"}
    with tempfile.TemporaryDirectory(prefix="specforge-onboarding-") as temporary:
        root = Path(temporary)

        def cli(*args, error=None):
            result = subprocess.run(
                [sys.executable, "-I", "-m", "specforge.cli", *map(str, args)],
                cwd=root, env=environment, capture_output=True, text=True, timeout=60,
            )
            if error is not None:
                assert result.returncode != 0 and error in result.stderr, result.stderr
            else:
                assert result.returncode == 0, result.stdout + result.stderr
            return result.stdout

        target = root / "target"
        target.mkdir()
        (target / "config.json").write_text(json.dumps({
            "model_type": "llama", "architectures": ["LlamaForCausalLM"],
            "hidden_size": 32, "vocab_size": 64, "num_hidden_layers": 8,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "intermediate_size": 64, "max_position_embeddings": 2048,
        }))
        catalog = json.loads(cli("algorithms"))
        assert {item["algorithm"] for item in catalog} == {"dspark", "dflash", "eagle3", "peagle", "domino"}
        cli("target", "inspect", target, "--local-only")
        for strategy in ("dspark", "dflash"):
            project = root / strategy
            args = ("target", "prepare", target, "--local-only", "--strategy", strategy,
                    "--hidden-states", root / "features", "--output-dir", project,
                    "--set", "model.draft_num_hidden_layers=2")
            cli(*args)
            config = project / "train.json"
            before = config.read_bytes()
            manifest = json.loads((project / "manifest.json").read_text())
            assert manifest["validation"]["configuration"] == "passed"
            assert all(manifest["validation"][key] == "not_run"
                       for key in ("capture", "training", "export", "serving"))
            assert json.loads((project / "draft.json").read_text())["num_hidden_layers"] == 2
            cli("train", "-c", config, "--plan")
            cli(*args, error="refusing to overwrite")
            assert config.read_bytes() == before
            cli("train", "-c", config, "--plan", "training.typo=1", error="typo")
            cli("train", "-c", config, "--plan", "model.draft_checkpoint_path=/weights",
                "training.resume_from=/checkpoint", error="mutually exclusive")
        cli("target", "prepare", target, "--local-only", "--strategy", "unimplemented",
            "--hidden-states", root / "features", "--output-dir", root / "invalid",
            error="unimplemented")
        assert not (root / "invalid").exists()
    print(json.dumps({"passed": True, "scope": "installed-wheel CPU metadata onboarding",
                      "algorithms_prepared": ["dspark", "dflash"], "gpu_started": False}))


if __name__ == "__main__":
    main()
