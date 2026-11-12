import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    import draftfit

    if not sys.flags.isolated:
        raise RuntimeError("run with python -I to exclude the source checkout")
    installed = Path(draftfit.__file__).resolve()
    distribution = importlib.metadata.distribution("draftfit")
    commands = {
        entry.name: entry.value
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
    }
    assert commands == {"draftfit": "draftfit.cli:main"}
    if "site-packages" not in installed.parts:
        raise RuntimeError(f"expected installed wheel, not checkout: {installed}")
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "WANDB_MODE": "disabled",
    }
    with tempfile.TemporaryDirectory(prefix="draftfit-onboarding-") as temporary:
        root = Path(temporary)

        def cli(*args, error=None):
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-m",
                    "draftfit.cli",
                    *map(str, args),
                ],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if error is not None:
                assert result.returncode != 0 and error in result.stderr, result.stderr
            else:
                assert result.returncode == 0, result.stdout + result.stderr
            return result.stdout

        result = subprocess.run(
            [sys.executable, "-I", "-m", "draftfit.training.search", "--help"],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        from draftfit.assets import export_assets, list_assets

        assert "docs/TRAINING_SEARCH.md" in list_assets()
        exported = export_assets(root / "exported-assets")
        assert (exported / "docs/TRAINING_SEARCH.md").is_file()
        for name in (
            "docs/research/dforktree-scoring.md",
            "docs/specs/dforktree-scoring.md",
            "docs/research/results/dforktree-scoring-20261112.json",
        ):
            assert (exported / name).is_file(), name

        def research(*args, error=None):
            result = subprocess.run(
                [sys.executable, "-I", "-m", "draftfit.research", *map(str, args)],
                cwd=exported,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if error is None:
                assert result.returncode == 0, result.stdout + result.stderr
                return json.loads(result.stdout)
            assert result.returncode != 0 and error in result.stderr, result.stderr

        recipe = "examples/configs/research/dforktree-scoring.yaml"
        output = exported / "outputs/research-dforktree-scoring"
        planned = research("-c", recipe, "--plan", "data.input_path=missing.json")
        assert planned["execution"] == "not-run"
        assert planned["implementation"] == "cpu_probability_scoring_reference"
        assert not output.exists()
        scored = research("-c", recipe)
        expected = (
            ("first", 0.0, [0.5, 0.3], 0.2),
            (
                "spine",
                0.3,
                [0.41633919763052696, 0.37863133148174183],
                0.20502947088773132,
            ),
            (
                "fork",
                0.6,
                [0.33503928046719367, 0.4618308843984667],
                0.20312983513433963,
            ),
        )
        assert scored["processed_count"] == len(scored["answers"]) == 3
        for answer, (identifier, weight, probabilities, tail) in zip(
            scored["answers"], expected
        ):
            assert answer["id"] == identifier
            assert answer["candidate_ids"] == [101, 102]
            assert answer["weight"] == weight
            assert len(answer["candidate_probs"]) == 2
            for actual, value in zip(answer["candidate_probs"], probabilities):
                assert math.isclose(actual, value, rel_tol=0, abs_tol=1e-12)
            assert math.isclose(
                answer["tail_probability"], tail, rel_tol=0, abs_tol=1e-12
            )
            assert math.isclose(
                math.fsum(answer["candidate_probs"]) + answer["tail_probability"],
                1,
                rel_tol=0,
                abs_tol=1e-15,
            )
            assert math.isclose(answer["clipped_pmi"][0], -math.log(2), abs_tol=1e-15)
            assert math.isclose(answer["clipped_pmi"][1], math.log(2), abs_tol=1e-15)
        assert scored["implementation_scope"] == "cpu_probability_scoring_reference"
        assert scored["input_kind"] == "synthetic"
        assert scored["input_digest_scope"] == "full_file"
        fixture = exported / "examples/configs/research/data/dforktree-scoring.json"
        assert (
            scored["input_sha256"] == hashlib.sha256(fixture.read_bytes()).hexdigest()
        )
        assert scored["tail_convention"] == "true_zero_no_floor"
        for field in (
            "ngram_counting",
            "corpus_smoothing",
            "tree_construction",
            "verifier",
            "serving_integration",
            "training",
        ):
            assert scored[field] == "not_implemented"
        assert scored["production_engine_validated"] is False
        assert scored["throughput_claim"] is False
        source = scored["source"]
        assert source["paper"] == "https://arxiv.org/html/2611.12356v1"
        assert source["url"] == "https://github.com/Perseus037/DForkTree"
        assert source["revision"] == "eeed8b698b8822aa2767ec5c191efc4e1b6cdbe3"
        assert source["license"] == "Apache-2.0"
        assert source["sha256"] == {
            "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
            "NOTICE": "db461002651716701a8183a54f7b60b6ec570e8dff0e18f564988e9ae83e91f4",
            "dforktree/ngram_gpu.py": (
                "97137b4692fbe28cb7bd77e3c711a8ed9a4c0919e7045ad9255c9cfa56a428fd"
            ),
            "dforktree/fc1_builder.py": (
                "7a7e261010f332bf7190d7b636c8b8cf582a7d87332f48ec564210e52d99b7a2"
            ),
            "dforktree/tree_ops.py": (
                "5ba3417fcfe251b45859ebc691d7e74f618bb67ebed41d218c1dfbe44c9647c5"
            ),
        }
        from importlib.resources import files

        vendor = files("draftfit.research._vendor")
        for original, packaged in (
            ("LICENSE", "DFORKTREE_LICENSE"),
            ("NOTICE", "DFORKTREE_NOTICE"),
        ):
            digest = hashlib.sha256(vendor.joinpath(packaged).read_bytes()).hexdigest()
            assert digest == source["sha256"][original]
        persisted = output / "result.json"
        result_bytes = persisted.read_bytes()
        assert json.loads(result_bytes) == scored
        resolved = json.loads((output / "recipe.json").read_text())
        assert resolved == research("-c", recipe, "--plan")["config"]
        research("-c", recipe, error="output already exists")
        assert persisted.read_bytes() == result_bytes
        research(
            "-c",
            recipe,
            "data.input_path=missing.json",
            "output_dir=failed/result",
            error="missing.json",
        )
        assert not (exported / "failed").exists()

        target = root / "target"
        target.mkdir()
        raw_data = root / "messages.jsonl"
        raw_data.write_text(
            "\n".join(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": f"Question {index}"},
                            {"role": "assistant", "content": f"Answer {index}"},
                        ]
                    }
                )
                for index in range(20)
            )
            + "\n"
        )
        train_data, holdout_data = root / "train.jsonl", root / "holdout.jsonl"
        cli("data", "prepare", "--help")
        data_args = (
            "data",
            "prepare",
            "--input",
            raw_data,
            "--output",
            train_data,
            "--split-eval",
            "--eval-output",
            holdout_data,
        )
        cli(*data_args)
        assert len(train_data.read_text().splitlines()) == 19
        assert len(holdout_data.read_text().splitlines()) == 1
        assert "conversations" in json.loads(train_data.read_text().splitlines()[0])
        cli(*data_args, error="refusing to overwrite")
        cli("benchmark", "--help")
        (target / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "llama",
                    "architectures": ["LlamaForCausalLM"],
                    "hidden_size": 32,
                    "vocab_size": 64,
                    "num_hidden_layers": 8,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 8,
                    "intermediate_size": 64,
                    "max_position_embeddings": 2048,
                }
            )
        )
        catalog = json.loads(cli("algorithms"))
        assert {item["algorithm"] for item in catalog} == {
            "dspark",
            "dflash",
            "dflash2",
            "eagle3",
            "eagle4",
            "peagle",
            "domino",
            "alr_ira",
            "osprey",
        }
        cli("target", "inspect", target, "--local-only")
        for strategy in ("dspark", "dflash", "dflash2", "eagle4"):
            project = root / strategy
            layers = 1 if strategy == "eagle4" else 2
            args = (
                "target",
                "prepare",
                target,
                "--local-only",
                "--strategy",
                strategy,
                "--hidden-states",
                root / "features",
                "--output-dir",
                project,
                "--set",
                f"model.draft_num_hidden_layers={layers}",
            )
            if strategy == "eagle4":
                args += ("--set", "training.attention_backend=sdpa")
            cli(*args)
            config = project / "train.json"
            before = config.read_bytes()
            manifest = json.loads((project / "manifest.json").read_text())
            assert manifest["validation"]["configuration"] == "passed"
            assert all(
                manifest["validation"][key] == "not_run"
                for key in ("capture", "training", "export", "serving")
            )
            assert (
                json.loads((project / "draft.json").read_text())["num_hidden_layers"]
                == layers
            )
            cli("train", "-c", config, "--plan")
            cli(*args, error="refusing to overwrite")
            assert config.read_bytes() == before
            cli("train", "-c", config, "--plan", "training.typo=1", error="typo")
            cli(
                "train",
                "-c",
                config,
                "--plan",
                "model.draft_checkpoint_path=/weights",
                "training.resume_from=/checkpoint",
                error="mutually exclusive",
            )
        cli(
            "target",
            "prepare",
            target,
            "--local-only",
            "--strategy",
            "unimplemented",
            "--hidden-states",
            root / "features",
            "--output-dir",
            root / "invalid",
            error="unimplemented",
        )
        assert not (root / "invalid").exists()
    print(
        json.dumps(
            {
                "passed": True,
                "scope": (
                    "installed-wheel CPU metadata onboarding "
                    "and probability scoring reference"
                ),
                "research_recipes_executed": ["dforktree_scoring"],
                "algorithms_prepared": ["dspark", "dflash", "dflash2", "eagle4"],
                "gpu_started": False,
            }
        )
    )


if __name__ == "__main__":
    main()
