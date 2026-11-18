"""Run from source after installing the wheel: python -I draftfit/research/tests/check_install.py."""

import hashlib
import json
import math
import os
import subprocess
import sys
import sysconfig
import tempfile
from importlib.resources import files
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3]
RECIPES = Path("examples/configs/research")


def research(exported, *args, error=None, module="draftfit.research"):
    command = [sys.executable, "-I", "-m", module]
    if module == "draftfit.cli":
        command.append("research")
    result = subprocess.run(
        [*command, *map(str, args)],
        cwd=exported,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "WANDB_MODE": "disabled",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    if error is not None:
        assert result.returncode != 0 and error in result.stderr, result.stderr
        return
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def run_recipe(exported, name, **kwargs):
    result = research(
        exported, "-c", RECIPES / f"{name}.yaml", f"output_dir={name}", **kwargs
    )
    assert json.loads((exported / name / "result.json").read_bytes()) == result
    return result


def check_resources(exported):
    package = files("draftfit.research")
    # Compare checkout resources as bytes only; all Python imports use the installed wheel.
    src = SOURCE / "draftfit/research"
    for path in [src / "real.py", src / "NOTICE.md", *(src / "papers").rglob("*")]:
        if path.is_file() and (
            path.suffix in (".py", ".md") or path.name.endswith(("LICENSE", "NOTICE"))
        ):
            name = path.relative_to(src).as_posix()
            assert package.joinpath(name).read_bytes() == path.read_bytes(), name

    for name, digest in {
        "treewy/upstream.py": "29c7d7cf103593090a1001c5a39086d95cb1f31cd6eb7b99219576de80ff76e0",
        "treewy/TREEWY_LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        "tabletune/upstream.py": "8d4c7e0d8b5d7a04f85f00fe7c28d80f1f76afd4c497657430b412df744ee80c",
        "tabletune/TABLETUNE_LICENSE": "c95bae1d1ce0235ecccd3560b772ec1efb97f348a79f0fbe0a634f0c2ccefe2c",
        "d3d/D3D_LICENSE": "81054477e017851780fab7d4c7c86e1477583f83f3c81aa1d60c0c4a2d6e347b",
        "real/REAL_LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    }.items():
        assert (
            hashlib.sha256(package.joinpath("papers", name).read_bytes()).hexdigest()
            == digest
        ), name

    for directory in ("docs/research", RECIPES):
        for path in (SOURCE / directory).rglob("*"):
            if path.suffix in (".md", ".json", ".yaml", ".yml"):
                name = path.relative_to(SOURCE)
                assert (exported / name).read_bytes() == path.read_bytes(), name
    for name in (
        "paper-intake",
        "d3d-corrected-kl",
        "dforktree-scoring",
        "real-loss",
        "mfd-prefix-tree",
    ):
        assert (exported / "docs/specs" / f"{name}.md").is_file(), name


def check_tabletune(exported):
    result = run_recipe(exported, "tabletune")
    assert result["schedule"] == [[1, 1, 3], [2, 4, 1], [5, 8, 0]]
    notes = exported / "paper-notes"
    notes.mkdir()
    (notes / "2611.05611_v1_fixture.md").write_text("# Public test fixture\n")
    intake = research(
        exported,
        "intake",
        "scan",
        "--notes-dir",
        notes,
        "--state",
        exported / "intake.json",
        module="draftfit.cli",
    )
    assert intake["pending_count"] == 1


def check_mfd(exported):
    recipe = RECIPES / "mfd-prefix-tree.yaml"
    args = ("-c", recipe, "output_dir=mfd-result")
    plan = research(exported, *args, "--plan", module="draftfit.cli")
    assert plan["execution"] == "not-run" and not (exported / "mfd-result").exists()
    selected = research(exported, *args, module="draftfit.cli")
    assert selected["model_execution_performed"] is False
    assert selected["serving_validation"] == "not-run"
    nodes = selected["answers"][0]["nodes"]
    assert [node["prefix"] for node in nodes] == [[0], [0, 0], [1], [1, 1]]
    assert [node["parent"] for node in nodes] == [0, 1, 0, 3]
    assert json.loads((exported / "mfd-result/result.json").read_bytes()) == selected
    research(exported, *args, error="output already exists", module="draftfit.cli")


def check_d3d(exported):
    corrected = run_recipe(exported, "d3d-corrected-kl")
    assert corrected["scope"] == "corrected_forward_kl_component_only"
    answer = corrected["answers"][0]
    assert answer["active_positions"] == 2
    assert answer["student_gradient_norm"] > 0
    assert all(
        abs(actual - expected) < 1e-12
        for actual, expected in zip(
            answer["corrected_teacher_probs"][0], [0.46, 0.36, 0.18], strict=True
        )
    )


def check_real(exported):
    from draftfit.research import real
    from draftfit.research.papers.real import method

    assert real.REALResult is method.REALResult
    assert real.real_loss is method.real_loss
    assert real.run_real_loss is method.run_real_loss
    result = run_recipe(exported, "real-loss")
    answer = result["answers"][0]
    assert math.isclose(
        answer["mean_acceptance_estimate"], 0.96, rel_tol=0, abs_tol=1e-12
    )
    assert math.isclose(answer["real_loss"], 0.52, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(answer["ce_loss"], math.log(10) / 2, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(
        answer["loss"],
        0.9 * 0.52 + 0.1 * math.log(10) / 2,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert (
        math.isfinite(answer["draft_gradient_norm"])
        and answer["draft_gradient_norm"] > 0
    )
    assert answer["active_blocks"] == 50 and answer["active_positions"] == 100
    assert (
        result["ce_target"] == "target_logits_argmax"
        and not result["ce_position_decay"]
    )
    assert (
        not result["sampling_provenance_verified"]
        and not result["author_wrapper_parity"]
    )
    assert (
        result["provenance"]["source_revision"]
        == "3dff4579c5f93f692108df8b5178c9e9200b8536"
    )
    assert result["provenance"]["license"] == "Apache-2.0"
    fixture = exported / RECIPES / "data/real-loss.json"
    assert (
        result["provenance"]["input_sha256"]
        == hashlib.sha256(fixture.read_bytes()).hexdigest()
    )


def check_dforktree(exported):
    recipe = "examples/configs/research/dforktree-scoring.yaml"
    output = exported / "outputs/research-dforktree-scoring"
    planned = research(exported, "-c", recipe, "--plan", "data.input_path=missing.json")
    assert planned["execution"] == "not-run"
    assert planned["implementation"] == "cpu_probability_scoring_reference"
    assert not output.exists()
    scored = research(exported, "-c", recipe)
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
        assert math.isclose(answer["tail_probability"], tail, rel_tol=0, abs_tol=1e-12)
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
    assert scored["input_sha256"] == hashlib.sha256(fixture.read_bytes()).hexdigest()
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

    paper = files("draftfit.research.papers.dforktree")
    for original, packaged in (
        ("LICENSE", "DFORKTREE_LICENSE"),
        ("NOTICE", "DFORKTREE_NOTICE"),
    ):
        digest = hashlib.sha256(paper.joinpath(packaged).read_bytes()).hexdigest()
        assert digest == source["sha256"][original]
    persisted = output / "result.json"
    result_bytes = persisted.read_bytes()
    assert json.loads(result_bytes) == scored
    resolved = json.loads((output / "recipe.json").read_text())
    assert resolved == research(exported, "-c", recipe, "--plan")["config"]
    research(exported, "-c", recipe, error="output already exists")
    assert persisted.read_bytes() == result_bytes
    research(
        exported,
        "-c",
        recipe,
        "data.input_path=missing.json",
        "output_dir=failed/result",
        error="missing.json",
    )
    assert not (exported / "failed").exists()


def main():
    if not sys.flags.isolated or not __debug__:
        raise RuntimeError("run with python -I and assertions enabled")
    import draftfit
    from draftfit.assets import export_assets

    installed = Path(draftfit.__file__).resolve()
    if not installed.is_relative_to(Path(sysconfig.get_path("purelib")).resolve()):
        raise RuntimeError(
            f"expected draftfit installed in this environment: {installed}"
        )
    with tempfile.TemporaryDirectory(prefix="draftfit-installed-research-") as temp:
        exported = export_assets(Path(temp) / "exported")
        check_resources(exported)
        check_tabletune(exported)
        check_d3d(exported)
        check_dforktree(exported)
        check_mfd(exported)
        check_real(exported)
        tree = run_recipe(exported, "treewy", module="draftfit.cli")
        assert tree["answers"][0]["committed_gdn_state"]
    print(
        "Installed research gate passed: TableTune, D3D, DForkTree, MFD, REAL and TreeWY (CPU)."
    )


if __name__ == "__main__":
    main()
