from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Optional

from draftfit.algorithms.builtin import builtin_algorithm_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG_DIR = REPO_ROOT / "examples" / "configs"
DRAFT_CONFIG_DIR = REPO_ROOT / "configs"


def _yaml_scalar(path: Path, key: str) -> Optional[str]:
    prefix = f"{key}:"
    matches = []
    for line in path.read_text().splitlines():
        content = line.split("#", 1)[0].strip()
        if not content.startswith(prefix):
            continue
        value = content[len(prefix) :].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        matches.append(value)
    if len(matches) > 1:
        raise AssertionError(f"{path} defines {key} more than once")
    return matches[0] if matches else None


def _local_draft_configs():
    for recipe in sorted(EXAMPLE_CONFIG_DIR.glob("*.yaml")):
        source = _yaml_scalar(recipe, "draft_model_config")
        if source is None or not source.endswith(".json"):
            continue
        strategy = _yaml_scalar(recipe, "strategy")
        yield recipe, strategy, REPO_ROOT / source


class ExampleDraftConfigWiringTest(unittest.TestCase):
    def test_local_draft_architecture_matches_recipe_strategy(self):
        recipes = list(_local_draft_configs())
        self.assertTrue(recipes)
        registry = builtin_algorithm_registry()

        for recipe, strategy, draft_config in recipes:
            with self.subTest(recipe=recipe.name):
                algorithm = registry.resolve(strategy)
                draft_provider = algorithm.providers.model.draft_config
                self.assertTrue(draft_config.is_file(), draft_config)
                payload = json.loads(draft_config.read_text())
                architectures = payload.get("architectures")
                self.assertIsInstance(architectures, list)
                self.assertEqual(len(architectures), 1)
                architecture = architectures[0]
                self.assertIn(
                    architecture,
                    draft_provider.compatible_architectures,
                )
                expected_auto_model = draft_provider.expected_auto_map_model
                actual_auto_model = payload.get("auto_map", {}).get("AutoModel")
                if (
                    expected_auto_model is not None
                    and architecture == draft_provider.architecture
                ):
                    self.assertEqual(
                        actual_auto_model,
                        expected_auto_model,
                    )
                elif actual_auto_model is not None:
                    self.assertEqual(
                        actual_auto_model.rsplit(".", 1)[-1],
                        architecture,
                    )

    def test_only_documented_experiments_lack_a_unified_recipe(self):
        referenced = {
            draft_config.resolve()
            for _, _, draft_config in _local_draft_configs()
            if draft_config.is_file()
        }
        checked_in = {path.resolve() for path in DRAFT_CONFIG_DIR.glob("*.json")}

        self.assertEqual(
            {path.name for path in checked_in - referenced},
            {
                "qwen2-5-vl-7b-eagle3.json",
                "qwen2.5-vl-32b-eagle3.json",
                "minicpm5-2b-dspark.json",
                "minicpm5-2b-dflash2.json",
                "minicpm5-1b-eagle4.json",
                "xing4-29b-a4b-dspark.json",
                "xing4-29b-a4b-dflash2.json",
                "zdtaichu5-9b-dspark.json",
                "zdtaichu5-9b-dflash2.json",
                "hemmingway-1-dspark.json",
                "hemmingway-1-dflash2.json",
                "agnes3-flash-dspark.json",
                "agnes3-flash-dflash2.json",
                "lfm2.5-8b-a1b-dspark.json",
                "lfm2.5-2.6b-dspark.json",
                "lfm2.5-8b-a1b-dspark.json",
                "lfm2.5-2.6b-dspark.json",
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
