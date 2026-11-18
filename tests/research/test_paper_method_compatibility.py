"""Legacy patch paths still affect paper implementations selected by the runner."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from draftfit.research.config import load_recipe
from draftfit.research.runner import run

RECIPES = Path(__file__).resolve().parents[2] / "examples/configs/research"


class PaperMethodCompatibilityTest(unittest.TestCase):
    def test_legacy_entry_point_patches_intercept_recipe_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            for slug, entry in (
                ("arr", "run_arr"),
                ("rift", "run_rift"),
                ("moe_spac", "run_moe_policy"),
                ("query_layout", "run_query_layout"),
                ("treewy", "run_treewy"),
            ):
                with self.subTest(slug=slug):
                    cfg = load_recipe(RECIPES / (slug.replace("_", "-") + ".yaml"))
                    cfg.output_dir = str(Path(directory) / slug)
                    with patch(
                        f"draftfit.research.{slug}.{entry}", return_value={"id": slug}
                    ) as patched:
                        self.assertEqual(run(cfg), {"id": slug})
                        patched.assert_called_once_with(cfg)

    def test_legacy_vendor_helper_patch_applies_inside_solver(self):
        from draftfit.research._vendor.treewy import _l2norm
        from draftfit.research.papers.treewy.method import run_treewy

        cfg = load_recipe(RECIPES / "treewy.yaml")
        cfg.data.input_path = str(RECIPES / "data/treewy.json")
        with patch("draftfit.research._vendor.treewy._l2norm", wraps=_l2norm) as norm:
            self.assertTrue(run_treewy(cfg)["answers"])
            self.assertTrue(norm.called)
