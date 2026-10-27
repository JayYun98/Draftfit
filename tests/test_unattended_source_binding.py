"""Continual input weights are immutable, including before the first update."""

import unittest

import torch
from safetensors.torch import load_file, save_file

from draftfit.modeling.auto import AutoDraftModel
from draftfit.unattended import Controller, load_plan, read, run, write
from tests import test_unattended_controller as fixture


class ContinualSourceBindingTest(unittest.TestCase):
    def test_seed_change_before_first_update_cannot_resume(self):
        case = fixture.UnattendedControllerTest()
        case.setUp()
        self.addCleanup(case.doCleanups)
        _, stages, _ = load_plan(case.path)
        seed = case.root / "seed"
        AutoDraftModel.from_config(stages[0].draft_config).save_pretrained(seed)
        base = read(case.root / "base.json")
        base["model"]["draft_checkpoint_path"] = str(seed)
        write(case.root / "base.json", base)
        config, stages, binding = load_plan(case.path)
        controller = Controller(config, stages, binding)
        controller.publisher.close()
        controller.lock.close()
        weights_path = seed / "model.safetensors"
        weights = load_file(str(weights_path))
        key = next(k for k, v in weights.items() if v.is_floating_point())
        weights[key] = weights[key] + torch.ones_like(weights[key])
        save_file(weights, str(weights_path))
        with self.assertRaisesRegex(ValueError, "changed"):
            run(case.path, resume=True)
        state = read(case.root / "unattended/state.json")
        self.assertEqual(state["total_steps"], 0)
        self.assertEqual(state["phases"], [])
