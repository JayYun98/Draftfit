"""Real CPU optimizer updates, WSD boundaries, and exact resume contracts."""

import copy
import io
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from draftfit.config.schema import TrainingConfig
from draftfit.lr_scheduler import WarmupStableDecayLR
from draftfit.optimizer import BF16Optimizer
from draftfit.training.assembly import _ConfiguredOptimizerFactory


def make_optimizer(**overrides):
    model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.fill_(1)
    settings = dict(lr=3e-6, total_steps=20, warmup_ratio=0.2, lr_scheduler="wsd")
    settings.update(overrides)
    return model, BF16Optimizer(model, **settings)


def update(model, optimizer, index):
    model.weight.grad = torch.full_like(model.weight, 0.01 * (1 + index % 3))
    used = optimizer.get_learning_rate()
    optimizer.step()
    return used


class WSDTests(unittest.TestCase):
    def test_actual_3000_update_campaign_boundaries_and_single_step_owner(self):
        model, optimizer = make_optimizer(total_steps=3000, warmup_ratio=0.05)
        with patch.object(
            optimizer.scheduler, "step", wraps=optimizer.scheduler.step
        ) as advance:
            used = [update(model, optimizer, index) for index in range(3005)]
        self.assertEqual(advance.call_count, 3005)
        self.assertEqual(optimizer.scheduler.last_epoch, 3005)
        self.assertEqual(
            (
                optimizer.scheduler.warmup_steps,
                optimizer.scheduler.stable_steps,
                optimizer.scheduler.decay_steps,
            ),
            (150, 2250, 600),
        )
        self.assertAlmostEqual(used[0], 3e-7)
        self.assertAlmostEqual(used[149], 3e-6)
        self.assertEqual(used[150:2400], [3e-6] * 2250)
        self.assertAlmostEqual(used[2400], 3e-6 * (1 - 0.9 / 600))
        self.assertEqual(used[2999:], [3e-6 * 0.1] * 6)
        self.assertTrue(all(a <= b for a, b in zip(used[:149], used[1:150])))
        self.assertTrue(all(a >= b for a, b in zip(used[2399:2999], used[2400:3000])))
        self.assertTrue(torch.isfinite(model.weight).all())
        for state in optimizer.optimizer.state.values():
            self.assertEqual(state["exp_avg"].dtype, torch.float32)
            self.assertEqual(state["exp_avg_sq"].dtype, torch.float32)

    def test_cosine_decay_and_distinct_parameter_group_peaks(self):
        parameters = [torch.nn.Parameter(torch.ones(1)) for _ in range(2)]
        optimizer = torch.optim.SGD(
            [
                {"params": [parameters[0]], "lr": 3e-6},
                {"params": [parameters[1]], "lr": 1e-5},
            ]
        )
        scheduler = WarmupStableDecayLR(
            optimizer,
            total_steps=10,
            warmup_steps=2,
            decay_steps=4,
            lr_decay_style="cosine",
        )
        used = []
        for index in range(13):
            used.append([group["lr"] for group in optimizer.param_groups])
            for parameter in parameters:
                parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scheduler.step()
            if index == 5:
                saved = copy.deepcopy(scheduler.state_dict())
        for index, factor in (
            (0, 0.1),
            (1, 1),
            (5, 1),
            (6, 0.1 + 0.9 * (1 + math.cos(math.pi / 4)) / 2),
            (9, 0.1),
            (12, 0.1),
        ):
            for observed, peak in zip(used[index], (3e-6, 1e-5)):
                self.assertAlmostEqual(observed, peak * factor)
        scheduler.load_state_dict(saved)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], used[6])
        optimizer.step()
        scheduler.step()
        self.assertEqual([group["lr"] for group in optimizer.param_groups], used[7])

    def test_invalid_wsd_parameters(self):
        for overrides in (
            {"total_steps": 0},
            {"total_steps": True},
            {"warmup_steps": -1},
            {"warmup_steps": 1},
            {"warmup_steps": 9, "decay_steps": 2},
            {"decay_steps": 0},
            {"decay_steps": 11},
            {"decay_steps": 1.5},
            {"min_lr_ratio": -0.1},
            {"min_lr_ratio": 1.1},
            {"min_lr_ratio": float("nan")},
            {"min_lr_ratio": float("inf")},
            {"lr_decay_style": "exponential"},
        ):
            with self.subTest(overrides=overrides):
                optimizer = torch.optim.SGD(
                    [torch.nn.Parameter(torch.ones(1))], lr=0.01
                )
                settings = dict(total_steps=10, warmup_steps=2, decay_steps=2)
                settings.update(overrides)
                with self.assertRaises(ValueError):
                    WarmupStableDecayLR(optimizer, **settings)
        for overrides in (
            {"warmup_ratio": float("nan")},
            {"warmup_ratio": 1},
            {"lr_decay_ratio": float("inf")},
            {"lr_decay_ratio": 0},
            {"warmup_ratio": 0.9, "lr_decay_ratio": 0.2},
            {"lr_decay_ratio": 0.001},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                make_optimizer(**overrides)

    def test_no_warmup_and_no_stable_phase(self):
        for warmup, decay in ((0, 10), (2, 8)):
            model, optimizer = make_optimizer(
                total_steps=10, warmup_ratio=warmup / 10, lr_decay_ratio=decay / 10
            )
            used = [update(model, optimizer, index) for index in range(12)]
            self.assertEqual(used[9:], [3e-6 * 0.1] * 3)

    def test_exact_optimizer_resume_across_phases_and_offload_modes(self):
        for style in ("linear", "cosine"):
            for split in (1, 3, 4, 15, 16, 19, 21):
                with self.subTest(style=style, split=split):
                    model, optimizer = make_optimizer(lr_decay_style=style)
                    for index in range(split):
                        update(model, optimizer, index)
                    stream = io.BytesIO()
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        stream,
                    )
                    stream.seek(0)
                    saved = torch.load(stream, weights_only=True)
                    resumed_model, resumed = make_optimizer(
                        lr_decay_style=style, offload_master=True
                    )
                    resumed_model.load_state_dict(saved["model"])
                    resumed.load_state_dict(saved["optimizer"])
                    self.assertEqual(
                        resumed.get_learning_rate(), optimizer.get_learning_rate()
                    )
                    for index in range(split, 24):
                        self.assertEqual(
                            update(model, optimizer, index),
                            update(resumed_model, resumed, index),
                        )
                        self.assertEqual(
                            optimizer.scheduler.last_epoch, resumed.scheduler.last_epoch
                        )
                        self.assertTrue(torch.equal(model.weight, resumed_model.weight))
                        for original, restored in zip(
                            optimizer.fp32_params, resumed.fp32_params
                        ):
                            self.assertTrue(torch.equal(original, restored))
                    self.assertTrue(
                        all(p.device.type == "cpu" for p in resumed.fp32_params)
                    )
                    for original, restored in zip(
                        optimizer.optimizer.state.values(),
                        resumed.optimizer.state.values(),
                    ):
                        for key in ("exp_avg", "exp_avg_sq", "step"):
                            self.assertTrue(torch.equal(original[key], restored[key]))

    def test_resume_rejects_changed_contract_before_optimizer_mutation(self):
        model, optimizer = make_optimizer()
        update(model, optimizer, 0)
        saved = copy.deepcopy(optimizer.state_dict())
        for override in (
            {"total_steps": 30},
            {"warmup_ratio": 0.3},
            {"lr_decay_ratio": 0.3},
            {"min_lr_ratio": 0.2},
            {"lr_decay_style": "cosine"},
            {"lr": 4e-6},
        ):
            with self.subTest(override=override):
                _, resumed = make_optimizer(**override)
                before = resumed.get_learning_rate()
                with self.assertRaisesRegex(ValueError, "WSD resume schedule mismatch"):
                    resumed.load_state_dict(saved)
                self.assertEqual(resumed.get_learning_rate(), before)
                self.assertEqual(resumed.optimizer.state, {})
        for field in (
            "total_steps",
            "warmup_steps",
            "decay_steps",
            "stable_steps",
            "min_lr_ratio",
        ):
            damaged = copy.deepcopy(saved)
            del damaged["scheduler_state_dict"][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                optimizer.load_state_dict(damaged)

    def test_schema_and_normal_assembly_use_wsd_fields(self):
        defaults = TrainingConfig()
        self.assertEqual(defaults.lr_scheduler, "cosine")
        self.assertEqual(
            (defaults.lr_decay_ratio, defaults.min_lr_ratio, defaults.lr_decay_style),
            (0.2, 0.1, "linear"),
        )
        training = TrainingConfig(
            lr_scheduler="wsd",
            total_steps=3000,
            warmup_ratio=0.05,
            lr_decay_ratio=0.2,
            min_lr_ratio=0.1,
            lr_decay_style="cosine",
            learning_rate=3e-6,
            optimizer_cpu_offload=True,
        )
        factory = _ConfiguredOptimizerFactory(SimpleNamespace(training=training))
        optimizer = factory(torch.nn.Linear(2, 2, dtype=torch.bfloat16))
        self.assertIsInstance(optimizer.scheduler, WarmupStableDecayLR)
        self.assertEqual(optimizer.scheduler.lr_decay_style, "cosine")
        self.assertEqual(optimizer.scheduler.total_steps, 3000)
        self.assertAlmostEqual(optimizer.get_learning_rate(), 3e-7)
        self.assertTrue(optimizer.offload_master)
        for kwargs in (
            {"warmup_ratio": 0.9},
            {"lr_decay_ratio": 0},
            {"min_lr_ratio": float("nan")},
            {"lr_decay_style": "invalid"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainingConfig(lr_scheduler="wsd", **kwargs)

    def test_real_joint_optimizer_uses_wsd_and_resume_contract(self):
        from draftfit.training.joint_family import (
            run_joint_family,
            validate_joint_family,
        )
        from tests import test_joint_family_core as fixtures

        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(
                os.environ,
                {
                    "SPECFORGE_DEVICE": "cpu",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
            ),
        ):
            root = Path(temporary)
            output, stages = fixtures.JointFamilyCoreTest().stages(root)
            for stage in stages:
                t = stage.config.training
                t.lr_scheduler = "wsd"
                t.total_steps = 10
                t.warmup_ratio = 0.2
                t.lr_decay_ratio = 0.3
                t.min_lr_ratio = 0.2
                t.lr_decay_style = "cosine"
            stages[1].config.training.min_lr_ratio = 0.4
            with self.assertRaisesRegex(ValueError, "min_lr_ratio"):
                validate_joint_family(stages)
            stages[1].config.training.min_lr_ratio = 0.2
            run_joint_family(output, stages, stop_at_step=1)
            first = torch.load(output / "joint-step1.pt", weights_only=True)
            schedule = first["optimizer"]["scheduler_state_dict"]
            self.assertEqual(schedule["last_epoch"], 1)
            self.assertEqual(
                (
                    schedule["total_steps"],
                    schedule["warmup_steps"],
                    schedule["decay_steps"],
                ),
                (10, 2, 3),
            )
            self.assertEqual(
                (schedule["min_lr_ratio"], schedule["lr_decay_style"]), (0.2, "cosine")
            )
            resumed_root = root / "resumed"
            run_joint_family(
                resumed_root, stages, resume_from=output / "joint-step1.pt"
            )
            resumed = torch.load(resumed_root / "joint-step2.pt", weights_only=True)
            self.assertEqual(
                resumed["optimizer"]["scheduler_state_dict"]["last_epoch"], 2
            )

    def test_exact_resume_requires_masters_and_consistent_lr_position(self):
        model, optimizer = make_optimizer()
        update(model, optimizer, 0)
        saved = copy.deepcopy(optimizer.state_dict())
        del saved["fp32_params"]
        with self.assertRaisesRegex(ValueError, "requires fp32_params"):
            optimizer.load_state_dict(saved)
        saved = copy.deepcopy(optimizer.state_dict())
        saved["scheduler_state_dict"]["_last_lr"] = [123.0]
        with self.assertRaisesRegex(ValueError, "LR disagrees"):
            optimizer.load_state_dict(saved)

    def test_search_validates_new_scheduler_distributions(self):
        from draftfit.training.search import SearchError, validate_contract
        from tests.test_training_search import contract

        space = contract()
        space["space"].update(
            {
                "training.lr_scheduler": {"kind": "categorical", "choices": ["wsd"]},
                "training.lr_decay_ratio": {
                    "kind": "float",
                    "low": 0.1,
                    "high": 0.3,
                    "log": False,
                },
                "training.min_lr_ratio": {
                    "kind": "float",
                    "low": 0.05,
                    "high": 0.2,
                    "log": False,
                },
                "training.lr_decay_style": {
                    "kind": "categorical",
                    "choices": ["linear", "cosine"],
                },
            }
        )
        validate_contract(space)
        for field, value in (
            ("training.lr_decay_ratio", 0),
            ("training.min_lr_ratio", -0.1),
        ):
            damaged = copy.deepcopy(space)
            damaged["space"][field]["low"] = value
            with self.subTest(field=field), self.assertRaises(SearchError):
                validate_contract(damaged)
        damaged = copy.deepcopy(space)
        damaged["space"]["training.lr_decay_style"]["choices"] = ["unsafe"]
        with self.assertRaises(SearchError):
            validate_contract(damaged)

    def test_unattended_presets_accept_wsd_and_reject_rounded_empty_decay(self):
        from draftfit.unattended import load_plan, write
        from tests import test_unattended_controller as fixtures

        fixture = fixtures.UnattendedControllerTest()
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.config["presets"] = [
            dict(
                id="wsd-pilot",
                when=["loss_plateau"],
                reason="WSD fixture",
                overrides=[
                    "training.lr_scheduler=wsd",
                    "training.warmup_ratio=0",
                    "training.lr_decay_ratio=0.5",
                    "training.min_lr_ratio=0.2",
                    "training.lr_decay_style=cosine",
                ],
                pilot_steps=2,
                pilot_cost_usd=0.1,
            )
        ]
        write(fixture.path, fixture.config)
        load_plan(fixture.path)
        fixture.config["presets"][0]["overrides"][2] = "training.lr_decay_ratio=0.2"
        write(fixture.path, fixture.config)
        with self.assertRaisesRegex(ValueError, "WSD decay must contain"):
            load_plan(fixture.path)
