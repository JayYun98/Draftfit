"""Real tiny Qwen3 target + assembled family candidates, without a tokenizer."""
import copy
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from draftfit.algorithms.builtin import builtin_algorithm_registry
from draftfit.family import build_family_plan
from draftfit.offline_capture.transformers import OfflineTransformersCapture
from draftfit.optimizer import BF16Optimizer
from draftfit.training.assembly import build_model_bundle
from draftfit.training.joint_family import _backward_window, _tie_backbones
from draftfit.training.joint_online import JointOnlineSource
from draftfit.training.joint_verification import (
    _capture, _project, _propose, evaluate_generation, suspend_training_state,
)
from tests.test_joint_online_cpu import online_fixture


class JointVerificationTest(unittest.TestCase):
    gemma = False
    execution_device = "cpu"
    target_device = None
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="joint-verification-")
        self.addCleanup(self.directory.cleanup)
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        self.env = mock.patch.dict(os.environ, {"SPECFORGE_DEVICE": self.execution_device})
        self.env.start()
        self.addCleanup(self.env.stop)
        root = Path(self.directory.name)
        online_fixture(root, seeded=False, gemma=self.gemma)
        if self.target_device is not None:
            config_path = root / "base.json"
            config = json.loads(config_path.read_text())
            config["training"]["joint_target_device"] = self.target_device
            config_path.write_text(json.dumps(config))
        _, self.stages = build_family_plan(str(root / "family.json"))
        self.bundles = []
        registry = builtin_algorithm_registry()
        for stage in self.stages:
            draft_path = Path(stage.config.model.draft_model_config)
            draft_path.parent.mkdir(parents=True, exist_ok=True)
            stage.draft_config.to_json_file(draft_path)
            bundle = build_model_bundle(stage.config, algorithm=registry.resolve(stage.config.training.strategy))
            if self.bundles:
                _tie_backbones((self.bundles[0], bundle), scratch=True)
                bundle.model.lm_head = self.bundles[0].model.lm_head
                bundle.model.embed_tokens = self.bundles[0].model.embed_tokens
                bundle.target_head = self.bundles[0].target_head
            bundle.model.train()
            self.bundles.append(bundle)
        self.source = JointOnlineSource(self.stages[0], needs_dspark=True)
        strategies = [registry.resolve(s.config.training.strategy).providers.step.build(b.model)
                      for s, b in zip(self.stages, self.bundles)]
        loader, _ = self.source.loader(self.stages[0], 0, True, True)
        _backward_window(self.stages, strategies, self.bundles, [next(iter(loader))])
        drafts = torch.nn.ModuleList([bundle.draft_model for bundle in self.bundles])
        self.optimizer = BF16Optimizer(drafts, lr=1e-3, total_steps=8, warmup_ratio=0.0)
        self.optimizer.step()
        # Preserve nonempty model AND master gradients across suspension.
        for parameter in drafts.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.ones_like(parameter)
        for master in self.optimizer.fp32_params:
            master.grad = torch.full_like(master, 0.125)
        self.config = dict(interval=1, max_prompts=2, max_new_tokens=8,
                           draft_tokens=2, trials=1, offload_optimizer=True)

    def snapshot(self):
        roots = [b.model for b in self.bundles] + [self.source._teacher._model]
        parameters = {id(p): p for root in roots for p in root.parameters()}
        parameters.update({id(p): p for p in self.optimizer.fp32_params})
        return dict(
            parameters=[(p, p.device, p.detach().clone(), p.grad,
                         None if p.grad is None else p.grad.clone()) for p in parameters.values()],
            modes={id(m): m.training for root in roots for m in root.modules()},
            optimizer=copy.deepcopy(self.optimizer.state_dict()),
            python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state().clone(),
            cuda_rng=torch.cuda.get_rng_state_all() if self.execution_device.startswith("cuda") else None,
            teacher=self.source._teacher,
            counters=(self.source.capture_calls, self.source.captured_samples, self.source.capture_seconds),
        )

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def assert_restored(self, before):
        for parameter, device, value, gradient, grad_value in before["parameters"]:
            self.assertEqual(parameter.device, device)
            self.assertTrue(torch.equal(parameter, value))
            self.assertIs(parameter.grad, gradient)
            if gradient is not None:
                self.assertTrue(torch.equal(parameter.grad, grad_value))
        for root in [b.model for b in self.bundles] + [self.source._teacher._model]:
            for module in root.modules():
                self.assertEqual(module.training, before["modes"][id(module)])
        self.assert_tree_equal(self.optimizer.state_dict(), before["optimizer"])
        self.assertEqual(random.getstate(), before["python"])
        np.testing.assert_equal(np.random.get_state(), before["numpy"])
        self.assertTrue(torch.equal(torch.get_rng_state(), before["torch"]))
        if before["cuda_rng"] is not None:
            for previous, current in zip(before["cuda_rng"], torch.cuda.get_rng_state_all()):
                self.assertTrue(torch.equal(previous, current))
        self.assertIs(self.source._teacher, before["teacher"])
        self.assertEqual((self.source.capture_calls, self.source.captured_samples,
                          self.source.capture_seconds), before["counters"])
        for bundle in self.bundles[1:]:
            self.assertIs(bundle.draft_model.layers[0].self_attn.q_proj.weight,
                          self.bundles[0].draft_model.layers[0].self_attn.q_proj.weight)

    def test_actual_family_generation_parity_timing_and_restoration(self):
        from transformers import Qwen3ForCausalLM
        target_class = Qwen3ForCausalLM
        if self.gemma:
            from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedForConditionalGeneration
            target_class = Gemma4UnifiedForConditionalGeneration

        target = target_class.from_pretrained(self.stages[0].config.model.target_model_path,
                                               torch_dtype=torch.bfloat16).to(self.execution_device).eval()
        before = self.snapshot()
        draft_forward = [mock.patch.object(b.draft_model, "forward", wraps=b.draft_model.forward)
                         for b in self.bundles]
        with draft_forward[0] as d0, draft_forward[1] as d1, draft_forward[2] as d2:
            result = evaluate_generation(self.stages, self.bundles, self.source, self.optimizer, 7, self.config)
        self.assertTrue(result["tokenparity"])
        self.assertFalse(result["production_engine_speed"])
        self.assertEqual(result["step"], 7)
        self.assertEqual({c["strategy"] for c in result["candidates"]}, {"dflash", "dspark", "dflash2"})
        self.assertTrue(all(d.call_count > 0 for d in (d0, d1, d2)))
        self.assertGreater(result["total_seconds"], 0)
        json.dumps(result, allow_nan=False)
        for run, binding in zip(result["baseline"]["runs"], result["bindings"]["prompts"]):
            prefix = torch.tensor([binding["prompt_token_ids"]], device=self.execution_device)
            direct = []
            with torch.no_grad():
                for _ in range(self.config["max_new_tokens"]):
                    token = target(prefix, use_cache=False).logits[:, -1].argmax(-1)
                    direct.append(token.item())
                    prefix = torch.cat((prefix, token[:, None]), dim=1)
            self.assertEqual(run["token_ids"], direct)
        for candidate in result["candidates"]:
            self.assertGreater(candidate["proposed_tokens"], 0)
            self.assertGreater(candidate["seconds"], 0)
            self.assertGreater(candidate["tok_s"], 0)
            self.assertLessEqual(candidate["accepted_tokens"], candidate["proposed_tokens"])
            self.assertEqual(candidate["accepted_tokens"], sum(
                trace["accepted_tokens"] for run in candidate["runs"] for trace in run["traces"]))
        self.assert_restored(before)

    def test_real_target_accept_and_reject_different_generated_proposals(self):
        # Controlled proposals isolate verifier semantics. Target remains real;
        # unpatched model proposal semantics are exercised separately above.
        def proposal(bundle, features, anchor, count):
            ids = torch.cat((features.input_ids,
                             anchor[:, None].to(features.input_ids.device)), dim=1)
            output = []
            for _ in range(count):
                captured = _capture(self.source._teacher, ids, self.config)
                token = _project(bundle, captured.last_hidden_states[:, -1]).argmax(-1)
                output.append(token)
                ids = torch.cat((ids, token[:, None].to(ids.device)), dim=1)
            return torch.stack(output, dim=1)

        with mock.patch("draftfit.training.joint_verification._propose", side_effect=proposal):
            accepted = evaluate_generation(self.stages, self.bundles, self.source, self.optimizer, 1, self.config)
        self.assertTrue(accepted["tokenparity"])
        self.assertTrue(all(c["accepted_tokens"] == c["proposed_tokens"] > 0
                            for c in accepted["candidates"]))

        def wrong(bundle, features, anchor, count):
            tokens = proposal(bundle, features, anchor, count)
            tokens[:, 0] = (tokens[:, 0] + 1) % bundle.target_vocab_size
            return tokens

        with mock.patch("draftfit.training.joint_verification._propose", side_effect=wrong):
            rejected = evaluate_generation(self.stages, self.bundles, self.source, self.optimizer, 1, self.config)
        self.assertTrue(rejected["tokenparity"])
        self.assertTrue(all(c["accepted_tokens"] == 0 and c["reject_events"] > 0
                            for c in rejected["candidates"]))

    def test_family_heads_use_generated_predecessors_and_selector(self):
        dspark = next(b for s, b in zip(self.stages, self.bundles) if s.config.training.strategy == "dspark")
        selector = next(b for s, b in zip(self.stages, self.bundles) if s.config.training.strategy == "dflash2")
        self.assertIsNotNone(dspark.draft_model.markov_head)
        with suspend_training_state(self.bundles, self.optimizer, self.source._teacher):
            prompt = self.source._rows[True][0]["input_ids"][:2].unsqueeze(0)
            features = _capture(self.source._teacher, prompt, self.config)
            anchor = _project(dspark, features.last_hidden_states[:, -1]).argmax(-1)
            head = dspark.draft_model.markov_head
            with mock.patch.object(head, "sample_block_tokens", wraps=head.sample_block_tokens) as sampled:
                _propose(dspark, features, anchor, 2)
            self.assertTrue(torch.equal(sampled.call_args.kwargs["first_prev_token_ids"], anchor))
            draft = selector.draft_model
            with mock.patch.object(draft, "score_candidates", wraps=draft.score_candidates) as scored:
                tokens = _propose(selector, features, anchor, 2)
            self.assertEqual(scored.call_count, 2)
            self.assertTrue(torch.equal(scored.call_args_list[0].args[2], anchor))
            self.assertTrue(torch.equal(scored.call_args_list[1].args[2], tokens[:, 0]))
            self.assertTrue(all("training_successor_ids" not in call.kwargs for call in scored.call_args_list))

    def test_finally_restores_rng_modes_buffers_gradients_after_error(self):
        self.bundles[0].draft_model.layers[0].eval()  # Preserve mixed modes.
        before = self.snapshot()
        teacher = self.source._teacher
        rotary = teacher._model.rotary_emb
        buffer_name, old_buffer = next(rotary.named_buffers(recurse=False))
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with suspend_training_state(self.bundles, self.optimizer, teacher):
                self.assertTrue(all(p.grad is None for p, *_ in before["parameters"]))
                self.assertTrue(all(t.device.type == "cpu" for t in self.optimizer.fp32_params))
                random.random()
                np.random.random()
                torch.rand(3)
                rotary.register_buffer(buffer_name, torch.zeros_like(old_buffer), persistent=False)
                raise RuntimeError("injected")
        self.assertIs(getattr(rotary, buffer_name), old_buffer)
        self.assert_restored(before)

    def test_evaluation_forward_error_restores_state(self):
        before = self.snapshot()
        with mock.patch.object(self.bundles[0].draft_model, "forward", side_effect=RuntimeError("draft failed")):
            with self.assertRaisesRegex(RuntimeError, "draft failed"):
                evaluate_generation(self.stages, self.bundles, self.source, self.optimizer, 1, self.config)
        self.assert_restored(before)

    def test_next_real_training_update_is_identical_with_and_without_verification(self):
        registry = builtin_algorithm_registry()
        strategies = [registry.resolve(s.config.training.strategy).providers.step.build(b.model)
                      for s, b in zip(self.stages, self.bundles)]
        loader, _ = self.source.loader(self.stages[0], 0, True, True)
        window = [next(iter(loader))]
        before = self.snapshot()

        def update():
            _backward_window(self.stages, strategies, self.bundles, window)
            self.optimizer.step()
            return copy.deepcopy(self.optimizer.state_dict()), [
                copy.deepcopy(bundle.draft_model.state_dict()) for bundle in self.bundles]

        baseline = update()
        for parameter, _, value, _, grad_value in before["parameters"]:
            parameter.data.copy_(value)
            parameter.grad = grad_value.clone() if grad_value is not None else None
        self.optimizer.load_state_dict(copy.deepcopy(before["optimizer"]))
        random.setstate(before["python"])
        np.random.set_state(before["numpy"])
        torch.set_rng_state(before["torch"])
        if before["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(before["cuda_rng"])
        evaluate_generation(self.stages, self.bundles, self.source, self.optimizer, 1, self.config)
        self.assert_tree_equal(baseline, update())

    def test_eos_stops_before_proposals_and_trials_bind_exact_workload(self):
        reference = evaluate_generation(self.stages, self.bundles, self.source, self.optimizer, 1, self.config)
        first = reference["baseline"]["runs"][0]["token_ids"][0]
        config = dict(self.config, stop_token_ids=[first], max_prompts=1, trials=2)
        result = evaluate_generation(self.stages, self.bundles, self.source, self.optimizer, 1, config)
        self.assertEqual(len(result["baseline"]["runs"]), 2)
        self.assertTrue(all(run["token_ids"] == [first] for run in result["baseline"]["runs"]))
        self.assertTrue(all(c["tokenparity"] and c["proposed_tokens"] == 0 for c in result["candidates"]))
        self.assertTrue(all(c["acceptance_rate"] is None for c in result["candidates"]))

    def test_unsupported_target_and_deadline_fail_without_training_mutation(self):
        before = self.snapshot()
        target_config = self.source._teacher._model.config
        with mock.patch.object(target_config, "model_type", "unsupported_test_target"):
            with self.assertRaisesRegex(ValueError, "unsupported.*unsupported_test_target"):
                evaluate_generation(self.stages, self.bundles, self.source, self.optimizer, 1, self.config)
        with self.assertRaisesRegex(TimeoutError, "deadline"):
            evaluate_generation(self.stages, self.bundles, self.source, self.optimizer, 1,
                                dict(self.config, deadline_unix=0.0))
        self.assert_restored(before)


class GemmaJointVerificationTest(JointVerificationTest):
    gemma = True


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('DRAFTFIT_RUN_QUEUE_CUDA') == '1',
                     'explicit CUDA intermediate verification gate')
class JointVerificationCUDA(JointVerificationTest):
    execution_device = "cuda"


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('DRAFTFIT_RUN_QUEUE_CUDA') == '1',
                     'explicit CUDA Gemma intermediate verification gate')
class GemmaJointVerificationCUDA(JointVerificationCUDA):
    gemma = True


if __name__ == "__main__":
    unittest.main()
