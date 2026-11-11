import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from draftfit.research.d3d import (
    corrected_kl_loss,
    corrected_teacher_probs,
    run_d3d_corrected_kl,
)


def forward_kl(target, student):
    return sum(q * math.log(q / p) for q, p in zip(target, student) if q)


def adapter_config(root, *, max_samples=32, input_kind="synthetic"):
    return SimpleNamespace(
        method="d3d_corrected_kl",
        data=SimpleNamespace(
            input_path=str(root / "input.jsonl"), max_samples=max_samples
        ),
        loss=SimpleNamespace(temperature=1.0, probability_margin=0.1),
        input_kind=input_kind,
        output_dir=str(root / "output"),
    )


def analytic_row():
    return {
        "id": "margin-and-mask",
        "student_logits": [
            [0.0, 0.0, 0.0],
            [math.log(0.5), math.log(0.25), math.log(0.25)],
            [0.0, 0.0, 0.0],
        ],
        "teacher_logits": [
            [math.log(0.1), math.log(0.6), math.log(0.3)],
            [math.log(0.7), math.log(0.2), math.log(0.1)],
            [0.0, 0.0, 0.0],
        ],
        "labels": [0, 0, -100],
        "weights": [1.0, 0.5, 0.0],
    }


class CorrectedTeacherProbsTest(unittest.TestCase):
    def test_minimal_margin_identity_and_other_token_ratios(self):
        teacher = torch.tensor(
            [[0.1, 0.6, 0.3], [0.7, 0.2, 0.1]], dtype=torch.float64
        ).log()
        original = teacher.clone()
        actual = corrected_teacher_probs(
            teacher, torch.tensor([0, 0]), probability_margin=0.1
        )
        expected = torch.tensor(
            [[0.46, 0.36, 0.18], [0.7, 0.2, 0.1]], dtype=torch.float64
        )
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-15)
        torch.testing.assert_close(actual.sum(-1), torch.ones(2, dtype=torch.float64))
        self.assertAlmostEqual(float(actual[0, 0] - actual[0, 1]), 0.1, places=14)
        self.assertAlmostEqual(float(actual[0, 1] / actual[0, 2]), 2.0, places=14)
        torch.testing.assert_close(actual[1], teacher.softmax(-1)[1], rtol=0, atol=0)
        torch.testing.assert_close(teacher, original, rtol=0, atol=0)
        smaller_mixture = (1 - 0.399) * torch.tensor(
            [0.1, 0.6, 0.3], dtype=torch.float64
        )
        smaller_mixture[0] += 0.399
        self.assertLess(float(smaller_mixture[0] - smaller_mixture[1]), 0.1)

    def test_tied_competitors_and_noncontiguous_batched_logits(self):
        teacher = (
            torch.tensor(
                [
                    [[0.4, 0.4, 0.2], [0.1, 0.6, 0.3]],
                    [[0.1, 0.6, 0.3], [0.4, 0.4, 0.2]],
                ],
                dtype=torch.float64,
            )
            .log()
            .transpose(0, 1)
        )
        labels = torch.tensor([[1, 0], [0, 1]]).transpose(0, 1)
        expected = torch.tensor(
            [
                [[0.36, 0.46, 0.18], [0.46, 0.36, 0.18]],
                [[0.46, 0.36, 0.18], [0.36, 0.46, 0.18]],
            ],
            dtype=torch.float64,
        ).transpose(0, 1)
        self.assertFalse(teacher.is_contiguous())
        actual = corrected_teacher_probs(teacher, labels, probability_margin=0.1)
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-15)

    def test_precision_and_fixed_teacher_boundary(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                teacher = torch.tensor(
                    [[0.0, 1.0, 2.0]], dtype=dtype, requires_grad=True
                )
                actual = corrected_teacher_probs(teacher, torch.tensor([0]))
                self.assertEqual(
                    actual.dtype,
                    torch.float64 if dtype == torch.float64 else torch.float32,
                )
                self.assertFalse(actual.requires_grad)
                self.assertIsNone(teacher.grad)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertAlmostEqual(float(actual.sum()), 1.0, places=6)
                self.assertAlmostEqual(
                    float(actual[0, 0] - actual[0, 2]), 0.01, places=6
                )

    def test_rejects_ignored_fractional_and_malformed_labels(self):
        logits = torch.zeros(2, 3)
        labels = (
            torch.tensor([0, -100]),
            torch.tensor([0, -1]),
            torch.tensor([0, 3]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.5, 1.0]),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([False, True]),
            torch.tensor([[0, 1]]),
            torch.tensor([0, 1], device="meta"),
            [0, 1],
        )
        for value in labels:
            with self.subTest(labels=value), self.assertRaises(ValueError):
                corrected_teacher_probs(logits, value)


class CorrectedKLLossTest(unittest.TestCase):
    def setUp(self):
        self.student = torch.tensor(
            [[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]], dtype=torch.float64
        ).log()
        self.teacher = torch.tensor(
            [[0.7, 0.2, 0.1], [0.8, 0.1, 0.1]], dtype=torch.float64
        ).log()
        self.labels = torch.tensor([0, 0])
        self.weights = torch.tensor([0.1, 0.2], dtype=torch.float64)

    def test_forward_kl_direction(self):
        teacher = torch.tensor([[0.1, 0.6, 0.3]], dtype=torch.float64).log()
        student = self.student[:1]
        actual = corrected_kl_loss(
            student,
            teacher,
            torch.tensor([0]),
            torch.tensor([0.125]),
            probability_margin=0.1,
        )
        target, distribution = [0.46, 0.36, 0.18], [0.5, 0.25, 0.25]
        self.assertAlmostEqual(
            float(actual), forward_kl(target, distribution), places=14
        )
        self.assertGreater(abs(float(actual) - forward_kl(distribution, target)), 1e-3)

    def test_weight_scaling_and_actual_subunit_denominator(self):
        first = forward_kl([0.7, 0.2, 0.1], [0.5, 0.25, 0.25])
        second = forward_kl([0.8, 0.1, 0.1], [0.2, 0.3, 0.5])
        expected = (0.1 * first + 0.2 * second) / 0.3
        for scale in (0.25, 1.0, 11.0, 1000.0):
            with self.subTest(scale=scale):
                actual = corrected_kl_loss(
                    self.student, self.teacher, self.labels, self.weights * scale
                )
                self.assertAlmostEqual(float(actual), expected, places=14)

    def test_temperature_changes_both_targets_and_student_and_scales_loss(self):
        teacher = torch.tensor([[0.1, 0.6, 0.3]], dtype=torch.float64).log()
        for temperature in (0.5, 2.0, 3.0):
            with self.subTest(temperature=temperature):
                teacher_mass = [q ** (1 / temperature) for q in (0.1, 0.6, 0.3)]
                q = [value / sum(teacher_mass) for value in teacher_mass]
                student_mass = [p ** (1 / temperature) for p in (0.5, 0.25, 0.25)]
                p = [value / sum(student_mass) for value in student_mass]
                eta = (0.1 - q[0] + max(q[1:])) / (1 - q[0] + max(q[1:]))
                target = [(1 - eta) * value for value in q]
                target[0] += eta
                corrected = corrected_teacher_probs(
                    teacher,
                    torch.tensor([0]),
                    temperature=temperature,
                    probability_margin=0.1,
                )
                torch.testing.assert_close(
                    corrected[0],
                    torch.tensor(target, dtype=torch.float64),
                    rtol=1e-13,
                    atol=1e-15,
                )
                actual = corrected_kl_loss(
                    self.student[:1],
                    teacher,
                    torch.tensor([0]),
                    torch.tensor([0.25]),
                    temperature=temperature,
                    probability_margin=0.1,
                )
                self.assertAlmostEqual(
                    float(actual), forward_kl(target, p) * temperature**2, places=13
                )

    def test_exact_one_hot_targets_and_extreme_logits_are_finite(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                teacher = torch.tensor(
                    [
                        [10000.0, -10000.0, 0.0],
                        [10000.0, -10000.0, 0.0],
                        [-10000.0, 10000.0, 0.0],
                    ],
                    dtype=dtype,
                    requires_grad=True,
                )
                student = torch.tensor(
                    [
                        [10000.0, -10000.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [-10000.0, 10000.0, 0.0],
                    ],
                    dtype=dtype,
                    requires_grad=True,
                )
                labels = torch.tensor([0, 1, 0])
                target = corrected_teacher_probs(
                    teacher, labels, probability_margin=0.1
                )
                torch.testing.assert_close(
                    target,
                    torch.tensor(
                        [[1.0, 0.0, 0.0], [0.45, 0.55, 0.0], [0.55, 0.45, 0.0]],
                        dtype=dtype,
                    ),
                )
                loss = corrected_kl_loss(
                    student, teacher, labels, torch.ones(3), probability_margin=0.1
                )
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(student.grad).all())
                self.assertIsNone(teacher.grad)

    def test_zero_mass_at_unrepresentable_student_log_probabilities(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                maximum = torch.finfo(dtype).max
                student = torch.tensor(
                    [[maximum, -maximum]], dtype=dtype, requires_grad=True
                )
                teacher = student.detach().clone()
                loss = corrected_kl_loss(
                    student, teacher, torch.tensor([0]), torch.ones(1)
                )
                loss.backward()
                self.assertEqual(float(loss.detach()), 0.0)
                self.assertTrue(torch.equal(student.grad, torch.zeros_like(student)))
                with self.assertRaises(ValueError):
                    corrected_kl_loss(
                        student, teacher, torch.tensor([1]), torch.ones(1)
                    )

    def test_masked_student_gradients_and_fixed_teacher_and_weights(self):
        for masked_label in (-100, 1):
            with self.subTest(masked_label=masked_label):
                student = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
                teacher = (
                    torch.tensor([[0.7, 0.2, 0.1]] * 3, dtype=torch.float64)
                    .log()
                    .requires_grad_()
                )
                weights = torch.tensor(
                    [0.2, 0.0, 0.3], dtype=torch.float64, requires_grad=True
                )
                labels = torch.tensor([0, masked_label, 0])
                loss = corrected_kl_loss(student, teacher, labels, weights)
                loss.backward()
                expected = torch.tensor(
                    [[1 / 3 - 0.7, 1 / 3 - 0.2, 1 / 3 - 0.1]], dtype=torch.float64
                ) * torch.tensor([[0.4], [0.0], [0.6]], dtype=torch.float64)
                torch.testing.assert_close(
                    student.grad, expected, rtol=1e-13, atol=1e-15
                )
                self.assertTrue(
                    torch.equal(student.grad[1], torch.zeros(3, dtype=torch.float64))
                )
                self.assertIsNone(teacher.grad)
                self.assertIsNone(weights.grad)

    def test_noncontiguous_3d_loss_and_integer_weights(self):
        student = self.student.repeat(2, 1).reshape(2, 2, 3).transpose(0, 1)
        teacher = self.teacher.repeat(2, 1).reshape(2, 2, 3).transpose(0, 1)
        labels = torch.zeros(2, 2, dtype=torch.long).transpose(0, 1)
        weights = torch.tensor([[1, 2], [3, 0]]).transpose(0, 1)
        expected = (
            4 * forward_kl([0.7, 0.2, 0.1], [0.5, 0.25, 0.25])
            + 2 * forward_kl([0.8, 0.1, 0.1], [0.2, 0.3, 0.5])
        ) / 6
        self.assertFalse(student.is_contiguous())
        loss = corrected_kl_loss(student, teacher, labels, weights)
        self.assertAlmostEqual(float(loss), expected, places=14)

    def test_low_precision_and_mixed_float64_loss(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                student = self.student.to(dtype).detach().requires_grad_()
                loss = corrected_kl_loss(
                    student, self.teacher.to(dtype), self.labels, self.weights.to(dtype)
                )
                self.assertEqual(
                    loss.dtype,
                    torch.float64 if dtype == torch.float64 else torch.float32,
                )
                loss.backward()
                self.assertTrue(torch.isfinite(student.grad).all())
        mixed = corrected_kl_loss(
            self.student.float(), self.teacher, self.labels, self.weights.float()
        )
        self.assertEqual(mixed.dtype, torch.float64)

    def test_real_linear_optimizer_update(self):
        model = torch.nn.Linear(2, 3, bias=False, dtype=torch.float64)
        with torch.no_grad():
            model.weight.copy_(
                torch.tensor([[0.0, 0.1], [0.2, 0.3], [-0.1, 0.2]], dtype=torch.float64)
            )
        features = torch.eye(2, dtype=torch.float64)
        teacher = self.teacher.detach().requires_grad_()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before_weights = model.weight.detach().clone()
        before_loss = corrected_kl_loss(
            model(features), teacher, self.labels, self.weights
        )
        optimizer.zero_grad()
        before_loss.backward()
        self.assertTrue(torch.isfinite(model.weight.grad).all())
        self.assertGreater(float(model.weight.grad.norm()), 0)
        optimizer.step()
        after_loss = corrected_kl_loss(
            model(features), teacher, self.labels, self.weights
        )
        self.assertTrue(torch.isfinite(model.weight).all())
        self.assertFalse(torch.equal(model.weight, before_weights))
        self.assertLess(float(after_loss.detach()), float(before_loss.detach()))
        self.assertIsNone(teacher.grad)

    def test_rejects_invalid_scalar_parameters_in_both_apis(self):
        invalid = {
            "temperature": (
                0,
                -1,
                float("nan"),
                float("inf"),
                True,
                "1",
                [1],
                torch.tensor(1.0),
                10**1000,
            ),
            "probability_margin": (
                0,
                1,
                -0.1,
                float("nan"),
                float("inf"),
                False,
                "0.1",
                [0.1],
                torch.tensor(0.1),
            ),
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(parameter=name, value=value):
                    with self.assertRaises(ValueError):
                        corrected_teacher_probs(
                            self.teacher, self.labels, **{name: value}
                        )
                    with self.assertRaises(ValueError):
                        corrected_kl_loss(
                            self.student,
                            self.teacher,
                            self.labels,
                            self.weights,
                            **{name: value},
                        )

    def test_rejects_logit_shapes_types_and_mismatches(self):
        invalid = (
            torch.zeros(3),
            torch.zeros(1, 1, 1, 3),
            torch.zeros(0, 3),
            torch.zeros(2, 1),
            torch.zeros(2, 3, dtype=torch.long),
            torch.zeros(2, 3, dtype=torch.complex64),
            [[0.0] * 3] * 2,
        )
        for logits in invalid:
            with self.subTest(logits=logits):
                with self.assertRaises(ValueError):
                    corrected_teacher_probs(logits, self.labels)
                with self.assertRaises(ValueError):
                    corrected_kl_loss(logits, self.teacher, self.labels, self.weights)
                with self.assertRaises(ValueError):
                    corrected_kl_loss(self.student, logits, self.labels, self.weights)
        with self.assertRaises(ValueError):
            corrected_kl_loss(
                self.student, torch.zeros(2, 4), self.labels, self.weights
            )
        with self.assertRaises(ValueError):
            corrected_kl_loss(
                self.student,
                torch.zeros(2, 3, device="meta"),
                self.labels,
                self.weights,
            )

    def test_rejects_invalid_labels_even_at_zero_weight(self):
        for labels in (
            torch.tensor([0, -1]),
            torch.tensor([0, 3]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.5, 1.0]),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([[0, 1]]),
            torch.tensor([0, 1], device="meta"),
        ):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                corrected_kl_loss(
                    self.student, self.teacher, labels, torch.tensor([1.0, 0.0])
                )
        with self.assertRaises(ValueError):
            corrected_kl_loss(
                self.student, self.teacher, torch.tensor([-100, 0]), self.weights
            )

    def test_rejects_invalid_weights_and_unrepresentable_sums(self):
        invalid = (
            torch.tensor([0.0, 0.0]),
            torch.tensor([-0.1, 0.2]),
            torch.tensor([float("nan"), 0.2]),
            torch.tensor([0.1, float("inf")]),
            torch.ones(2, 1),
            torch.ones(2, dtype=torch.complex64),
            torch.ones(2, device="meta"),
            [0.1, 0.2],
        )
        for weights in invalid:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                corrected_kl_loss(self.student, self.teacher, self.labels, weights)
        with self.assertRaises(ValueError):
            corrected_kl_loss(
                self.student.float(),
                self.teacher.float(),
                self.labels,
                torch.tensor([2e38, 2e38]),
            )

    def test_rejects_nonfinite_logits_including_masked_rows(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            teacher = self.teacher.clone()
            teacher[1, 0] = bad
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    corrected_teacher_probs(teacher, self.labels)
                with self.assertRaises(ValueError):
                    corrected_kl_loss(
                        self.student, teacher, self.labels, torch.tensor([1.0, 0.0])
                    )
                with self.assertRaises(ValueError):
                    corrected_kl_loss(
                        teacher, self.teacher, self.labels, torch.tensor([1.0, 0.0])
                    )

    def test_rejects_unrepresentable_scaled_logits_and_final_loss(self):
        logits = torch.full((2, 3), 2e38)
        with self.assertRaises(ValueError):
            corrected_teacher_probs(logits, self.labels, temperature=0.1)
        for student, teacher in (
            (logits, self.teacher.float()),
            (self.student.float(), logits),
        ):
            with (
                self.subTest(student=student, teacher=teacher),
                self.assertRaises(ValueError),
            ):
                corrected_kl_loss(
                    student, teacher, self.labels, self.weights.float(), temperature=0.1
                )
        with self.assertRaises(ValueError):
            corrected_kl_loss(
                torch.zeros(2, 3, dtype=torch.float64),
                torch.zeros(2, 3, dtype=torch.float64),
                self.labels,
                self.weights,
                temperature=1e200,
            )


class D3DAdapterTest(unittest.TestCase):
    def test_analytic_report_backward_diagnostic_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = (json.dumps(analytic_row()) + "\n").encode()
            (root / "input.jsonl").write_bytes(content)
            with torch.no_grad():
                result = run_d3d_corrected_kl(
                    adapter_config(root, input_kind="recorded")
                )
            answer = result["answers"][0]
            self.assertEqual(answer["id"], "margin-and-mask")
            expected_loss = (
                forward_kl([0.46, 0.36, 0.18], [1 / 3] * 3)
                + 0.5 * forward_kl([0.7, 0.2, 0.1], [0.5, 0.25, 0.25])
            ) / 1.5
            self.assertAlmostEqual(answer["kl_loss"], expected_loss, places=14)
            torch.testing.assert_close(
                torch.tensor(answer["corrected_teacher_probs"], dtype=torch.float64),
                torch.tensor(
                    [[0.46, 0.36, 0.18], [0.7, 0.2, 0.1], [1 / 3] * 3],
                    dtype=torch.float64,
                ),
                rtol=1e-13,
                atol=1e-15,
            )
            expected_gradient = [
                (p - q) / 1.5 for p, q in zip([1 / 3] * 3, [0.46, 0.36, 0.18])
            ] + [(p - q) / 3 for p, q in zip([0.5, 0.25, 0.25], [0.7, 0.2, 0.1])]
            self.assertAlmostEqual(
                answer["student_gradient_norm"],
                math.sqrt(sum(g * g for g in expected_gradient)),
                places=14,
            )
            self.assertEqual(answer["active_positions"], 2)
            self.assertEqual(
                result["provenance"]["input_sha256"],
                hashlib.sha256(content).hexdigest(),
            )
            self.assertNotIn(str(root), json.dumps(result["provenance"]))
            self.assertNotIn("student_logits", result["provenance"])
            self.assertEqual(result["input_kind"], "recorded")
            self.assertEqual(result["scope"], "corrected_forward_kl_component_only")
            self.assertEqual(result["teacher_role"], "fixed_full_depth_drafter")
            json.dumps(result, allow_nan=False)
            self.assertFalse((root / "output").exists())

    def test_empty_and_malformed_json_lines(self):
        fractional = analytic_row()
        fractional["labels"] = [0.5, 0, -100]
        floating = analytic_row()
        floating["labels"] = [0.0, 0.0, -100.0]
        boolean = analytic_row()
        boolean["labels"] = [True, 0, -100]
        malformed_weights = analytic_row()
        malformed_weights["weights"] = [[1.0, 0.5, 0.0]]
        for content in (
            "",
            "\n",
            "{",
            "[]\n",
            "{}\n",
            json.dumps({"samples": [analytic_row()]}),
            json.dumps(fractional),
            json.dumps(floating),
            json.dumps(boolean),
            json.dumps(malformed_weights),
        ):
            with (
                self.subTest(content=content),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                (root / "input.jsonl").write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    run_d3d_corrected_kl(adapter_config(root))

    def test_max_samples_and_hash_of_the_entire_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = analytic_row()
            row.pop("id")
            content = (
                json.dumps(row) + "\n" + json.dumps(row) + "\nnot JSON\n"
            ).encode()
            (root / "input.jsonl").write_bytes(content)
            for maximum in (1, 2):
                result = run_d3d_corrected_kl(adapter_config(root, max_samples=maximum))
                self.assertEqual(
                    [answer["id"] for answer in result["answers"]], list(range(maximum))
                )
                self.assertEqual(
                    result["provenance"]["input_sha256"],
                    hashlib.sha256(content).hexdigest(),
                )
            with self.assertRaisesRegex(ValueError, "line 3"):
                run_d3d_corrected_kl(adapter_config(root, max_samples=3))


if __name__ == "__main__":
    unittest.main()
