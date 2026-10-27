"""Domain order is immaterial; coverage and uniqueness remain mandatory."""

import time
import unittest
from copy import deepcopy
from types import SimpleNamespace as NS

from draftfit.training.joint_control import observe_generation
from tests.test_joint_policy import policy


class JointControlDomainsTest(unittest.TestCase):
    def setUp(self):
        self.domains = ["general", "code", "math", "science"]
        self.settings = {
            "policy": policy(domains=list(self.domains)),
            "preset": "base",
            "allow_reference_timings": False,
        }
        rows = [
            dict(
                trial=0,
                row_index=i,
                token_ids=[10 + i, 17],
                generated_tokens=2,
                seconds=1.0,
                accepted_tokens=1,
                proposed_tokens=2,
            )
            for i in range(4)
        ]
        self.report = dict(
            schema="draftfit-joint-generation-v1",
            production_engine_speed=False,
            step=5,
            artifact_sha256="1" * 64,
            capture_fingerprint="capture-v1",
            timing_scope="full_generation_including_prefill_no_kv_cache",
            domains=sorted(self.domains),
            bindings=dict(
                config={"max_prompts": 4},
                prompts=[dict(domain=d) for d in self.domains],
            ),
            baseline=dict(runs=deepcopy(rows)),
            candidates=[dict(name="dspark", tokenparity=True, runs=rows)],
        )
        training = NS(strategy="dspark", model_dump=lambda **_: {"strategy": "dspark"})
        self.stages = [
            NS(
                name="dspark",
                config=NS(training=training),
                draft_config=NS(to_dict=lambda: {"hidden_size": 8}),
            )
        ]

    def observe(self):
        return observe_generation(
            self.report,
            {"dspark": {"eval/avg_loss": 1.0}},
            self.stages,
            self.settings,
            {},
            {"timestamp_unix": time.time(), "steps_used": 5},
        )

    def test_permuted_domains_preserve_full_coverage_and_inputs(self):
        for report_domains in (sorted(self.domains), list(reversed(self.domains))):
            with self.subTest(report_domains=report_domains):
                self.report["domains"] = report_domains
                original = deepcopy((self.report, self.settings))
                result = self.observe()
                serving = result["history"]["dspark"][0]["serving"]
                self.assertEqual(
                    {d["name"] for d in serving["domains"]}, set(self.domains)
                )
                self.assertTrue(all(d["requests"] == 1 for d in serving["domains"]))
                self.assertTrue(
                    result["decisions"]["dspark"]["current_serving_validated"]
                )
                self.assertEqual((self.report, self.settings), original)

    def test_missing_extra_and_duplicate_domains_are_rejected(self):
        cases = [
            (self.domains[:-1], sorted(self.domains)),
            (self.domains, sorted(self.domains)[:-1]),
            (self.domains, sorted(self.domains) + ["other"]),
            (self.domains + ["general"], sorted(self.domains)),
            (self.domains, sorted(self.domains) + ["general"]),
            (self.domains + ["general"], self.domains + ["general"]),
        ]
        for policy_domains, report_domains in cases:
            with self.subTest(policy=policy_domains, report=report_domains):
                self.settings["policy"]["domains"] = policy_domains
                self.report["domains"] = report_domains
                with self.assertRaises(ValueError):
                    self.observe()


if __name__ == "__main__":
    unittest.main()
