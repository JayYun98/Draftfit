"""Actual controller boundaries stop on measured runtime shortfall and retain best."""

import unittest

from draftfit.unattended import run, write
from tests import test_unattended_controller as fixtures


class ForecastControllerTest(unittest.TestCase):
    # Reuse fixture setup without duplicating the inherited controller suite.
    setUp = fixtures.UnattendedControllerTest.setUp

    def test_forecast_stops_after_committed_checkpoint_and_exports(self):
        self.config["policy"]["patience"] = 10
        self.config["forecast"] = dict(
            hourly_usd=1.0,
            overhead_seconds=1000.0,
            safety_factor=1.25,
            minimum_samples=1,
        )
        write(self.path, self.config)
        result = run(self.path)
        self.assertEqual(result["total_steps"], 1)
        self.assertEqual(
            result["phases"][-1]["last_observation"]["response"]["reason"],
            "forecast_runtime_budget",
        )
        self.assertFalse(result["forecast"]["feasible"])
        self.assertEqual(len(result["best"]), 3)
        resumed = run(self.path, resume=True)
        self.assertEqual(resumed["best"], result["best"])
