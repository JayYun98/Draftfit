import pickle
import tempfile
import unittest
from pathlib import Path

import torch


class _UnexpectedCheckpointObject:
    def __reduce__(self):
        return (self._mark_executed, ())

    @staticmethod
    def _mark_executed():
        raise AssertionError("checkpoint pickle payload was executed")


class ExportCheckpointSecurityTest(unittest.TestCase):
    def test_export_loader_accepts_normal_checkpoint_payload(self):
        from specforge.export.checkpoint_io import resolve_training_state

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_state.pt"
            expected = {
                "draft_state_dict": {"weight": torch.ones(2)},
                "global_step": 3,
                "strategy": "dflash",
                "target_layer_ids": (1, 5),
            }
            torch.save(expected, path)

            state = resolve_training_state(str(path))

        self.assertEqual(state["global_step"], expected["global_step"])
        self.assertEqual(state["strategy"], expected["strategy"])
        self.assertEqual(state["target_layer_ids"], expected["target_layer_ids"])
        self.assertTrue(torch.equal(state["draft_state_dict"]["weight"], torch.ones(2)))

    def test_export_loader_rejects_pickle_objects_without_executing_them(self):
        from specforge.export.checkpoint_io import resolve_training_state

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malicious.pt"
            torch.save({"draft_state_dict": _UnexpectedCheckpointObject()}, path)

            with self.assertRaises(pickle.UnpicklingError):
                resolve_training_state(str(path))


if __name__ == "__main__":
    unittest.main()
