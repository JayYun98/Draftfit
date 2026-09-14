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
    def test_resume_rejects_objects_in_shared_and_rank_payloads(self):
        from specforge.training.checkpoint import CheckpointManager

        with tempfile.TemporaryDirectory() as directory:
            manager = CheckpointManager(directory, "security")
            checkpoint = Path(manager.checkpoint_dir(1))
            checkpoint.mkdir()
            shared = checkpoint / "training_state.pt"
            torch.save({"bad": _UnexpectedCheckpointObject()}, shared)
            with self.assertRaises(pickle.UnpicklingError):
                manager.load(1)
            with self.assertRaises(pickle.UnpicklingError):
                manager.read_resume_state(str(checkpoint))
            torch.save({"global_step": 1, "world_size": 1}, shared)
            torch.save(
                {"bad": _UnexpectedCheckpointObject()},
                checkpoint / "training_state_rank0.pt",
            )
            with self.assertRaises(pickle.UnpicklingError):
                manager.read_resume_state(str(checkpoint))

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
