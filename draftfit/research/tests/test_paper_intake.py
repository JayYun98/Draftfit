import fcntl
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from draftfit.research.intake import record, scan


class PaperIntakeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.notes = self.root / "notes"
        self.notes.mkdir()
        self.state = self.root / "private" / "intake.json"

    def note(
        self, version=1, text="# Example paper\nPublic summary.\n", suffix="paper"
    ):
        path = self.notes / f"2611.05611_v{version}_{suffix}.md"
        path.write_text(text, encoding="utf-8")
        return path

    def test_scan_preserves_pending_work_and_does_not_write_notes(self):
        path = self.note()
        before = path.read_bytes()
        (self.notes / "index.md").write_text("# Literature index\n")
        first = scan(self.notes, self.state)
        persisted = self.state.read_bytes()
        second = scan(self.notes, self.state)
        self.assertEqual(first["pending_count"], 1)
        self.assertEqual(second["pending_count"], 1)
        self.assertEqual(second["changed"], [])
        self.assertEqual(second["context_notes"], ["index.md"])
        self.assertEqual(persisted, self.state.read_bytes())
        self.assertEqual(before, path.read_bytes())
        self.assertNotIn("Public summary", self.state.read_text())

    def test_latest_version_and_all_notes_at_that_version(self):
        self.note(1)
        self.note(2, suffix="revised")
        self.note(2, suffix="audit")
        result = scan(self.notes, self.state)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["version"], 2)
        self.assertEqual(len(candidate["notes"]), 2)
        self.assertTrue(all("_v2_" in n["path"] for n in candidate["notes"]))

    def test_decisions_are_idempotent_and_revisions_reopen_with_history(self):
        self.note()
        candidate = scan(self.notes, self.state)["candidates"][0]
        arguments = dict(
            paper_id=candidate["paper_id"],
            fingerprint=candidate["fingerprint"],
            status="pr_open",
            reason="Offline reference implementation submitted.",
            pr_url="https://github.com/example/draftfit/pull/12",
        )
        first = record(self.notes, self.state, **arguments)
        second = record(self.notes, self.state, **arguments)
        self.assertEqual(first, second)
        self.assertEqual(scan(self.notes, self.state)["pending_count"], 0)
        self.note(2)
        revised = scan(self.notes, self.state)["candidates"][0]
        self.assertEqual(revised["version"], 2)
        self.assertIn(arguments["pr_url"], revised["previous_pr_urls"])
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            record(self.notes, self.state, **arguments)
        history = json.loads(self.state.read_text())["papers"][candidate["paper_id"]][
            "decisions"
        ]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["version"], 1)

    def test_changed_note_reopens_a_rejected_candidate(self):
        path = self.note()
        candidate = scan(self.notes, self.state)["candidates"][0]
        record(
            self.notes,
            self.state,
            candidate["paper_id"],
            candidate["fingerprint"],
            "rejected",
            "Withdrawn according to the primary source.",
        )
        self.assertEqual(scan(self.notes, self.state)["candidates"], [])
        path.write_text("# Revised paper\nNew primary-source status.\n")
        revised = scan(self.notes, self.state)["candidates"][0]
        self.assertNotEqual(revised["fingerprint"], candidate["fingerprint"])

    def test_missing_latest_note_does_not_restore_an_obsolete_version(self):
        self.note(1)
        latest = self.note(2)
        scan(self.notes, self.state)
        latest.unlink()
        result = scan(self.notes, self.state)
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["unavailable"], ["2611.05611"])
        self.note(2)
        self.assertEqual(scan(self.notes, self.state)["pending_count"], 1)

    def test_removing_every_paper_marks_sources_unavailable(self):
        path = self.note()
        scan(self.notes, self.state)
        path.unlink()
        result = scan(self.notes, self.state)
        self.assertEqual(result["pending_count"], 0)
        self.assertEqual(result["unavailable"], ["2611.05611"])
        self.assertFalse(
            json.loads(self.state.read_text())["papers"]["2611.05611"]["available"]
        )
        with self.assertRaisesRegex(ValueError, "no versioned paper notes"):
            scan(self.notes, self.root / "fresh-state.json")
        self.assertFalse((self.root / "fresh-state.json").exists())

    def test_dry_run_creates_no_state_and_state_cannot_write_into_notes(self):
        self.note()
        self.assertEqual(scan(self.notes, self.state, dry_run=True)["pending_count"], 1)
        self.assertFalse(self.state.parent.exists())
        with self.assertRaisesRegex(ValueError, "outside"):
            scan(self.notes, self.notes / "state.json")

    def test_lock_releases_without_deleting_the_lock_file(self):
        self.note()
        scan(self.notes, self.state)
        with self.state.with_suffix(".json.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "in use"):
                scan(self.notes, self.state)
        self.assertEqual(scan(self.notes, self.state)["pending_count"], 1)

    def test_wrong_directory_symlinks_and_invalid_decisions_fail(self):
        self.note()
        candidate = scan(self.notes, self.state)["candidates"][0]
        other = self.root / "other"
        other.mkdir()
        (other / "2611.05611_v1_other.md").write_text("# Other source\n")
        with self.assertRaisesRegex(ValueError, "directory"):
            scan(other, self.state)
        with self.assertRaisesRegex(ValueError, "PR URL"):
            record(
                self.notes,
                self.state,
                candidate["paper_id"],
                candidate["fingerprint"],
                "pr_open",
                "Submitted.",
            )
        (self.notes / "2611.05612_v1_link.md").symlink_to(
            other / "2611.05611_v1_other.md"
        )
        with self.assertRaisesRegex(ValueError, "symlink"):
            scan(self.notes, self.state)

    def test_record_rechecks_disk_when_a_note_changed_after_the_scan(self):
        path = self.note()
        candidate = scan(self.notes, self.state)["candidates"][0]
        before = self.state.read_bytes()
        path.write_text("# Changed after the scan\n")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            record(
                self.notes,
                self.state,
                candidate["paper_id"],
                candidate["fingerprint"],
                "rejected",
                "Stale review.",
            )
        self.assertEqual(before, self.state.read_bytes())

    def test_real_cli_dispatch_and_invalid_limit(self):
        self.note()
        command = [
            sys.executable,
            "-m",
            "draftfit.research",
            "intake",
            "scan",
            "--notes-dir",
            str(self.notes),
            "--state",
            str(self.state),
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["pending_count"], 1)
        invalid = subprocess.run(
            command + ["--limit", "0"], text=True, capture_output=True
        )
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("limit", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
