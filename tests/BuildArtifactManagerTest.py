#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("manage_build_artifacts", ROOT / "scripts" / "manage_build_artifacts.py")
assert SPEC is not None and SPEC.loader is not None
manager = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = manager
SPEC.loader.exec_module(manager)


class BuildArtifactManagerTest(unittest.TestCase):
    def make_root(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory(dir=ROOT)
        root = Path(temporary.name) / "build-scratch"
        root.mkdir()
        return temporary, root

    def test_scan_uses_direct_children_as_retention_units(self) -> None:
        temporary, root = self.make_root()
        with temporary:
            (root / "run-a" / "nested").mkdir(parents=True)
            (root / "run-a" / "nested" / "weights.bin").write_bytes(b"1234")
            (root / "trace.json").write_bytes(b"12")
            entries = manager.scan_entries(root, manager.DEFAULT_PROTECTED_NAMES)
            self.assertEqual({entry.name for entry in entries}, {"run-a", "trace.json"})
            by_name = {entry.name: entry for entry in entries}
            self.assertEqual(by_name["run-a"].size, 4)
            self.assertEqual(by_name["run-a"].file_count, 1)

    def test_size_and_count_limits_delete_oldest_unprotected_entries(self) -> None:
        temporary, root = self.make_root()
        with temporary:
            now_ns = 10_000_000_000
            entries = [
                manager.ArtifactEntry(".litenn-cache", root / ".litenn-cache", 8, 1, 1, "directory", True),
                manager.ArtifactEntry("old", root / "old", 7, 1, 2, "directory", False),
                manager.ArtifactEntry("middle", root / "middle", 6, 1, 3, "directory", False),
                manager.ArtifactEntry("new", root / "new", 5, 1, 4, "directory", False),
            ]
            plan = manager.plan_cleanup(entries, now_ns=now_ns, max_total_bytes=14, max_entries=2)
            self.assertEqual([item.entry.name for item in plan], ["old", "middle"])
            self.assertTrue(all(item.entry.name != ".litenn-cache" for item in plan))

    def test_age_policy_uses_newest_file_activity(self) -> None:
        temporary, root = self.make_root()
        with temporary:
            day_ns = 86400 * 1_000_000_000
            entries = [
                manager.ArtifactEntry("old", root / "old", 1, 1, day_ns, "directory", False),
                manager.ArtifactEntry("new", root / "new", 1, 1, 9 * day_ns, "directory", False),
            ]
            plan = manager.plan_cleanup(entries, now_ns=10 * day_ns, older_than_days=5)
            self.assertEqual([item.entry.name for item in plan], ["old"])

    def test_file_limit_counts_nested_files_not_just_direct_children(self) -> None:
        temporary, root = self.make_root()
        with temporary:
            entries = [
                manager.ArtifactEntry("old", root / "old", 8, 8, 1, "directory", False),
                manager.ArtifactEntry("new", root / "new", 2, 2, 2, "directory", False),
            ]
            plan = manager.plan_cleanup(entries, now_ns=3, max_files=3)
            self.assertEqual([item.entry.name for item in plan], ["old"])
            self.assertEqual(plan[0].reasons, ("file count limit",))

    def test_nested_build_trees_and_shared_caches_are_protected(self) -> None:
        temporary, root = self.make_root()
        with temporary:
            for name, marker in (
                ("reference-build", "compiler/CMakeCache.txt"),
                ("cache-owner", ".litenn-cache/weights.bin"),
                ("external-cache-owner", ".litenn-shared-weights/weights.bin"),
            ):
                artifact = root / name / marker
                artifact.parent.mkdir(parents=True)
                artifact.write_bytes(b"payload")
            entries = manager.scan_entries(root, manager.DEFAULT_PROTECTED_NAMES)
            self.assertTrue(all(entry.protected for entry in entries))
            self.assertEqual(manager.plan_cleanup(entries, now_ns=1, max_files=0), [])

    def test_check_detects_limit_without_deleting(self) -> None:
        temporary, root = self.make_root()
        with temporary:
            artifact = root / "trace.json"
            artifact.write_bytes(b"{}")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(manager.main(["--root", str(root), "--max-files", "0", "--check"]), 1)
                self.assertEqual(manager.main(["--root", str(root), "--max-files", "1", "--check"]), 0)
            self.assertTrue(artifact.exists())

    def test_unreachable_budget_is_reported_after_apply(self) -> None:
        temporary, root = self.make_root()
        with temporary:
            protected = root / ".litenn-cache"
            protected.mkdir()
            (protected / "weights.bin").write_bytes(b"weights")
            (root / "old.log").write_bytes(b"old")
            report = root.parent / "inventory.json"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = manager.main(
                    ["--root", str(root), "--max-total-gib", "0", "--apply", "--json-out", str(report)]
                )
            self.assertEqual(result, 1)
            self.assertTrue((protected / "weights.bin").exists())
            self.assertFalse((root / "old.log").exists())
            self.assertIn("cannot meet", output.getvalue())
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(payload["applied"])
            self.assertEqual(payload["remainingSize"], 7)
            self.assertTrue(payload["remainingBudgetViolations"])

    def test_apply_meets_budget_and_retains_keep_entry(self) -> None:
        temporary, root = self.make_root()
        with temporary:
            (root / "old.log").write_bytes(b"old")
            (root / "evidence.json").write_bytes(b"{}")
            with contextlib.redirect_stdout(io.StringIO()):
                result = manager.main(
                    ["--root", str(root), "--max-files", "1", "--keep", "evidence.json", "--apply"]
                )
            self.assertEqual(result, 0)
            self.assertFalse((root / "old.log").exists())
            self.assertTrue((root / "evidence.json").exists())

    def test_protected_only_over_budget_is_not_reported_as_success(self) -> None:
        temporary, root = self.make_root()
        with temporary:
            cache = root / ".litenn-shared-weights"
            cache.mkdir()
            (cache / "weights.bin").write_bytes(b"weights")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(manager.main(["--root", str(root), "--max-files", "0", "--check"]), 1)
            self.assertTrue((cache / "weights.bin").exists())

    def test_cli_rejects_invalid_policies(self) -> None:
        for options in (
            ["--check"],
            ["--check", "--apply"],
            ["--max-files", "-1"],
            ["--max-total-gib", "nan"],
            ["--max-total-gib", "inf"],
            ["--older-than-days", "nan"],
        ):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    manager.parse_args(options)

    def test_validate_root_refuses_repository_and_cmake_tree(self) -> None:
        with self.assertRaisesRegex(ValueError, "repository root"):
            manager.validate_root(ROOT, ROOT)
        temporary, root = self.make_root()
        with temporary:
            (root / "CMakeCache.txt").write_text("fixture", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CMake build tree"):
                manager.validate_root(root, ROOT)
            self.assertEqual(manager.validate_root(root, ROOT, allow_cmake_tree=True), root.resolve())

    def test_delete_planned_removes_only_selected_direct_child(self) -> None:
        temporary, root = self.make_root()
        with temporary:
            selected = root / "selected"
            selected.mkdir()
            (selected / "artifact.bin").write_bytes(b"payload")
            retained = root / "retained"
            retained.mkdir()
            entry = manager.scan_entries(root, set())[0]
            if entry.name != "selected":
                entry = next(value for value in manager.scan_entries(root, set()) if value.name == "selected")
            manager.delete_planned(root, [manager.CleanupItem(entry, ("test",))])
            self.assertFalse(selected.exists())
            self.assertTrue(retained.exists())

    def test_delete_preflights_all_targets_before_removing_any(self) -> None:
        temporary, root = self.make_root()
        with temporary:
            (root / "selected").mkdir()
            (root / ".litenn-cache").mkdir()
            entries = {entry.name: entry for entry in manager.scan_entries(root, manager.DEFAULT_PROTECTED_NAMES)}
            plan = [manager.CleanupItem(entries[name], ("test",)) for name in ("selected", ".litenn-cache")]
            with self.assertRaisesRegex(ValueError, "protected entry"):
                manager.delete_planned(root, plan)
            self.assertTrue((root / "selected").exists())

    def test_delete_refuses_target_outside_root(self) -> None:
        temporary, root = self.make_root()
        with temporary:
            target = root.parent / "outside.txt"
            target.write_bytes(b"keep")
            entry = manager.ArtifactEntry(target.name, target, 4, 1, 1, "file", False)
            with self.assertRaisesRegex(ValueError, "outside the artifact root"):
                manager.delete_planned(root, [manager.CleanupItem(entry, ("test",))])
            self.assertTrue(target.exists())

    @unittest.skipUnless(os.name == "nt", "Windows junction regression")
    def test_windows_junction_does_not_scan_or_delete_target(self) -> None:
        temporary, root = self.make_root()
        with temporary:
            target = root.parent / "external"
            target.mkdir()
            (target / "weights.bin").write_bytes(b"keep")
            junction = root / "junction"
            result = subprocess.run(
                ["cmd", "/d", "/c", "mklink", "/J", str(junction), str(target)],
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            try:
                entry = manager.scan_entries(root, set())[0]
                self.assertTrue(entry.protected)
                self.assertEqual(entry.size, 0)
                self.assertEqual(entry.file_count, 0)
                self.assertEqual(manager.plan_cleanup([entry], now_ns=1, max_entries=0), [])
                unprotected = manager.ArtifactEntry(entry.name, entry.path, 0, 0, 1, "directory", False)
                with self.assertRaisesRegex(ValueError, "symlink or junction"):
                    manager.delete_planned(root, [manager.CleanupItem(unprotected, ("test",))])
                self.assertTrue((target / "weights.bin").exists())
            finally:
                junction.rmdir()

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink support is unavailable")
    def test_symlink_is_always_protected(self) -> None:
        temporary, root = self.make_root()
        with temporary:
            target = root / "target"
            target.mkdir()
            link = root / "link"
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError:
                self.skipTest("creating symlinks requires additional privileges")
            entry = next(value for value in manager.scan_entries(root, set()) if value.name == "link")
            self.assertEqual(entry.kind, "symlink")
            self.assertTrue(entry.protected)


if __name__ == "__main__":
    unittest.main()
