#!/usr/bin/env python3
"""Inventory and safely prune LiteNN scratch build artifacts.

The tool treats each direct child of the selected root as one retention unit.
It never follows symlinks and defaults to a dry run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path


CMAKE_TREE_MARKERS = ("CMakeCache.txt", "build.ninja", "Makefile")
DEFAULT_PROTECTED_NAMES = frozenset({".litenn-cache", ".litenn-shared-weights"})


@dataclass(frozen=True)
class ArtifactEntry:
    name: str
    path: Path
    size: int
    file_count: int
    newest_mtime_ns: int
    kind: str
    protected: bool


@dataclass(frozen=True)
class CleanupItem:
    entry: ArtifactEntry
    reasons: tuple[str, ...]


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{size} B"
        value /= 1024.0
    return f"{size} B"


def validate_root(root: Path, repo_root: Path, allow_cmake_tree: bool = False) -> Path:
    resolved_root = root.resolve(strict=True)
    resolved_repo = repo_root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError(f"artifact root is not a directory: {resolved_root}")
    if resolved_root == resolved_repo:
        raise ValueError("refusing to manage the repository root")
    try:
        resolved_root.relative_to(resolved_repo)
    except ValueError as error:
        raise ValueError(f"artifact root must be inside repository root: {resolved_repo}") from error
    if not allow_cmake_tree:
        markers = [name for name in CMAKE_TREE_MARKERS if (resolved_root / name).exists()]
        if markers:
            raise ValueError(
                "refusing to prune a CMake build tree without --allow-cmake-tree: " + ", ".join(markers)
            )
    return resolved_root


def is_link_or_junction(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _scan_directory(path: Path) -> tuple[int, int, int, bool]:
    size = 0
    file_count = 0
    newest_mtime_ns = path.stat(follow_symlinks=False).st_mtime_ns
    contains_protected = False
    stack = [path]
    while stack:
        current = stack.pop()
        with os.scandir(current) as iterator:
            for child in iterator:
                info = child.stat(follow_symlinks=False)
                newest_mtime_ns = max(newest_mtime_ns, info.st_mtime_ns)
                if is_link_or_junction(info):
                    contains_protected = True
                    continue
                if child.is_dir(follow_symlinks=False):
                    contains_protected |= child.name in DEFAULT_PROTECTED_NAMES
                    stack.append(Path(child.path))
                elif child.is_file(follow_symlinks=False):
                    contains_protected |= child.name in CMAKE_TREE_MARKERS
                    size += info.st_size
                    file_count += 1
    return size, file_count, newest_mtime_ns, contains_protected


def scan_entries(root: Path, protected_names: set[str] | frozenset[str]) -> list[ArtifactEntry]:
    entries: list[ArtifactEntry] = []
    for path in root.iterdir():
        info = path.stat(follow_symlinks=False)
        if is_link_or_junction(info):
            size, file_count, kind = 0, 0, "symlink"
            protected = True
            newest_mtime_ns = info.st_mtime_ns
        elif path.is_dir():
            size, file_count, newest_mtime_ns, contains_protected = _scan_directory(path)
            kind = "directory"
            protected = path.name in protected_names or contains_protected
        else:
            size, file_count, newest_mtime_ns = info.st_size, 1, info.st_mtime_ns
            kind = "file"
            protected = path.name in protected_names
        entries.append(
            ArtifactEntry(path.name, path, size, file_count, newest_mtime_ns, kind, protected)
        )
    return entries


def plan_cleanup(
    entries: list[ArtifactEntry],
    *,
    now_ns: int,
    older_than_days: float | None = None,
    max_total_bytes: int | None = None,
    max_entries: int | None = None,
    max_files: int | None = None,
) -> list[CleanupItem]:
    reasons_by_name: dict[str, list[str]] = {}
    if older_than_days is not None:
        cutoff_ns = now_ns - int(older_than_days * 86400 * 1_000_000_000)
        for entry in entries:
            if not entry.protected and entry.newest_mtime_ns < cutoff_ns:
                reasons_by_name.setdefault(entry.name, []).append(f"older than {older_than_days:g} days")

    def survivors() -> list[ArtifactEntry]:
        return [entry for entry in entries if entry.name not in reasons_by_name]

    candidates = sorted(
        (entry for entry in entries if not entry.protected and entry.name not in reasons_by_name),
        key=lambda entry: (entry.newest_mtime_ns, entry.name.lower()),
    )
    candidate_index = 0
    while True:
        remaining = survivors()
        total_exceeded = max_total_bytes is not None and sum(entry.size for entry in remaining) > max_total_bytes
        count_exceeded = max_entries is not None and len(remaining) > max_entries
        files_exceeded = max_files is not None and sum(entry.file_count for entry in remaining) > max_files
        if not total_exceeded and not count_exceeded and not files_exceeded:
            break
        if candidate_index >= len(candidates):
            break
        entry = candidates[candidate_index]
        candidate_index += 1
        reasons = reasons_by_name.setdefault(entry.name, [])
        if total_exceeded:
            reasons.append("total size limit")
        if count_exceeded:
            reasons.append("entry count limit")
        if files_exceeded:
            reasons.append("file count limit")

    by_name = {entry.name: entry for entry in entries}
    return [
        CleanupItem(by_name[name], tuple(reasons))
        for name, reasons in sorted(
            reasons_by_name.items(), key=lambda item: (by_name[item[0]].newest_mtime_ns, item[0].lower())
        )
    ]


def delete_planned(root: Path, items: list[CleanupItem]) -> None:
    resolved_root = root.resolve(strict=True)
    # Validate the entire plan before deleting anything, including Windows reparse points.
    for item in items:
        target = item.entry.path
        if target.parent.resolve(strict=True) != resolved_root or target.name != item.entry.name:
            raise ValueError(f"refusing to delete a target outside the artifact root: {target}")
        if item.entry.protected:
            raise ValueError(f"refusing to delete a protected entry: {target}")
        if is_link_or_junction(target.stat(follow_symlinks=False)):
            raise ValueError(f"refusing to delete a symlink or junction: {target}")
        if target.resolve(strict=True).parent != resolved_root:
            raise ValueError(f"refusing to delete a resolved target outside the artifact root: {target}")
    for item in items:
        target = item.entry.path
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def entry_json(entry: ArtifactEntry) -> dict[str, object]:
    result = asdict(entry)
    result["path"] = str(entry.path)
    return result


def budget_violations(
    entries: list[ArtifactEntry], *, max_total_bytes: int | None, max_entries: int | None, max_files: int | None
) -> list[str]:
    violations = []
    size = sum(entry.size for entry in entries)
    files = sum(entry.file_count for entry in entries)
    if max_total_bytes is not None and size > max_total_bytes:
        violations.append(f"size {human_size(size)} exceeds {human_size(max_total_bytes)}")
    if max_entries is not None and len(entries) > max_entries:
        violations.append(f"entry count {len(entries)} exceeds {max_entries}")
    if max_files is not None and files > max_files:
        violations.append(f"file count {files} exceeds {max_files}")
    return violations


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("build"), help="Scratch artifact root")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--keep", action="append", default=[], metavar="NAME", help="Protect a direct child by name")
    parser.add_argument("--older-than-days", type=float, help="Delete inactive entries older than this age")
    parser.add_argument("--max-total-gib", type=float, help="Prune oldest entries until the root fits this size")
    parser.add_argument("--max-entries", type=int, help="Prune oldest entries until this count is reached")
    parser.add_argument("--max-files", type=int, help="Prune oldest entries until the recursive file count fits")
    parser.add_argument("--top", type=int, default=25, help="Number of largest entries to print")
    parser.add_argument("--json-out", type=Path, help="Optional machine-readable inventory and plan")
    parser.add_argument("--allow-cmake-tree", action="store_true", help="Allow managing a CMake build tree")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--apply", action="store_true", help="Apply the plan; the default is dry-run")
    modes.add_argument("--check", action="store_true", help="Do not delete; exit 1 if a retention policy is exceeded")
    args = parser.parse_args(argv)
    if args.older_than_days is not None and (not math.isfinite(args.older_than_days) or args.older_than_days < 0):
        parser.error("--older-than-days must be finite and non-negative")
    if args.max_total_gib is not None and (not math.isfinite(args.max_total_gib) or args.max_total_gib < 0):
        parser.error("--max-total-gib must be finite and non-negative")
    if args.max_entries is not None and args.max_entries < 0:
        parser.error("--max-entries must be non-negative")
    if args.max_files is not None and args.max_files < 0:
        parser.error("--max-files must be non-negative")
    if args.check and all(
        value is None for value in (args.older_than_days, args.max_total_gib, args.max_entries, args.max_files)
    ):
        parser.error("--check requires at least one retention policy")
    for name in args.keep:
        if Path(name).name != name or name in (".", ".."):
            parser.error("--keep accepts direct child names only")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = validate_root(args.root, args.repo_root, args.allow_cmake_tree)
    protected_names = set(DEFAULT_PROTECTED_NAMES) | set(args.keep)
    entries = scan_entries(root, protected_names)
    max_total_bytes = None
    if args.max_total_gib is not None:
        max_total_bytes = int(args.max_total_gib * 1024**3)
    plan = plan_cleanup(
        entries,
        now_ns=time.time_ns(),
        older_than_days=args.older_than_days,
        max_total_bytes=max_total_bytes,
        max_entries=args.max_entries,
        max_files=args.max_files,
    )

    total_size = sum(entry.size for entry in entries)
    reclaimed_size = sum(item.entry.size for item in plan)
    print(f"Artifact root: {root}")
    print(f"Entries: {len(entries)}, files: {sum(entry.file_count for entry in entries)}")
    print(f"Logical size: {human_size(total_size)}")
    print(f"Protected size: {human_size(sum(entry.size for entry in entries if entry.protected))}")
    print("\nLargest entries:")
    for entry in sorted(entries, key=lambda value: (-value.size, value.name.lower()))[: args.top]:
        marker = " [protected]" if entry.protected else ""
        print(f"  {human_size(entry.size):>12}  {entry.file_count:>7} files  {entry.name}{marker}")

    mode = "APPLY" if args.apply else "CHECK" if args.check else "DRY RUN"
    print(f"\nCleanup plan ({mode}): {len(plan)} entries, {human_size(reclaimed_size)} reclaimable")
    for item in plan:
        print(f"  {human_size(item.entry.size):>12}  {item.entry.name}  [{', '.join(item.reasons)}]")
    if plan and not args.apply:
        print("\nNo files were deleted. Re-run with --apply after reviewing the plan.")

    limits = dict(max_total_bytes=max_total_bytes, max_entries=args.max_entries, max_files=args.max_files)
    planned_names = {item.entry.name for item in plan}
    remaining = [entry for entry in entries if entry.name not in planned_names]
    if args.apply:
        delete_planned(root, plan)
        remaining = scan_entries(root, protected_names)
        print(f"Deleted {len(plan)} entries; reclaimed {human_size(reclaimed_size)} logical bytes.")
    remaining_violations = budget_violations(remaining, **limits)
    remaining_size = sum(entry.size for entry in remaining)
    label = "Remaining" if args.apply else "Projected remaining"
    print(
        f"{label}: {human_size(remaining_size)}, {len(remaining)} entries, "
        f"{sum(entry.file_count for entry in remaining)} files"
    )
    if remaining_violations:
        print("WARNING: cleanup cannot meet the requested limits without removing protected entries:")
        for violation in remaining_violations:
            print(f"  {violation}")

    if args.json_out:
        payload = {
            "schema": "litenn.build_artifact_inventory.v1",
            "root": str(root),
            "totalSize": total_size,
            "entries": [entry_json(entry) for entry in entries],
            "cleanupPlan": [
                {"entry": entry_json(item.entry), "reasons": list(item.reasons)} for item in plan
            ],
            "applied": args.apply,
            "remainingSize": remaining_size,
            "remainingBudgetViolations": remaining_violations,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.check:
        return int(bool(plan or budget_violations(entries, **limits)))
    if args.apply:
        return int(bool(remaining_violations))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
