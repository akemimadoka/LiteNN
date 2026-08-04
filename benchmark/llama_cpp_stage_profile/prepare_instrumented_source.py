#!/usr/bin/env python3
"""Create a detached llama.cpp worktree with LiteNN CPU stage counters applied."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(command: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=check, text=True, capture_output=True)


def patch_is_applied(worktree: Path, patch: Path) -> bool:
    result = run(["git", "apply", "--reverse", "--check", str(patch)], cwd=worktree, check=False)
    return result.returncode == 0


def prepare(source: Path, worktree: Path, patch: Path, revision: str) -> None:
    source = source.resolve()
    worktree = worktree.resolve()
    patch = patch.resolve()
    if not (source / ".git").exists() or not (source / "ggml" / "include" / "ggml-cpu.h").is_file():
        raise SystemExit(f"llama.cpp source is not a git worktree: {source}")
    if not patch.is_file():
        raise SystemExit(f"stage-counter patch not found: {patch}")

    if not worktree.exists():
        worktree.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "worktree", "add", "--detach", str(worktree), revision], cwd=source)
    elif not (worktree / ".git").is_file():
        raise SystemExit(f"refusing to reuse a non-worktree directory: {worktree}")

    source_revision = run(["git", "rev-parse", revision], cwd=source).stdout.strip()
    worktree_revision = run(["git", "rev-parse", "HEAD"], cwd=worktree).stdout.strip()
    if source_revision != worktree_revision:
        raise SystemExit(
            f"instrumented worktree is at {worktree_revision}, expected {source_revision}; choose a new --worktree"
        )
    if not patch_is_applied(worktree, patch):
        run(["git", "apply", "--check", str(patch)], cwd=worktree)
        run(["git", "apply", str(patch)], cwd=worktree)

    print(worktree)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=script_dir.parents[1] / "third_party" / "llama.cpp")
    parser.add_argument("--worktree", required=True, type=Path)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--patch", type=Path, default=script_dir / "llama_cpp_cpu_stage_counters.patch")
    args = parser.parse_args()
    prepare(args.source, args.worktree, args.patch, args.revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
