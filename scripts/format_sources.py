#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".cu",
    ".cuh",
}

EXCLUDED_PREFIXES = (
    "third_party/",
    "build/",
    "build-",
    ".cache/",
    ".clangd/",
)

EXCLUDED_PARTS = {
    "__pycache__",
    "CMakeFiles",
}


def run(args, cwd=None):
    return subprocess.run(args, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE).stdout


def repo_root():
    try:
        return Path(run(["git", "rev-parse", "--show-toplevel"]).strip())
    except subprocess.CalledProcessError:
        return Path(__file__).resolve().parents[1]


def tracked_source_files(root):
    files = run(["git", "-C", str(root), "ls-files"], cwd=root).splitlines()
    result = []
    for name in files:
        normalized = name.replace("\\", "/")
        path = Path(normalized)
        if path.suffix not in SOURCE_SUFFIXES:
            continue
        if normalized.startswith(EXCLUDED_PREFIXES):
            continue
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        result.append(normalized)
    return result


def chunks(values, size):
    for index in range(0, len(values), size):
        yield values[index : index + size]


def main():
    parser = argparse.ArgumentParser(description="Format all tracked LiteNN C/C++ source files with clang-format.")
    parser.add_argument("--clang-format", default=os.environ.get("CLANG_FORMAT", "clang-format"))
    parser.add_argument("--check", action="store_true", help="check formatting without modifying files")
    args = parser.parse_args()

    root = repo_root()
    files = tracked_source_files(root)
    if not files:
        print("No source files found for clang-format.")
        return 0

    base_command = [args.clang_format, "--style=file"]
    if args.check:
        base_command.extend(["--dry-run", "--Werror"])
    else:
        base_command.append("-i")

    try:
        for batch in chunks(files, 100):
            subprocess.run(base_command + batch, cwd=root, check=True)
    except FileNotFoundError:
        print(f"clang-format executable not found: {args.clang_format}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as ex:
        return ex.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
