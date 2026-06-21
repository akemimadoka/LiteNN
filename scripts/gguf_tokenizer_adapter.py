#!/usr/bin/env python3
"""Run the optional llama.cpp tokenizer adapter with manifest-backed evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tool", required=True, type=Path, help="litenn_llamacpp_adapter executable")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    tokenize = subparsers.add_parser("tokenize")
    add_common_arguments(tokenize)
    tokenize.add_argument("--text", required=True)
    detokenize = subparsers.add_parser("detokenize")
    add_common_arguments(detokenize)
    detokenize.add_argument("--token-ids", required=True)
    chat_template = subparsers.add_parser("chat-template")
    add_common_arguments(chat_template)
    chat_template.add_argument("--text", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.tool.is_file():
        raise SystemExit(f"llama.cpp tokenizer adapter does not exist: {args.tool}")
    if not args.model.is_file():
        raise SystemExit(f"GGUF model does not exist: {args.model}")
    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = args.token_ids if args.operation == "detokenize" else args.text
    command = [str(args.tool), args.operation, str(args.model), payload, str(args.output)]
    environment = os.environ.copy()
    runtime_paths = [args.tool.parent, args.tool.parent / "bin"]
    environment["PATH"] = os.pathsep.join(str(path) for path in runtime_paths) + os.pathsep + environment.get("PATH", "")
    completed = subprocess.run(command, text=True, capture_output=True, env=environment)
    stdout_path = args.workdir / f"{args.operation}.stdout.txt"
    stderr_path = args.workdir / f"{args.operation}.stderr.txt"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    manifest = {
        "schema": "litenn.llamacpp_tokenizer_run.v1",
        "operation": args.operation,
        "model": str(args.model),
        "command": command,
        "returncode": completed.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "output": str(args.output),
    }
    manifest_path = args.workdir / f"{args.operation}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
