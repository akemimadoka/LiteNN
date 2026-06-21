#!/usr/bin/env python3
"""Capture exact-token, per-step llama.cpp decode logits and write a manifest."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True, type=Path, help="litenn_llamacpp_adapter executable")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prompt-token-ids", required=True)
    parser.add_argument("--generated-token-ids", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if not args.tool.is_file():
        raise SystemExit(f"llama.cpp decode golden tool does not exist: {args.tool}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logits_dir = args.out_dir / "logits"
    command = [
        str(args.tool),
        "decode-logits",
        str(args.model),
        args.prompt_token_ids,
        args.generated_token_ids,
        str(logits_dir),
    ]
    environment = os.environ.copy()
    runtime_paths = [args.tool.parent, args.tool.parent / "bin"]
    environment["PATH"] = os.pathsep.join(str(path) for path in runtime_paths) + os.pathsep + environment.get("PATH", "")
    completed = subprocess.run(command, text=True, capture_output=True, env=environment)
    (args.out_dir / "llamacpp_decode.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (args.out_dir / "llamacpp_decode.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        return completed.returncode

    generated_count = len([part for part in args.generated_token_ids.split(",") if part.strip()])
    manifest_command = [
        sys.executable,
        str(Path(__file__).with_name("gguf_make_decode_logits_manifest.py")),
        "--model",
        str(args.model),
        "--prompt-token-ids",
        args.prompt_token_ids,
        "--generated-token-ids",
        args.generated_token_ids,
        "--output",
        str(args.out_dir / "manifest.json"),
        "--producer",
        "llama.cpp API via litenn_llamacpp_adapter",
    ]
    for step in range(1, generated_count + 1):
        manifest_command += ["--logits", f"{step}={logits_dir / f'decode-step-{step}.txt'}"]
    return subprocess.run(manifest_command).returncode


if __name__ == "__main__":
    sys.exit(main())
