#!/usr/bin/env python3
"""Capture llama.cpp reference artifacts for LiteNN GGUF parity work.

Examples:
  python311 scripts/gguf_capture_llamacpp_golden.py \
    --model path/to/model.gguf \
    --prompt "hello" \
    --out-dir build/gguf_golden/qwen_prompt \
    --llama-debug third_party/llama.cpp/build/bin/llama-debug \
    --llama-cli third_party/llama.cpp/build/bin/llama-cli \
    --predict 16 --seed 42
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CommandResult:
    name: str
    command: list[str]
    returncode: int
    stdout_path: Path
    stderr_path: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def exe_names(base: str) -> tuple[str, ...]:
    if platform.system() == "Windows":
        return (base + ".exe", base)
    return (base,)


def discover_tool(explicit: Path | None, names: Iterable[str]) -> Path | None:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"executable does not exist: {explicit}")
        return explicit
    for name in names:
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)
    root = repo_root() / "third_party" / "llama.cpp"
    for build_dir in ("build", "build-release", "build-cuda", "build/bin", "build/bin/Release"):
        candidate_root = root / build_dir
        for name in names:
            candidate = candidate_root / name
            if candidate.exists():
                return candidate
    return None


def run_command(name: str, command: list[str], out_dir: Path) -> CommandResult:
    stdout_path = out_dir / f"{name}.stdout.txt"
    stderr_path = out_dir / f"{name}.stderr.txt"
    completed = subprocess.run(command, text=True, capture_output=True)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return CommandResult(name, command, completed.returncode, stdout_path, stderr_path)


def rel(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Input GGUF model")
    parser.add_argument("--prompt", required=True, help="Prompt text to capture")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for captured artifacts")
    parser.add_argument("--llama-debug", type=Path, help="Path to llama.cpp llama-debug executable")
    parser.add_argument("--llama-cli", type=Path, help="Path to llama.cpp llama-cli executable")
    parser.add_argument("--ctx-size", type=int, default=0, help="llama.cpp context size, 0 means model default")
    parser.add_argument("--threads", type=int, help="llama.cpp thread count")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for llama-cli final-text capture")
    parser.add_argument("--predict", type=int, default=0, help="If >0, capture final generated text with llama-cli")
    parser.add_argument("--debug-extra", action="append", default=[], help="Extra argument for llama-debug")
    parser.add_argument("--cli-extra", action="append", default=[], help="Extra argument for llama-cli")
    parser.add_argument("--allow-failure", action="store_true", help="Write manifest even if a command fails")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = repo_root()
    out_dir: Path = args.out_dir
    debug_dir = out_dir / "llamacpp_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    debug = discover_tool(args.llama_debug, exe_names("llama-debug"))
    cli = discover_tool(args.llama_cli, exe_names("llama-cli"))
    if debug is None:
        raise SystemExit("llama-debug executable was not found; pass --llama-debug")
    if args.predict > 0 and cli is None:
        raise SystemExit("llama-cli executable was not found; pass --llama-cli or set --predict 0")

    common = ["-m", str(args.model), "-p", args.prompt, "-c", str(args.ctx_size)]
    if args.threads is not None:
        common += ["-t", str(args.threads)]

    results: list[CommandResult] = []
    debug_command = [
        str(debug),
        *common,
        "--save-logits",
        "--logits-output-dir",
        str(debug_dir),
        *args.debug_extra,
    ]
    results.append(run_command("llama_debug", debug_command, out_dir))

    if args.predict > 0 and cli is not None:
        cli_command = [
            str(cli),
            *common,
            "--seed",
            str(args.seed),
            "--predict",
            str(args.predict),
            "--no-display-prompt",
            *args.cli_extra,
        ]
        results.append(run_command("llama_cli", cli_command, out_dir))

    failures = [result for result in results if result.returncode != 0]
    debug_artifacts = sorted(path for path in debug_dir.iterdir() if path.is_file())
    manifest = {
        "schema": "litenn.llamacpp_golden_capture.v1",
        "createdUtc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "prompt": args.prompt,
        "ctxSize": args.ctx_size,
        "threads": args.threads,
        "seed": args.seed,
        "predict": args.predict,
        "debugArtifacts": [rel(path, out_dir) for path in debug_artifacts],
        "commands": [
            {
                "name": result.name,
                "command": result.command,
                "returncode": result.returncode,
                "stdout": rel(result.stdout_path, out_dir),
                "stderr": rel(result.stderr_path, out_dir),
            }
            for result in results
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))

    if failures and not args.allow_failure:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
