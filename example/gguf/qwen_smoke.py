#!/usr/bin/env python3
"""Run a practical LiteNN GGUF/Qwen smoke sequence.

Examples:
  python311 example/gguf/qwen_smoke.py \
    --model path/to/qwen.gguf \
    --litenn build-release/tools/gguf/litenn_gguf_convert.exe \
    --token-ids 1,2,3,4 \
    --max-tokens 8 \
    --output build/gguf_qwen_smoke/generated_token_ids.txt \
    --workdir build/gguf_qwen_smoke

  python311 example/gguf/qwen_smoke.py \
    --model path/to/qwen.gguf \
    --token-ids 1,2,3,4 \
    --capture-llamacpp \
    --llama-debug third_party/llama.cpp/build/bin/llama-debug.exe \
    --llama-cli third_party/llama.cpp/build/bin/llama-cli.exe \
    --prompt "hello" \
    --compare-logits \
    --compare-text
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def discover_litenn(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"LiteNN GGUF tool does not exist: {explicit}")
        return explicit
    root = repo_root()
    for name in ("litenn_gguf_convert.exe", "litenn_gguf_convert"):
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)
    for candidate in (
        root / "build-release" / "tools" / "gguf" / "litenn_gguf_convert.exe",
        root / "build-release" / "tools" / "gguf" / "litenn_gguf_convert",
        root / "build" / "tools" / "gguf" / "litenn_gguf_convert.exe",
        root / "build" / "tools" / "gguf" / "litenn_gguf_convert",
    ):
        if candidate.exists():
            return candidate
    raise SystemExit("litenn_gguf_convert executable was not found; pass --litenn")


def run_step(name: str, command: list[str], workdir: Path) -> dict[str, object]:
    completed = subprocess.run(command, text=True, capture_output=True)
    stdout = workdir / f"{name}.stdout.txt"
    stderr = workdir / f"{name}.stderr.txt"
    stdout.write_text(completed.stdout, encoding="utf-8")
    stderr.write_text(completed.stderr, encoding="utf-8")
    return {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "stdout": str(stdout),
        "stderr": str(stderr),
    }


def require_ok(step: dict[str, object]) -> None:
    if int(step["returncode"]) != 0:
        raise SystemExit(f"step failed: {step['name']} (see {step['stderr']})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Input GGUF model")
    parser.add_argument("--litenn", type=Path, help="Path to litenn_gguf_convert")
    parser.add_argument("--workdir", type=Path, default=Path("build/gguf_qwen_smoke"))
    parser.add_argument("--token-ids", help="Externally tokenized prompt ids, comma-separated")
    parser.add_argument("--prompt", help="Text prompt for llama.cpp capture")
    parser.add_argument("--steps", dest="steps", type=int, default=8)
    parser.add_argument("--max-tokens", dest="steps", type=int, help="Alias for --steps")
    parser.add_argument("--output", type=Path, help="Generated token-id output path for LiteNN token-id decode")
    parser.add_argument("--profile", default="qwen2-like-causal-lm")
    parser.add_argument(
        "--backend-policy",
        choices=("cpu-interpreter",),
        default="cpu-interpreter",
        help="Execution policy for this smoke driver; CUDA/AOT policies are tracked separately",
    )
    parser.add_argument("--sample", choices=("greedy", "random"), default="greedy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capture-llamacpp", action="store_true")
    parser.add_argument("--compare-logits", action="store_true")
    parser.add_argument("--compare-text", action="store_true")
    parser.add_argument("--llama-debug", type=Path)
    parser.add_argument("--llama-cli", type=Path)
    parser.add_argument("--allow-analysis-failure", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.capture_llamacpp and not args.prompt:
        raise SystemExit("--capture-llamacpp requires --prompt")
    if (args.compare_logits or args.compare_text) and not args.capture_llamacpp:
        raise SystemExit("--compare-logits/--compare-text require --capture-llamacpp")
    if not args.token_ids and not args.capture_llamacpp:
        raise SystemExit("provide --token-ids or enable --capture-llamacpp")
    if args.output is not None and not args.token_ids:
        raise SystemExit("--output is only used with --token-ids in this smoke driver")

    root = repo_root()
    workdir: Path = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    litenn = discover_litenn(args.litenn)
    steps: list[dict[str, object]] = []

    analyze = run_step("analyze", [str(litenn), "--analyze-llm", str(args.model), args.profile], workdir)
    steps.append(analyze)
    if not args.allow_analysis_failure:
        require_ok(analyze)

    token_ids = args.token_ids
    capture_dir = workdir / "llamacpp_capture"
    if args.capture_llamacpp:
        capture_cmd = [
            sys.executable,
            str(root / "scripts" / "gguf_capture_llamacpp_golden.py"),
            "--model",
            str(args.model),
            "--prompt",
            args.prompt,
            "--out-dir",
            str(capture_dir),
            "--predict",
            str(args.steps),
            "--seed",
            str(args.seed),
        ]
        if args.llama_debug is not None:
            capture_cmd += ["--llama-debug", str(args.llama_debug)]
        if args.llama_cli is not None:
            capture_cmd += ["--llama-cli", str(args.llama_cli)]
        capture = run_step("capture_llamacpp", capture_cmd, workdir)
        steps.append(capture)
        require_ok(capture)

        replay_cmd = [
            sys.executable,
            str(root / "scripts" / "gguf_run_litenn_from_golden.py"),
            "--manifest",
            str(capture_dir / "manifest.json"),
            "--litenn",
            str(litenn),
            "--steps",
            str(args.steps),
            "--sample",
            args.sample,
            "--seed",
            str(args.seed),
        ]
        replay = run_step("litenn_replay_from_golden", replay_cmd, workdir)
        steps.append(replay)
        require_ok(replay)

        if args.compare_logits:
            compare_cmd = [
                sys.executable,
                str(root / "scripts" / "gguf_compare_llamacpp_logits.py"),
                "--manifest",
                str(capture_dir / "manifest.json"),
                "--litenn",
                str(litenn),
            ]
            compare = run_step("compare_prefill_logits", compare_cmd, workdir)
            steps.append(compare)
            require_ok(compare)

        if args.compare_text:
            compare_text_cmd = [
                sys.executable,
                str(root / "scripts" / "gguf_compare_generation_text.py"),
                "--manifest",
                str(capture_dir / "manifest.json"),
                "--replay-manifest",
                str(capture_dir / "litenn_decode_manifest.json"),
            ]
            compare_text = run_step("compare_generation_text", compare_text_cmd, workdir)
            steps.append(compare_text)
            require_ok(compare_text)

    if token_ids:
        decode_output = args.output if args.output is not None else workdir / "litenn_decode_tokens.txt"
        decode_output.parent.mkdir(parents=True, exist_ok=True)
        decode = run_step(
            "litenn_decode_token_ids",
            [
                str(litenn),
                "--run-llama-decode-loop-token-ids",
                str(args.model),
                token_ids,
                str(args.steps),
                "--output",
                str(decode_output),
                "--sample",
                args.sample,
                "--seed",
                str(args.seed),
            ],
            workdir,
        )
        steps.append(decode)
        require_ok(decode)

    report = {
        "schema": "litenn.gguf_qwen_smoke.v1",
        "model": str(args.model),
        "backend_policy": args.backend_policy,
        "workdir": str(workdir),
        "token_output": str(args.output) if args.output is not None else None,
        "steps": steps,
    }
    report_path = workdir / "qwen_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
