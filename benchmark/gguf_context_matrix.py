#!/usr/bin/env python3
"""Run a GGUF/Qwen long-context decode matrix and summarize the rows.

The script is intentionally model-path neutral: pass private GGUF paths on the
command line, and keep the output directory under an ignored build/results path.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


RUN_MS_RE = re.compile(r"\brun_ms=(?P<value>[0-9.eE+-]+)\b")
BUILD_MS_RE = re.compile(r"\bbuild_ms=(?P<value>[0-9.eE+-]+)\b")
GENERATED_TOKENS_RE = re.compile(r"\bgenerated_tokens=(?P<value>\d+)\b")
MS_PER_GENERATED_TOKEN_RE = re.compile(r"\bms_per_generated_token=(?P<value>[0-9.eE+-]+)\b")
TOKENS_PER_SECOND_RE = re.compile(r"\bgenerated_tokens_per_second=(?P<value>[0-9.eE+-]+)\b")
PROMPT_REPLAY_MS_RE = re.compile(r"\bprompt_replay_ms=(?P<value>[0-9.eE+-]+)\b")
GENERATION_MS_RE = re.compile(r"\bgeneration_ms=(?P<value>[0-9.eE+-]+)\b")
FALLBACK_COUNT_RE = re.compile(r"\bfallback_count=(?P<value>\d+)\b")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_targets(text: str) -> list[int]:
    targets = []
    for item in text.split(","):
        stripped = item.strip().lower().replace("_", "")
        if not stripped:
            continue
        multiplier = 1
        if stripped.endswith("k"):
            multiplier = 1024
            stripped = stripped[:-1]
        elif stripped.endswith("m"):
            multiplier = 1024 * 1024
            stripped = stripped[:-1]
        value = int(stripped) * multiplier
        if value <= 0:
            raise argparse.ArgumentTypeError("context targets must be positive")
        targets.append(value)
    if not targets:
        raise argparse.ArgumentTypeError("at least one context target is required")
    return targets


def read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def first_float(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    return float(match.group("value")) if match is not None else None


def first_int(pattern: re.Pattern[str], text: str) -> int | None:
    match = pattern.search(text)
    return int(match.group("value")) if match is not None else None


def load_report(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"report is not a JSON object: {path}")
    return data


def step_text(report: dict[str, object], name: str) -> str:
    steps = report.get("steps")
    if not isinstance(steps, list):
        return ""
    for step in steps:
        if not isinstance(step, dict) or step.get("name") != name:
            continue
        stdout = Path(str(step.get("stdout", "")))
        stderr = Path(str(step.get("stderr", "")))
        base = Path(str(report.get("workdir", ".")))
        if not stdout.is_absolute():
            stdout = base / stdout.name if stdout.parent == Path(".") else stdout
        if not stderr.is_absolute():
            stderr = base / stderr.name if stderr.parent == Path(".") else stderr
        return read_text(stdout) + "\n" + read_text(stderr)
    return ""


def summarize_report(target: int, report_path: Path, returncode: int) -> dict[str, object]:
    report = load_report(report_path)
    decode_text = step_text(report, "litenn_decode_token_ids") or step_text(report, "litenn_replay_from_golden")
    analyze_text = step_text(report, "analyze")
    compile_only = bool(report.get("compile_only", False))
    return {
        "targetContext": target,
        "returncode": returncode,
        "compileOnly": compile_only,
        "decodeMode": report.get("decode_mode"),
        "backendPolicy": report.get("backend_policy"),
        "maxCacheLength": report.get("max_cache_length"),
        "runMs": first_float(RUN_MS_RE, decode_text),
        "buildMs": first_float(BUILD_MS_RE, decode_text + "\n" + analyze_text),
        "generatedTokens": first_int(GENERATED_TOKENS_RE, decode_text),
        "msPerGeneratedToken": first_float(MS_PER_GENERATED_TOKEN_RE, decode_text),
        "generatedTokensPerSecond": first_float(TOKENS_PER_SECOND_RE, decode_text),
        "promptReplayMs": first_float(PROMPT_REPLAY_MS_RE, decode_text),
        "generationMs": first_float(GENERATION_MS_RE, decode_text),
        "fallbackCount": first_int(FALLBACK_COUNT_RE, decode_text),
        "report": str(report_path),
        "trace": report.get("trace"),
        "waterfall": report.get("waterfall"),
    }


def write_outputs(out_dir: Path, rows: list[dict[str, object]], commands: list[list[str]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gguf_context_matrix.json").write_text(
        json.dumps({ "rows": rows, "commands": commands }, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# GGUF Context Matrix",
        "",
        "| Target | RC | Mode | Compile only | Build ms | Run ms | Gen tokens | ms/token | tok/s | Fallbacks | Report |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        def value(key: str) -> str:
            raw = row.get(key)
            if raw is None:
                return "n/a"
            if isinstance(raw, float):
                return f"{raw:.3f}"
            return str(raw)

        lines.append(
            "| "
            + " | ".join(
                [
                    value("targetContext"),
                    value("returncode"),
                    value("decodeMode"),
                    value("compileOnly"),
                    value("buildMs"),
                    value("runMs"),
                    value("generatedTokens"),
                    value("msPerGeneratedToken"),
                    value("generatedTokensPerSecond"),
                    value("fallbackCount"),
                    f"`{value('report')}`",
                ]
            )
            + " |"
        )
    (out_dir / "gguf_context_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--litenn", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark/results/gguf_context_matrix"))
    parser.add_argument("--targets", type=parse_targets, default=parse_targets("2k,32k,128k,1m"))
    parser.add_argument("--qwen-smoke", type=Path, default=repo_root() / "example" / "gguf" / "qwen_smoke.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--token-ids", default="1,2,3")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--stateful", action="store_true", default=True)
    parser.add_argument("--paged-reference-decode", action="store_true")
    parser.add_argument("--paged-resident-pages", type=int)
    parser.add_argument("--aot-cache-dir", type=Path)
    parser.add_argument("--require-aot-cache-hit", action="store_true")
    parser.add_argument("--no-aot-cache-write", action="store_true")
    parser.add_argument("--stream-stats", action="store_true")
    parser.add_argument("--no-compile-diagnostics", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    commands: list[list[str]] = []
    for target in args.targets:
        workdir = args.out_dir / f"context_{target}"
        command = [
            args.python,
            str(args.qwen_smoke),
            "--model",
            str(args.model),
            "--workdir",
            str(workdir),
            "--token-ids",
            args.token_ids,
            "--steps",
            str(args.steps),
            "--max-cache-length",
            str(target),
        ]
        if args.litenn is not None:
            command.extend(["--litenn", str(args.litenn)])
        if args.stateful:
            command.append("--stateful")
        if args.compile_only:
            command.append("--compile-only")
        if args.paged_reference_decode:
            command.append("--paged-reference-decode")
        if args.paged_resident_pages is not None:
            command.extend(["--paged-resident-pages", str(args.paged_resident_pages)])
        if args.aot_cache_dir is not None:
            command.extend(["--aot-cache-dir", str(args.aot_cache_dir)])
        if args.require_aot_cache_hit:
            command.append("--require-aot-cache-hit")
        if args.no_aot_cache_write:
            command.append("--no-aot-cache-write")
        if args.stream_stats:
            command.append("--stream-stats")
        if args.no_compile_diagnostics:
            command.append("--no-compile-diagnostics")
        commands.append(command)
        if args.dry_run:
            rows.append({ "targetContext": target, "returncode": None, "command": command })
            continue

        completed = subprocess.run(command, text=True, check=False)
        report_path = workdir / "qwen_smoke_report.json"
        if report_path.exists():
            rows.append(summarize_report(target, report_path, completed.returncode))
        else:
            rows.append({ "targetContext": target, "returncode": completed.returncode, "report": str(report_path) })
        if completed.returncode != 0:
            break

    write_outputs(args.out_dir, rows, commands)
    print(f"context matrix written to {args.out_dir}")
    return 0 if all(row.get("returncode") in (0, None) for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
