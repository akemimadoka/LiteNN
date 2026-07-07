#!/usr/bin/env python3
"""Run a GGUF/Qwen LiteNN CPU AOT decode thread matrix.

Pass private GGUF paths at runtime and keep outputs under an ignored build
directory. The saved command list redacts the model path by default.
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


def parse_threads(text: str) -> list[int]:
    threads = []
    for item in text.split(","):
        stripped = item.strip().lower()
        if not stripped:
            continue
        value = 0 if stripped in ("auto", "default") else int(stripped)
        if value < 0:
            raise argparse.ArgumentTypeError("thread counts must be non-negative")
        threads.append(value)
    if not threads:
        raise argparse.ArgumentTypeError("at least one thread count is required")
    return threads


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
        return read_text(stdout) + "\n" + read_text(stderr)
    return ""


def summarize_report(thread_count: int, report_path: Path, returncode: int) -> dict[str, object]:
    report = load_report(report_path)
    decode_text = step_text(report, "litenn_decode_token_ids") or step_text(report, "litenn_replay_from_golden")
    analyze_text = step_text(report, "analyze")
    return {
        "threadCount": thread_count,
        "threadLabel": "auto" if thread_count == 0 else f"T{thread_count}",
        "returncode": returncode,
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


def redact_command(command: list[str], model: Path, replacement: str) -> list[str]:
    model_text = str(model)
    return [replacement if item == model_text else item for item in command]


def attach_profile_bundle(row: dict[str, object], thread_count: int, report_path: Path, out_dir: Path, python: str) -> None:
    bundle_dir = out_dir / f"threads_{thread_count}_profile_bundle"
    command = [
        python,
        str(repo_root() / "benchmark" / "profile_bundle.py"),
        "--skip-litenn-profile",
        "--out-dir",
        str(bundle_dir),
        "--qwen-smoke-report",
        str(report_path),
    ]
    completed = subprocess.run(command, text=True, check=False)
    row["profileBundleReturncode"] = completed.returncode
    row["profileBundle"] = str(bundle_dir)
    row["profileSummary"] = str(bundle_dir / "gguf_decode_summary.json")
    row["profileTrace"] = str(bundle_dir / "gguf_decode_trace.json")


def write_profile_compare(out_dir: Path, rows: list[dict[str, object]], python: str) -> dict[str, object] | None:
    summaries = [
        Path(str(row["profileSummary"]))
        for row in rows
        if row.get("profileSummary") is not None and Path(str(row["profileSummary"])).exists()
    ]
    if not summaries:
        return None

    compare_dir = out_dir / "profile_summary_compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    command = [
        python,
        str(repo_root() / "benchmark" / "gguf_decode_compare.py"),
        "--output-dir",
        str(compare_dir),
    ]
    for summary in summaries:
        command.extend(["--litenn-profile-summary", str(summary)])

    stdout_path = compare_dir / "gguf_decode_compare.stdout.txt"
    stderr_path = compare_dir / "gguf_decode_compare.stderr.txt"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, text=True, stdout=stdout, stderr=stderr, check=False)

    return {
        "returncode": completed.returncode,
        "outputDir": str(compare_dir),
        "json": str(compare_dir / "gguf_decode_compare.json"),
        "markdown": str(compare_dir / "gguf_decode_compare.md"),
        "csv": str(compare_dir / "gguf_decode_compare.csv"),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def write_outputs(
    out_dir: Path, rows: list[dict[str, object]], commands: list[list[str]], profile_compare: dict[str, object] | None
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gguf_decode_thread_matrix.json").write_text(
        json.dumps({ "rows": rows, "commands": commands, "profileCompare": profile_compare }, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# GGUF Decode Thread Matrix",
        "",
        "| Threads | RC | Mode | Build ms | Run ms | Gen tokens | ms/token | tok/s | Prompt ms | Generation ms | Fallbacks | Report | Profile summary |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
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
                    value("threadLabel"),
                    value("returncode"),
                    value("decodeMode"),
                    value("buildMs"),
                    value("runMs"),
                    value("generatedTokens"),
                    value("msPerGeneratedToken"),
                    value("generatedTokensPerSecond"),
                    value("promptReplayMs"),
                    value("generationMs"),
                    value("fallbackCount"),
                    f"`{value('report')}`",
                    f"`{value('profileSummary')}`",
                ]
            )
            + " |"
        )
    if profile_compare is not None:
        lines.extend(
            [
                "",
                "## Profile-Summary Compare",
                "",
                f"- returncode: {profile_compare.get('returncode')}",
                f"- markdown: `{profile_compare.get('markdown')}`",
                f"- csv: `{profile_compare.get('csv')}`",
                f"- json: `{profile_compare.get('json')}`",
            ]
        )
    (out_dir / "gguf_decode_thread_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--litenn", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark/results/gguf_decode_thread_matrix"))
    parser.add_argument("--threads", type=parse_threads, default=parse_threads("0,2,4,8,16,32"))
    parser.add_argument("--qwen-smoke", type=Path, default=repo_root() / "example" / "gguf" / "qwen_smoke.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--token-ids", default="1,2,3")
    parser.add_argument("--prompt")
    parser.add_argument("--llamacpp-tokenizer-tool", type=Path)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-cache-length", type=int)
    parser.add_argument("--aot-cache-dir", type=Path)
    parser.add_argument("--require-aot-cache-hit", action="store_true")
    parser.add_argument("--no-aot-cache-write", action="store_true")
    parser.add_argument("--llvm-opt-level", type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--cpu-aot-affinity", choices=("none", "compact"))
    parser.add_argument(
        "--cpu-aot-q8k-staged-matmul",
        action="store_true",
        help="Forward Q8_K-staged GGML_Q6_K CPU AOT matmul opt-in to qwen_smoke.py",
    )
    parser.add_argument(
        "--cpu-aot-ggml-prepacked-weights",
        action="store_true",
        help="Forward prepared GGML_Q4_K/GGML_Q6_K CPU AOT weight opt-in to qwen_smoke.py",
    )
    parser.add_argument(
        "--cpu-aot-ggml-prepacked-weight-policy",
        choices=("disabled", "profitable", "all"),
        help="Forward prepared GGML CPU AOT weight policy to qwen_smoke.py",
    )
    parser.add_argument("--stateful", action="store_true", default=True)
    parser.add_argument("--stream-stats", action="store_true")
    parser.add_argument("--profile-bundles", action="store_true")
    parser.add_argument("--no-profile-compare", action="store_true")
    parser.add_argument("--redacted-model-name", default="<model>")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = args.model.resolve()

    rows: list[dict[str, object]] = []
    commands: list[list[str]] = []
    for thread_count in args.threads:
        workdir = args.out_dir / f"threads_{thread_count}"
        command = [
            args.python,
            str(args.qwen_smoke),
            "--model",
            str(model),
            "--workdir",
            str(workdir),
            "--max-tokens",
            str(args.max_tokens),
        ]
        if args.litenn is not None:
            command.extend(["--litenn", str(args.litenn)])
        if args.prompt is not None:
            command.extend(["--prompt", args.prompt])
        else:
            command.extend(["--token-ids", args.token_ids])
        if args.llamacpp_tokenizer_tool is not None:
            command.extend(["--llamacpp-tokenizer-tool", str(args.llamacpp_tokenizer_tool)])
        if args.max_cache_length is not None:
            command.extend(["--max-cache-length", str(args.max_cache_length)])
        if args.aot_cache_dir is not None:
            command.extend(["--aot-cache-dir", str(args.aot_cache_dir)])
        if args.require_aot_cache_hit:
            command.append("--require-aot-cache-hit")
        if args.no_aot_cache_write:
            command.append("--no-aot-cache-write")
        if args.llvm_opt_level is not None:
            command.extend(["--llvm-opt-level", str(args.llvm_opt_level)])
        if args.cpu_aot_affinity is not None:
            command.extend(["--cpu-aot-affinity", args.cpu_aot_affinity])
        if args.cpu_aot_q8k_staged_matmul:
            command.append("--cpu-aot-q8k-staged-matmul")
        if args.cpu_aot_ggml_prepacked_weights:
            command.append("--cpu-aot-ggml-prepacked-weights")
        if args.cpu_aot_ggml_prepacked_weight_policy is not None:
            command.extend(["--cpu-aot-ggml-prepacked-weight-policy", args.cpu_aot_ggml_prepacked_weight_policy])
        if thread_count > 0:
            command.extend(["--cpu-aot-threads", str(thread_count)])
        if args.stateful:
            command.append("--stateful")
        if args.stream_stats:
            command.append("--stream-stats")

        commands.append(redact_command(command, model, args.redacted_model_name))
        if args.dry_run:
            rows.append({ "threadCount": thread_count, "threadLabel": "auto" if thread_count == 0 else f"T{thread_count}" })
            continue

        completed = subprocess.run(command, text=True, check=False)
        report_path = workdir / "qwen_smoke_report.json"
        if report_path.exists():
            row = summarize_report(thread_count, report_path, completed.returncode)
            if args.profile_bundles:
                attach_profile_bundle(row, thread_count, report_path, args.out_dir, args.python)
            rows.append(row)
        else:
            rows.append(
                {
                    "threadCount": thread_count,
                    "threadLabel": "auto" if thread_count == 0 else f"T{thread_count}",
                    "returncode": completed.returncode,
                    "report": str(report_path),
                }
            )
    profile_compare = None
    if args.profile_bundles and not args.no_profile_compare:
        profile_compare = write_profile_compare(args.out_dir, rows, args.python)
    write_outputs(args.out_dir, rows, commands, profile_compare)
    return 0 if all(row.get("returncode") in (None, 0) for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
