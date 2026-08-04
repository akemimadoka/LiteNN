#!/usr/bin/env python3
"""Run a CPU-only llama.cpp llama-bench control matrix for GGUF decode.

The script intentionally accepts model paths at runtime and redacts them from
the saved JSON by default, so benchmark evidence can be shared without leaking
local filesystem layout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def executable_name(name: str) -> str:
    return f"{name}.exe" if sys.platform == "win32" else name


def default_llama_bench_candidates() -> list[Path]:
    root = repo_root()
    binary = executable_name("llama-bench")
    return [
        root / "third_party" / "llama.cpp" / "build" / "bin" / binary,
        root / "third_party" / "llama.cpp" / "build" / "bin" / "Release" / binary,
        root / "third_party" / "llama.cpp" / "build" / "bin" / "RelWithDebInfo" / binary,
        root / "third_party" / "llama.cpp" / "build" / "tools" / "llama-bench" / binary,
        root / "third_party" / "llama.cpp" / "build" / "tools" / "llama-bench" / "Release" / binary,
        root / "third_party" / "llama.cpp" / "build" / "tools" / "llama-bench" / "RelWithDebInfo" / binary,
    ]


def resolve_llama_bench(path: Path | None) -> Path:
    if path is not None:
        resolved = path.resolve()
        if not resolved.is_file():
            raise SystemExit(f"llama-bench executable not found: {path}")
        return resolved
    for candidate in default_llama_bench_candidates():
        if candidate.is_file():
            return candidate.resolve()
    raise SystemExit("llama-bench executable not found; pass --llama-bench explicitly")


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def parse_json_stdout(stdout: str) -> list[dict[str, object]]:
    try:
        document = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"llama-bench did not produce valid JSON: {exc}") from exc
    if not isinstance(document, list):
        raise SystemExit("llama-bench JSON output must be an array")
    return [entry for entry in document if isinstance(entry, dict)]


def redact_rows(rows: list[dict[str, object]], redacted_model_name: str) -> list[dict[str, object]]:
    redacted = []
    for row in rows:
        next_row = dict(row)
        if "model_filename" in next_row:
            next_row["model_filename"] = redacted_model_name
        redacted.append(next_row)
    return redacted


def redact_text(
    text: str, model: Path, redacted_model_name: str, executable: Path | None = None
) -> str:
    candidates = {
        candidate: redacted_model_name
        for candidate in {str(model), str(model.resolve()), model.as_posix(), model.resolve().as_posix()}
    }
    if executable is not None:
        candidates.update(
            {
                candidate: "<llama-bench>"
                for candidate in {
                    str(executable),
                    str(executable.resolve()),
                    executable.as_posix(),
                    executable.resolve().as_posix(),
                }
            }
        )
    for candidate, replacement in list(candidates.items()):
        candidates[json.dumps(candidate)[1:-1]] = replacement
    redacted = text
    for candidate in sorted(candidates, key=len, reverse=True):
        if candidate:
            redacted = redacted.replace(candidate, candidates[candidate])
    return redacted


def write_markdown(path: Path, rows: list[dict[str, object]], command: list[str]) -> None:
    decode_rows = [row for row in rows if int(row.get("n_gen", 0) or 0) > 0]
    lines = [
        "# llama.cpp CPU Control",
        "",
        "Command:",
        "",
        "```text",
        " ".join(command),
        "```",
        "",
        "| threads | n_gen | tokens/s | ms/token | backend | ngl | flash_attn |",
        "| ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in decode_rows:
        tokens_per_second = float(row.get("avg_ts", 0.0) or 0.0)
        ms_per_token = 1000.0 / tokens_per_second if tokens_per_second > 0.0 else 0.0
        lines.append(
            f"| {int(row.get('n_threads', 0) or 0)} | {int(row.get('n_gen', 0) or 0)} | "
            f"{tokens_per_second:.6g} | {ms_per_token:.6g} | {row.get('backends', 'unknown')} | "
            f"{int(row.get('n_gpu_layers', 0) or 0)} | {row.get('flash_attn', 'unknown')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="GGUF model path, only passed to llama-bench")
    parser.add_argument("--llama-bench", type=Path, help="Path to llama-bench executable")
    parser.add_argument("--output-json", required=True, type=Path, help="Sanitized llama-bench JSON output")
    parser.add_argument("--output-md", type=Path, help="Optional human-readable summary")
    parser.add_argument("--threads", nargs="+", default=[2, 4, 8, 16, 32], type=positive_int)
    parser.add_argument("--n-gen", default=16, type=positive_int)
    parser.add_argument("--n-prompt", default=0, type=nonnegative_int)
    parser.add_argument("--repetitions", default=3, type=positive_int)
    parser.add_argument("--batch-size", default=2048, type=positive_int)
    parser.add_argument("--ubatch-size", default=512, type=positive_int)
    parser.add_argument("--gpu-layers", default=0, type=nonnegative_int)
    parser.add_argument("--flash-attn", choices=["0", "1"], default="0")
    parser.add_argument("--redacted-model-name", default="<model>")
    parser.add_argument("--keep-model-filename", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[], help="Additional llama-bench argument")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    llama_bench = resolve_llama_bench(args.llama_bench)
    model = args.model.resolve()
    if not model.is_file():
        raise SystemExit(f"model file not found: {args.model}")

    output_json = args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md = args.output_md
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(llama_bench),
        "-m",
        str(model),
        "-o",
        "json",
        "-p",
        str(args.n_prompt),
        "-n",
        str(args.n_gen),
        "-r",
        str(args.repetitions),
        "-b",
        str(args.batch_size),
        "-ub",
        str(args.ubatch_size),
        "-t",
        ",".join(str(thread) for thread in args.threads),
        "-ngl",
        str(args.gpu_layers),
        "-fa",
        args.flash_attn,
        *args.extra_arg,
    ]

    process = subprocess.run(command, text=True, capture_output=True, check=False)
    output_json.with_suffix(output_json.suffix + ".stdout.txt").write_text(
        redact_text(process.stdout, model, args.redacted_model_name, llama_bench), encoding="utf-8"
    )
    output_json.with_suffix(output_json.suffix + ".stderr.txt").write_text(
        redact_text(process.stderr, model, args.redacted_model_name, llama_bench), encoding="utf-8"
    )
    if process.returncode != 0:
        raise SystemExit(f"llama-bench failed with return code {process.returncode}")

    rows = parse_json_stdout(process.stdout)
    if not args.keep_model_filename:
        rows = redact_rows(rows, args.redacted_model_name)
    output_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if output_md is not None:
        printable_command = [
            "<llama-bench>"
            if arg == str(llama_bench)
            else args.redacted_model_name
            if arg == str(model)
            else arg
            for arg in command
        ]
        write_markdown(output_md, rows, printable_command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
