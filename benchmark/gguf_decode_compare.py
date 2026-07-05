#!/usr/bin/env python3
"""Build GGUF decode comparison tables from LiteNN and external benchmark evidence."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


RUN_MS_RE = re.compile(r"\brun_ms=(?P<value>[0-9.eE+-]+)\b")
GENERATED_TOKENS_RE = re.compile(r"\bgenerated_tokens=(?P<value>\d+)\b")
BACKEND_RE = re.compile(r"\bbackend=(?P<value>[A-Za-z0-9_.-]+)\b")
DECODE_MODE_RE = re.compile(r"\bdecode_mode=(?P<value>[A-Za-z0-9_.-]+)\b")
FALLBACK_RE = re.compile(r"\bfallback=(?P<value>true|false)\b")
FALLBACK_COUNT_RE = re.compile(r"\bfallback_count=(?P<value>\d+)\b")
MS_PER_GENERATED_TOKEN_RE = re.compile(r"\bms_per_generated_token=(?P<value>[0-9.eE+-]+)\b")
GENERATED_TOKENS_PER_SECOND_RE = re.compile(r"\bgenerated_tokens_per_second=(?P<value>[0-9.eE+-]+)\b")
PROMPT_REPLAY_MS_RE = re.compile(r"\bprompt_replay_ms=(?P<value>[0-9.eE+-]+)\b")
GENERATION_MS_RE = re.compile(r"\bgeneration_ms=(?P<value>[0-9.eE+-]+)\b")


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def find_step(report: dict[str, object], name: str) -> dict[str, object] | None:
    steps = report.get("steps")
    if not isinstance(steps, list):
        return None
    return next((step for step in steps if isinstance(step, dict) and step.get("name") == name), None)


def resolve_evidence_path(raw: object, manifest: Path) -> Path:
    path = Path(str(raw))
    if path.is_file() or path.is_absolute():
        return path
    return manifest.parent / path


def litenn_row(path: Path) -> dict[str, object]:
    report = load_json(path)
    if not isinstance(report, dict) or report.get("schema") not in (
        "litenn.gguf_qwen_smoke.v1",
        "litenn.gguf_qwen_smoke.v2",
    ):
        raise SystemExit(f"unsupported LiteNN smoke report: {path}")
    step = find_step(report, "litenn_decode_token_ids") or find_step(report, "litenn_replay_from_golden")
    if step is None or step.get("returncode") != 0:
        raise SystemExit(f"LiteNN smoke report has no successful decode step: {path}")
    stdout = resolve_evidence_path(step.get("stdout"), path)
    stdout_text = stdout.read_text(encoding="utf-8")
    run_match = RUN_MS_RE.search(stdout_text)
    tokens_match = GENERATED_TOKENS_RE.search(stdout_text)
    if run_match is None or tokens_match is None:
        raise SystemExit(f"LiteNN decode stdout has no run_ms/generated_tokens metrics: {stdout}")
    run_ms = float(run_match.group("value"))
    token_count = int(tokens_match.group("value"))
    if run_ms <= 0 or token_count <= 0:
        raise SystemExit(f"LiteNN decode metrics must be positive: {stdout}")
    backend_match = BACKEND_RE.search(stdout_text)
    decode_mode_match = DECODE_MODE_RE.search(stdout_text)
    fallback_match = FALLBACK_RE.search(stdout_text)
    fallback_count_match = FALLBACK_COUNT_RE.search(stdout_text)
    ms_per_token_match = MS_PER_GENERATED_TOKEN_RE.search(stdout_text)
    tokens_per_second_match = GENERATED_TOKENS_PER_SECOND_RE.search(stdout_text)
    prompt_replay_ms_match = PROMPT_REPLAY_MS_RE.search(stdout_text)
    generation_ms_match = GENERATION_MS_RE.search(stdout_text)
    tokens_per_second = (
        float(tokens_per_second_match.group("value"))
        if tokens_per_second_match is not None
        else token_count * 1000.0 / run_ms
    )
    ms_per_token = (
        float(ms_per_token_match.group("value"))
        if ms_per_token_match is not None
        else run_ms / token_count
    )
    cpu_aot_options = report.get("cpu_aot_options")
    config = "n/a"
    if isinstance(cpu_aot_options, dict):
        config = (
            f"opt={cpu_aot_options.get('llvm_opt_level', 'n/a')},"
            f"T={cpu_aot_options.get('thread_count', 'auto')},"
            f"aff={cpu_aot_options.get('affinity', 'default')},"
            f"q8k={int(bool(cpu_aot_options.get('q8k_staged_matmul', False)))}"
        )
    return {
        "implementation": "LiteNN",
        "backend": backend_match.group("value") if backend_match is not None else report.get("backend_policy"),
        "decodeMode": (
            decode_mode_match.group("value") if decode_mode_match is not None else report.get("decode_mode", "unknown")
        ),
        "config": config,
        "tokens": token_count,
        "totalMs": run_ms,
        "promptReplayMs": float(prompt_replay_ms_match.group("value")) if prompt_replay_ms_match is not None else None,
        "generationMs": float(generation_ms_match.group("value")) if generation_ms_match is not None else None,
        "msPerToken": ms_per_token,
        "tokensPerSecond": tokens_per_second,
        "fallbackUsed": fallback_match.group("value") == "true" if fallback_match is not None else report.get("fallback_used"),
        "fallbackCount": int(fallback_count_match.group("value")) if fallback_count_match is not None else None,
        "topHelper": None,
        "helperSharePercent": None,
        "topOperator": None,
        "operatorSharePercent": None,
        "source": str(path),
    }


def litenn_profile_summary_row(path: Path) -> dict[str, object]:
    summary = load_json(path)
    if not isinstance(summary, dict) or "steps" not in summary:
        raise SystemExit(f"unsupported LiteNN GGUF decode summary: {path}")
    steps = summary.get("steps")
    helpers = summary.get("helpers")
    operators = summary.get("operators")
    if not isinstance(steps, list):
        raise SystemExit(f"LiteNN GGUF decode summary has no steps array: {path}")

    step_dicts = [step for step in steps if isinstance(step, dict)]
    total_step_ms = float(summary.get("total_step_ms", 0.0))
    if total_step_ms <= 0.0:
        total_step_ms = sum(float(step.get("step_ms", 0.0)) for step in step_dicts)
    def generated_tokens(step: dict[str, object]) -> int:
        try:
            return int(step.get("generated_tokens", 0) or 0)
        except (TypeError, ValueError):
            return 0

    token_count = max(
        [generated_tokens(step) for step in step_dicts if generated_tokens(step) > 0],
        default=sum(1 for step in step_dicts if step.get("phase") == "generation"),
    )
    if total_step_ms <= 0.0 or token_count <= 0:
        raise SystemExit(f"LiteNN GGUF decode summary must contain positive time and generated tokens: {path}")

    helper_rows = helpers if isinstance(helpers, list) else []
    top_helper = next((helper for helper in helper_rows if isinstance(helper, dict)), None)
    top_helper_name = None
    top_helper_share = None
    if isinstance(top_helper, dict):
        top_helper_name = str(top_helper.get("helper", "unknown"))
        percent = top_helper.get("percent_of_steps")
        top_helper_share = float(percent) if percent is not None else None
    operator_rows = operators if isinstance(operators, list) else []
    top_operator = next((operator for operator in operator_rows if isinstance(operator, dict)), None)
    top_operator_name = None
    top_operator_share = None
    if isinstance(top_operator, dict):
        top_operator_name = f"{top_operator.get('operator', 'unknown')}/{top_operator.get('role', 'unknown')}"
        percent = top_operator.get("percent_of_steps")
        top_operator_share = float(percent) if percent is not None else None

    tokens_per_second = token_count * 1000.0 / total_step_ms
    return {
        "implementation": "LiteNN",
        "backend": "cpu-aot",
        "decodeMode": "profile-summary",
        "config": "from-profile-bundle",
        "tokens": token_count,
        "totalMs": total_step_ms,
        "promptReplayMs": None,
        "generationMs": total_step_ms,
        "msPerToken": total_step_ms / token_count,
        "tokensPerSecond": tokens_per_second,
        "fallbackUsed": None,
        "fallbackCount": None,
        "topHelper": top_helper_name,
        "helperSharePercent": top_helper_share,
        "topOperator": top_operator_name,
        "operatorSharePercent": top_operator_share,
        "source": str(path),
    }


def llama_rows(path: Path) -> list[dict[str, object]]:
    document = load_json(path)
    if not isinstance(document, list):
        raise SystemExit("llama-bench JSON must contain an array")
    rows = []
    for entry in document:
        if not isinstance(entry, dict) or int(entry.get("n_gen", 0)) <= 0:
            continue
        tokens_per_second = float(entry.get("avg_ts", 0.0))
        if tokens_per_second <= 0:
            continue
        gpu_layers = int(entry.get("n_gpu_layers", 0))
        rows.append(
            {
                "implementation": "llama.cpp",
                "backend": "gpu" if gpu_layers > 0 else "cpu",
                "decodeMode": entry.get("test", "decode"),
                "config": f"gpu_layers={gpu_layers}",
                "tokens": int(entry["n_gen"]),
                "totalMs": int(entry["n_gen"]) * 1000.0 / tokens_per_second,
                "promptReplayMs": None,
                "generationMs": int(entry["n_gen"]) * 1000.0 / tokens_per_second,
                "msPerToken": 1000.0 / tokens_per_second,
                "tokensPerSecond": tokens_per_second,
                "fallbackUsed": False,
                "fallbackCount": 0,
                "topHelper": None,
                "helperSharePercent": None,
                "topOperator": None,
                "operatorSharePercent": None,
                "source": str(path),
            }
        )
    return rows


def pytorch_rows(path: Path) -> list[dict[str, object]]:
    document = load_json(path)
    entries = document.get("rows") if isinstance(document, dict) else document
    if not isinstance(entries, list):
        raise SystemExit("PyTorch/HF JSON must contain an array or a rows array")
    rows = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        tokens_per_second = float(entry.get("tokensPerSecond", 0.0))
        if tokens_per_second <= 0:
            continue
        rows.append(
            {
                "implementation": str(entry.get("implementation", "PyTorch/HF")),
                "backend": str(entry.get("backend", "unknown")),
                "decodeMode": str(entry.get("decodeMode", entry.get("mode", "decode"))),
                "config": str(entry.get("config", "n/a")),
                "tokens": entry.get("tokens"),
                "totalMs": entry.get("totalMs"),
                "promptReplayMs": entry.get("promptReplayMs"),
                "generationMs": entry.get("generationMs"),
                "msPerToken": 1000.0 / tokens_per_second,
                "tokensPerSecond": tokens_per_second,
                "fallbackUsed": entry.get("fallbackUsed", False),
                "fallbackCount": entry.get("fallbackCount"),
                "topHelper": entry.get("topHelper"),
                "helperSharePercent": entry.get("helperSharePercent"),
                "topOperator": entry.get("topOperator"),
                "operatorSharePercent": entry.get("operatorSharePercent"),
                "source": str(path),
            }
        )
    return rows


def backend_class(backend: object) -> str:
    text = str(backend).lower()
    return "gpu" if "cuda" in text or text == "gpu" else "cpu"


def baseline_family(implementation: object) -> str | None:
    text = str(implementation).lower()
    if text == "llama.cpp":
        return "llama.cpp"
    if "pytorch" in text or text.startswith("torch") or "huggingface" in text or text == "hf":
        return "PyTorch/HF"
    return None


def add_baseline_deltas(rows: list[dict[str, object]]) -> None:
    baselines: dict[tuple[str, str], float] = {}
    for row in rows:
        family = baseline_family(row["implementation"])
        if family is not None:
            key = (family, backend_class(row["backend"]))
            baselines[key] = max(baselines.get(key, 0.0), float(row["tokensPerSecond"]))
    for row in rows:
        kind = backend_class(row["backend"])
        throughput = float(row["tokensPerSecond"])
        for implementation, field in (("llama.cpp", "vsLlamaCppPercent"), ("PyTorch/HF", "vsPyTorchPercent")):
            baseline = baselines.get((implementation, kind))
            row[field] = (throughput / baseline - 1.0) * 100.0 if baseline else None


def write_outputs(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gguf_decode_compare.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    preferred_fields = [
        "implementation",
        "backend",
        "decodeMode",
        "config",
        "tokens",
        "totalMs",
        "promptReplayMs",
        "generationMs",
        "msPerToken",
        "tokensPerSecond",
        "vsLlamaCppPercent",
        "vsPyTorchPercent",
        "fallbackUsed",
        "fallbackCount",
        "topHelper",
        "helperSharePercent",
        "topOperator",
        "operatorSharePercent",
        "source",
    ]
    observed_fields = {field for row in rows for field in row}
    fields = [field for field in preferred_fields if field in observed_fields]
    fields.extend(sorted(observed_fields.difference(fields)))
    with (output_dir / "gguf_decode_compare.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Implementation | Backend | Mode | Config | ms/token | token/s | vs llama.cpp | vs PyTorch/HF | Top Helper | Helper Share | Top Operator | Operator Share | Fallback | Fallback Count |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        format_delta = lambda value: "n/a" if value is None else f"{float(value):+.2f}%"
        format_percent = lambda value: "n/a" if value is None else f"{float(value):.2f}%"
        format_optional = lambda value: "n/a" if value is None else str(value)
        lines.append(
            f"| {row['implementation']} | {row['backend']} | {row.get('decodeMode', 'decode')} | "
            f"{row.get('config', 'n/a')} | "
            f"{float(row['msPerToken']):.4f} | "
            f"{float(row['tokensPerSecond']):.3f} | {format_delta(row['vsLlamaCppPercent'])} | "
            f"{format_delta(row['vsPyTorchPercent'])} | {row.get('topHelper') or 'n/a'} | "
            f"{format_percent(row.get('helperSharePercent'))} | {row.get('topOperator') or 'n/a'} | "
            f"{format_percent(row.get('operatorSharePercent'))} | {format_optional(row['fallbackUsed'])} | "
            f"{format_optional(row.get('fallbackCount'))} |"
        )
    (output_dir / "gguf_decode_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--litenn-smoke-report", action="append", type=Path, default=[])
    parser.add_argument("--litenn-profile-summary", action="append", type=Path, default=[])
    parser.add_argument("--llama-bench-json", action="append", type=Path, default=[])
    parser.add_argument("--pytorch-json", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    rows = [litenn_row(path) for path in args.litenn_smoke_report]
    rows.extend(litenn_profile_summary_row(path) for path in args.litenn_profile_summary)
    for path in args.llama_bench_json:
        rows.extend(llama_rows(path))
    for path in args.pytorch_json:
        rows.extend(pytorch_rows(path))
    if not rows:
        raise SystemExit("provide at least one benchmark evidence file")
    add_baseline_deltas(rows)
    write_outputs(rows, args.output_dir)
    print(json.dumps({"rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
