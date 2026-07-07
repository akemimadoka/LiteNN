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
            f"q8k={int(bool(cpu_aot_options.get('q8k_staged_matmul', False)))},"
            f"prepack={int(bool(cpu_aot_options.get('ggml_prepacked_weights', False)))},"
            f"prepack_policy={cpu_aot_options.get('ggml_prepacked_weight_policy', 'default')}"
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
        "residualSharePercent": None,
        "topNode": None,
        "topNodeKind": None,
        "topFormat": None,
        "topActivation": None,
        "topResolvedThreads": None,
        "moduleRunMs": None,
        "moduleRunSharePercent": None,
        "moduleNonHelperMs": None,
        "moduleNonHelperSharePercent": None,
        "hostOverheadMs": None,
        "hostOverheadSharePercent": None,
        "source": str(path),
    }


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def helper_percent(total_ms: float, denominator_ms: float) -> float | None:
    return (total_ms * 100.0 / denominator_ms) if denominator_ms > 0.0 else None


def rank_totals(items: list[dict[str, object]], key_fields: tuple[str, ...]) -> list[dict[str, object]]:
    totals: dict[tuple[str, ...], dict[str, object]] = {}
    for item in items:
        key = tuple(str(item.get(field, "unknown")) for field in key_fields)
        total = totals.setdefault(
            key,
            {
                **{field: key[index] for index, field in enumerate(key_fields)},
                "calls": 0,
                "total_ms": 0.0,
            },
        )
        total["calls"] = as_int(total.get("calls")) + as_int(item.get("calls"))
        total["total_ms"] = as_float(total.get("total_ms")) + as_float(item.get("total_ms"))
    return sorted(totals.values(), key=lambda item: (-as_float(item.get("total_ms")), str(item)))


def litenn_profile_summary_rows(path: Path) -> list[dict[str, object]]:
    summary = load_json(path)
    if not isinstance(summary, dict) or "steps" not in summary:
        raise SystemExit(f"unsupported LiteNN GGUF decode summary: {path}")
    steps = summary.get("steps")
    helper_events = summary.get("helper_events")
    node_timings = summary.get("node_timings")
    if not isinstance(steps, list):
        raise SystemExit(f"LiteNN GGUF decode summary has no steps array: {path}")

    step_dicts = [step for step in steps if isinstance(step, dict)]
    helper_event_dicts = [helper for helper in helper_events if isinstance(helper, dict)] if isinstance(helper_events, list) else []
    node_timing_dicts = [node for node in node_timings if isinstance(node, dict)] if isinstance(node_timings, list) else []
    if not step_dicts:
        raise SystemExit(f"LiteNN GGUF decode summary must contain positive time and generated tokens: {path}")

    def make_row(label: str, selected_steps: list[dict[str, object]]) -> dict[str, object] | None:
        if not selected_steps:
            return None
        selected_step_ids = {as_int(step.get("step")) for step in selected_steps}
        total_step_ms = sum(as_float(step.get("step_ms")) for step in selected_steps)
        helper_ms = sum(as_float(step.get("helper_total_ms")) for step in selected_steps)
        residual_ms = sum(as_float(step.get("residual_ms")) for step in selected_steps)
        token_count = len(selected_steps) if label == "generation" else sum(
            1 for step in selected_steps if step.get("phase") == "generation"
        )
        if token_count <= 0:
            token_count = max((as_int(step.get("generated_tokens")) for step in selected_steps), default=0)
        if total_step_ms <= 0.0 or token_count <= 0:
            return None

        selected_helpers = [
            helper for helper in helper_event_dicts if as_int(helper.get("step")) in selected_step_ids
        ]
        helper_rows = rank_totals(selected_helpers, ("helper", "detail"))
        top_helper = helper_rows[0] if helper_rows else None
        top_helper_name = str(top_helper.get("helper", "unknown")) if isinstance(top_helper, dict) else None
        top_helper_share = (
            helper_percent(as_float(top_helper.get("total_ms")), total_step_ms)
            if isinstance(top_helper, dict)
            else None
        )

        operator_rows = rank_totals(selected_helpers, ("operator", "role"))
        top_operator = operator_rows[0] if operator_rows else None
        top_operator_name = (
            f"{top_operator.get('operator', 'unknown')}/{top_operator.get('role', 'unknown')}"
            if isinstance(top_operator, dict)
            else None
        )
        top_operator_share = (
            helper_percent(as_float(top_operator.get("total_ms")), total_step_ms)
            if isinstance(top_operator, dict)
            else None
        )

        selected_nodes = [node for node in node_timing_dicts if as_int(node.get("step")) in selected_step_ids]
        node_rows = rank_totals(selected_nodes, ("node_kind", "node_name"))
        top_node = node_rows[0] if node_rows else None
        top_node_name = str(top_node.get("node_name", "unknown")) if isinstance(top_node, dict) else None
        top_node_kind = str(top_node.get("node_kind", "unknown")) if isinstance(top_node, dict) else None
        top_node_event = max(selected_nodes, key=lambda node: as_float(node.get("total_ms")), default=None)
        module_run_ms = sum(as_float(step.get("module_run_ms")) for step in selected_steps if step.get("module_run_ms") is not None)
        module_non_helper_ms = sum(
            as_float(step.get("module_non_helper_ms"))
            for step in selected_steps
            if step.get("module_non_helper_ms") is not None
        )
        host_overhead_ms = sum(
            as_float(step.get("host_overhead_ms")) for step in selected_steps if step.get("host_overhead_ms") is not None
        )

        tokens_per_second = token_count * 1000.0 / total_step_ms
        return {
            "implementation": "LiteNN",
            "backend": "cpu-aot",
            "decodeMode": f"profile-summary-{label}",
            "config": "from-profile-bundle",
            "tokens": token_count,
            "totalMs": total_step_ms,
            "promptReplayMs": total_step_ms if label == "prompt_replay" else None,
            "generationMs": total_step_ms if label == "generation" else None,
            "msPerToken": total_step_ms / token_count,
            "tokensPerSecond": tokens_per_second,
            "fallbackUsed": None,
            "fallbackCount": None,
            "topHelper": top_helper_name,
            "helperSharePercent": top_helper_share,
            "topOperator": top_operator_name,
            "operatorSharePercent": top_operator_share,
            "residualSharePercent": helper_percent(residual_ms, total_step_ms),
            "helperTotalMs": helper_ms,
            "residualMs": residual_ms,
            "topNode": top_node_name,
            "topNodeKind": top_node_kind,
            "topFormat": (
                str(top_node_event.get("format"))
                if isinstance(top_node_event, dict) and top_node_event.get("format") is not None
                else None
            ),
            "topActivation": (
                str(top_node_event.get("activation"))
                if isinstance(top_node_event, dict) and top_node_event.get("activation") is not None
                else None
            ),
            "topResolvedThreads": (
                str(top_node_event.get("resolved_threads"))
                if isinstance(top_node_event, dict) and top_node_event.get("resolved_threads") is not None
                else None
            ),
            "moduleRunMs": module_run_ms if module_run_ms > 0.0 else None,
            "moduleRunSharePercent": helper_percent(module_run_ms, total_step_ms) if module_run_ms > 0.0 else None,
            "moduleNonHelperMs": module_non_helper_ms if module_non_helper_ms > 0.0 else None,
            "moduleNonHelperSharePercent": (
                helper_percent(module_non_helper_ms, total_step_ms) if module_non_helper_ms > 0.0 else None
            ),
            "hostOverheadMs": host_overhead_ms if host_overhead_ms > 0.0 else None,
            "hostOverheadSharePercent": helper_percent(host_overhead_ms, total_step_ms) if host_overhead_ms > 0.0 else None,
            "source": str(path),
        }

    rows = [
        make_row("all", step_dicts),
        make_row("prompt_replay", [step for step in step_dicts if step.get("phase") == "prompt_replay"]),
        make_row("generation", [step for step in step_dicts if step.get("phase") == "generation"]),
    ]
    return [row for row in rows if row is not None]


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
                "residualSharePercent": None,
                "topNode": None,
                "topNodeKind": None,
                "topFormat": None,
                "topActivation": None,
                "topResolvedThreads": None,
                "moduleRunMs": None,
                "moduleRunSharePercent": None,
                "moduleNonHelperMs": None,
                "moduleNonHelperSharePercent": None,
                "hostOverheadMs": None,
                "hostOverheadSharePercent": None,
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
                "residualSharePercent": entry.get("residualSharePercent"),
                "topNode": entry.get("topNode"),
                "topNodeKind": entry.get("topNodeKind"),
                "topFormat": entry.get("topFormat"),
                "topActivation": entry.get("topActivation"),
                "topResolvedThreads": entry.get("topResolvedThreads"),
                "moduleRunMs": entry.get("moduleRunMs"),
                "moduleRunSharePercent": entry.get("moduleRunSharePercent"),
                "moduleNonHelperMs": entry.get("moduleNonHelperMs"),
                "moduleNonHelperSharePercent": entry.get("moduleNonHelperSharePercent"),
                "hostOverheadMs": entry.get("hostOverheadMs"),
                "hostOverheadSharePercent": entry.get("hostOverheadSharePercent"),
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
        "helperTotalMs",
        "helperSharePercent",
        "topOperator",
        "operatorSharePercent",
        "topFormat",
        "topActivation",
        "topResolvedThreads",
        "residualMs",
        "residualSharePercent",
        "moduleRunMs",
        "moduleRunSharePercent",
        "moduleNonHelperMs",
        "moduleNonHelperSharePercent",
        "hostOverheadMs",
        "hostOverheadSharePercent",
        "topNode",
        "topNodeKind",
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
        "| Implementation | Backend | Mode | Config | ms/token | token/s | vs llama.cpp | vs PyTorch/HF | Top Helper | Helper ms | Helper Share | Top Node | Top Operator | Operator Share | Format | Activation | Threads | Module ms | Module Share | Module non-helper ms | Module non-helper share | Host Overhead ms | Host Overhead Share | Residual ms | Residual Share | Fallback | Fallback Count |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        format_delta = lambda value: "n/a" if value is None else f"{float(value):+.2f}%"
        format_percent = lambda value: "n/a" if value is None else f"{float(value):.2f}%"
        format_ms = lambda value: "n/a" if value is None else f"{float(value):.3f}"
        format_optional = lambda value: "n/a" if value is None else str(value)
        lines.append(
            f"| {row['implementation']} | {row['backend']} | {row.get('decodeMode', 'decode')} | "
            f"{row.get('config', 'n/a')} | "
            f"{float(row['msPerToken']):.4f} | "
            f"{float(row['tokensPerSecond']):.3f} | {format_delta(row['vsLlamaCppPercent'])} | "
            f"{format_delta(row['vsPyTorchPercent'])} | {row.get('topHelper') or 'n/a'} | "
            f"{format_ms(row.get('helperTotalMs'))} | "
            f"{format_percent(row.get('helperSharePercent'))} | {row.get('topNode') or 'n/a'} | "
            f"{row.get('topOperator') or 'n/a'} | "
            f"{format_percent(row.get('operatorSharePercent'))} | "
            f"{row.get('topFormat') or 'n/a'} | {row.get('topActivation') or 'n/a'} | "
            f"{row.get('topResolvedThreads') or 'n/a'} | {format_ms(row.get('moduleRunMs'))} | "
            f"{format_percent(row.get('moduleRunSharePercent'))} | "
            f"{format_ms(row.get('moduleNonHelperMs'))} | "
            f"{format_percent(row.get('moduleNonHelperSharePercent'))} | "
            f"{format_ms(row.get('hostOverheadMs'))} | "
            f"{format_percent(row.get('hostOverheadSharePercent'))} | {format_ms(row.get('residualMs'))} | "
            f"{format_percent(row.get('residualSharePercent'))} | "
            f"{format_optional(row['fallbackUsed'])} | "
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
    for path in args.litenn_profile_summary:
        rows.extend(litenn_profile_summary_rows(path))
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
