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
    tokens_per_second = token_count * 1000.0 / run_ms
    return {
        "implementation": "LiteNN",
        "backend": report.get("backend_policy"),
        "tokens": token_count,
        "totalMs": run_ms,
        "msPerToken": run_ms / token_count,
        "tokensPerSecond": tokens_per_second,
        "fallbackUsed": report.get("fallback_used"),
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
                "tokens": int(entry["n_gen"]),
                "totalMs": int(entry["n_gen"]) * 1000.0 / tokens_per_second,
                "msPerToken": 1000.0 / tokens_per_second,
                "tokensPerSecond": tokens_per_second,
                "fallbackUsed": False,
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
                "tokens": entry.get("tokens"),
                "totalMs": entry.get("totalMs"),
                "msPerToken": 1000.0 / tokens_per_second,
                "tokensPerSecond": tokens_per_second,
                "fallbackUsed": entry.get("fallbackUsed", False),
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
    fields = list(rows[0].keys()) if rows else []
    with (output_dir / "gguf_decode_compare.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Implementation | Backend | ms/token | token/s | vs llama.cpp | vs PyTorch/HF | Fallback |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        format_delta = lambda value: "n/a" if value is None else f"{float(value):+.2f}%"
        lines.append(
            f"| {row['implementation']} | {row['backend']} | {float(row['msPerToken']):.4f} | "
            f"{float(row['tokensPerSecond']):.3f} | {format_delta(row['vsLlamaCppPercent'])} | "
            f"{format_delta(row['vsPyTorchPercent'])} | {row['fallbackUsed']} |"
        )
    (output_dir / "gguf_decode_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--litenn-smoke-report", action="append", type=Path, default=[])
    parser.add_argument("--llama-bench-json", action="append", type=Path, default=[])
    parser.add_argument("--pytorch-json", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    rows = [litenn_row(path) for path in args.litenn_smoke_report]
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
