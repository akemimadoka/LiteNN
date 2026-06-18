#!/usr/bin/env python3
"""Build a mobile-oriented performance and memory baseline report.

Examples:
  python311 scripts/mobile_baseline_report.py \
    --benchmark-json benchmark/results/litenn_mobile.json \
    --size-json build/mobile_size_report.json \
    --markdown build/mobile_baseline.md \
    --json build/mobile_baseline.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BENCH_RE = re.compile(r"^(?P<backend>[^/]+)/(?P<model>.+)/batch:(?P<batch>\d+)(?:/|$)")

MODEL_GROUPS = (
    ("mlp", ("MLP(",)),
    ("cnn", ("Conv", "Pool", "CNN")),
    ("tiny-transformer", ("Transformer", "Attention", "Softmax", "LayerNorm", "RMSNorm")),
)


@dataclass(frozen=True)
class BenchEntry:
    group: str
    model: str
    batch: int
    backend: str
    ms: float
    samples_per_second: float | None


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def classify_model(model: str) -> str | None:
    for group, needles in MODEL_GROUPS:
        if any(needle in model for needle in needles):
            return group
    return None


def read_benchmark_json(paths: Iterable[Path]) -> list[BenchEntry]:
    entries: list[BenchEntry] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for raw in data.get("benchmarks", []):
            name = str(raw.get("name", ""))
            if not name.endswith("/real_time"):
                continue
            match = BENCH_RE.match(name)
            if match is None:
                continue
            model = match.group("model")
            group = classify_model(model)
            if group is None:
                continue

            real_time = float(raw["real_time"])
            unit = raw.get("time_unit", "ms")
            if unit == "us":
                ms = real_time / 1000.0
            elif unit == "ns":
                ms = real_time / 1_000_000.0
            elif unit == "ms":
                ms = real_time
            else:
                raise ValueError(f"Unsupported Google Benchmark time unit '{unit}' in {path}")

            batch = int(match.group("batch"))
            entries.append(
                BenchEntry(
                    group=group,
                    model=model,
                    batch=batch,
                    backend=match.group("backend"),
                    ms=ms,
                    samples_per_second=(batch * 1000.0 / ms) if ms > 0.0 else None,
                )
            )
    entries.sort(key=lambda entry: (entry.group, entry.model, entry.batch, entry.backend))
    return entries


def merge_size_reports(paths: Iterable[Path]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for category, size in data.get("totals", {}).items():
            totals[category] = totals.get(category, 0) + int(size)
    return totals


def write_markdown(benchmarks: list[BenchEntry], sizes: dict[str, int]) -> str:
    lines = [
        "# LiteNN Mobile Baseline Report",
        "",
        "## Size",
        "",
        "| Category | Total |",
        "| --- | ---: |",
    ]
    for category in sorted(sizes):
        lines.append(f"| {category} | {human_size(sizes[category])} |")
    if not sizes:
        lines.append("| none | 0 B |")

    lines.extend(
        [
            "",
            "## Performance",
            "",
            "| Group | Model | Batch | Backend | ms/batch | samples/sec |",
            "| --- | --- | ---: | --- | ---: | ---: |",
        ]
    )
    for entry in benchmarks:
        throughput = "" if entry.samples_per_second is None else f"{entry.samples_per_second:.2f}"
        lines.append(
            f"| {entry.group} | `{entry.model}` | {entry.batch} | `{entry.backend}` | {entry.ms:.6f} | {throughput} |"
        )
    if not benchmarks:
        lines.append("| none | - | 0 | - | 0 | 0 |")
    lines.append("")
    return "\n".join(lines)


def write_json(benchmarks: list[BenchEntry], sizes: dict[str, int]) -> dict[str, object]:
    return {
        "sizes": sizes,
        "benchmarks": [
            {
                "group": entry.group,
                "model": entry.model,
                "batch": entry.batch,
                "backend": entry.backend,
                "ms": entry.ms,
                "samples_per_second": entry.samples_per_second,
            }
            for entry in benchmarks
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-json", action="append", default=[], type=Path, help="Google Benchmark JSON input")
    parser.add_argument("--size-json", action="append", default=[], type=Path, help="mobile_size_report.py JSON input")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown output")
    parser.add_argument("--json", type=Path, help="Optional JSON output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmarks = read_benchmark_json(args.benchmark_json)
    sizes = merge_size_reports(args.size_json)
    markdown = write_markdown(benchmarks, sizes)
    print(markdown)

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(write_json(benchmarks, sizes), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
