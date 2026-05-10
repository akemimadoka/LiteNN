#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkStat:
    name: str
    time_value: float
    time_unit: str
    samples_per_second: float | None


@dataclass(frozen=True)
class ComparisonRow:
    name: str
    baseline: BenchmarkStat
    contender: BenchmarkStat

    @property
    def delta_percent(self) -> float:
        if self.baseline.time_value == 0.0:
            return 0.0
        return (self.contender.time_value - self.baseline.time_value) / self.baseline.time_value * 100.0

    @property
    def status(self) -> str:
        if self.delta_percent >= 1.0:
            return "slower"
        if self.delta_percent <= -1.0:
            return "faster"
        return "stable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LiteNN google benchmark JSON outputs.")
    parser.add_argument("--baseline", required=True, help="Path to the baseline benchmark JSON file.")
    parser.add_argument("--contender", required=True, help="Path to the contender benchmark JSON file.")
    parser.add_argument("--baseline-label", default="baseline", help="Display label for the baseline run.")
    parser.add_argument("--contender-label", default="contender", help="Display label for the contender run.")
    parser.add_argument("--output", required=True, help="Markdown file to write.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def entry_metric(entry: dict[str, Any]) -> float:
    if "real_time" in entry:
        return float(entry["real_time"])
    return float(entry["cpu_time"])


def entry_counter(entry: dict[str, Any], key: str) -> float | None:
    direct_value = entry.get(key)
    if isinstance(direct_value, (int, float)):
        return float(direct_value)

    counters = entry.get("counters")
    if not isinstance(counters, dict):
        return None

    counter_value = counters.get(key)
    if isinstance(counter_value, (int, float)):
        return float(counter_value)
    if isinstance(counter_value, dict):
        nested_value = counter_value.get("value")
        if isinstance(nested_value, (int, float)):
            return float(nested_value)
    return None


def collect_stats(path: Path) -> dict[str, BenchmarkStat]:
    payload = load_json(path)
    entries = payload.get("benchmarks")
    if not isinstance(entries, list):
        raise ValueError(f"{path} does not contain a benchmark list")

    aggregate_means: dict[str, BenchmarkStat] = {}
    samples: dict[str, list[float]] = defaultdict(list)
    units: dict[str, str] = {}
    rates: dict[str, float | None] = {}

    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            continue
        name = raw_entry.get("name")
        if not isinstance(name, str):
            continue
        if name.endswith(("_median", "_stddev", "_cv")):
            continue

        run_type = raw_entry.get("run_type")
        if run_type == "aggregate":
            if raw_entry.get("aggregate_name") != "mean":
                continue
            aggregate_means[name] = BenchmarkStat(
                name=name,
                time_value=entry_metric(raw_entry),
                time_unit=str(raw_entry.get("time_unit", "ns")),
                samples_per_second=entry_counter(raw_entry, "samples_per_second"),
            )
            continue

        samples[name].append(entry_metric(raw_entry))
        units.setdefault(name, str(raw_entry.get("time_unit", "ns")))
        if name not in rates:
            rates[name] = entry_counter(raw_entry, "samples_per_second")

    results = dict(aggregate_means)
    for name, values in samples.items():
        if name in results or not values:
            continue
        results[name] = BenchmarkStat(
            name=name,
            time_value=sum(values) / len(values),
            time_unit=units.get(name, "ns"),
            samples_per_second=rates.get(name),
        )
    return results


def format_rate(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:,.0f}"


def format_markdown(rows: list[ComparisonRow], new_names: list[str], missing_names: list[str],
                    baseline_label: str, contender_label: str) -> str:
    slower_count = sum(1 for row in rows if row.status == "slower")
    faster_count = sum(1 for row in rows if row.status == "faster")
    stable_count = len(rows) - slower_count - faster_count

    lines = [
        "## Benchmark Diff",
        "",
        f"Baseline: {baseline_label}",
        f"Contender: {contender_label}",
        "",
        f"Matched benchmarks: {len(rows)}",
        f"Slower: {slower_count}",
        f"Faster: {faster_count}",
        f"Stable: {stable_count}",
        f"New benchmarks: {len(new_names)}",
        f"Missing benchmarks: {len(missing_names)}",
        "",
        "| Benchmark | Baseline | Contender | Delta | Status | Baseline samples/s | Contender samples/s |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]

    for row in rows:
        if row.baseline.time_unit != row.contender.time_unit:
            raise ValueError(
                f"Mismatched time units for {row.name}: {row.baseline.time_unit} vs {row.contender.time_unit}"
            )
        unit = row.baseline.time_unit
        lines.append(
            "| {name} | {baseline:.4f} {unit} | {contender:.4f} {unit} | {delta:+.2f}% | {status} | {baseline_rate} | {contender_rate} |".format(
                name=row.name,
                baseline=row.baseline.time_value,
                contender=row.contender.time_value,
                unit=unit,
                delta=row.delta_percent,
                status=row.status,
                baseline_rate=format_rate(row.baseline.samples_per_second),
                contender_rate=format_rate(row.contender.samples_per_second),
            )
        )

    if new_names:
        lines.extend(["", "### New benchmarks", ""])
        lines.extend(f"- {name}" for name in new_names)

    if missing_names:
        lines.extend(["", "### Missing benchmarks", ""])
        lines.extend(f"- {name}" for name in missing_names)

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    baseline = collect_stats(Path(args.baseline))
    contender = collect_stats(Path(args.contender))

    matched_names = sorted(set(baseline) & set(contender))
    rows = [
        ComparisonRow(name=name, baseline=baseline[name], contender=contender[name])
        for name in matched_names
    ]
    rows.sort(key=lambda row: (-row.delta_percent, row.name))

    new_names = sorted(set(contender) - set(baseline))
    missing_names = sorted(set(baseline) - set(contender))

    markdown = format_markdown(
        rows=rows,
        new_names=new_names,
        missing_names=missing_names,
        baseline_label=args.baseline_label,
        contender_label=args.contender_label,
    )
    output_path = Path(args.output)
    output_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
