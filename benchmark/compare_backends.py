"""Generate backend comparison tables from LiteNN/ggml/Vulkan/CUDA and PyTorch benchmark outputs.

Typical usage:
  python311 benchmark/compare_backends.py \
    --litenn-json benchmark/results/litenn_ggml_full_2026-05-19.json \
    --pytorch-text benchmark/results/pytorch_all_default_2026-05-19.txt \
    --pytorch-cpu1-text benchmark/results/pytorch_cpu_threads1_2026-05-19.txt \
    --out-md benchmark/results/backend_comparison.md \
    --out-csv benchmark/results/backend_comparison.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Result:
    model: str
    batch: int
    backend: str
    ms: float


BENCH_RE = re.compile(r"^(?P<backend>[^/]+)/(?P<model>.+)/batch:(?P<batch>\d+)(?:/|$)")
PYTORCH_ROW_RE = re.compile(
    r"^(?P<model>.+?)\s+(?P<batch>\d+)\s+(?P<ms>[0-9.]+)ms\s+(?P<throughput>[0-9.]+)/s$"
)
DEFAULT_MODEL_SET = {
    "Linear(784->10)",
    "MLP(784->128->10)",
    "MLP(784->512->256->10)",
}


def canonical_backend(name: str) -> str:
    aliases = {
        "GGMLRunIntoT1": "GGML 1T",
        "GGMLRunIntoT16": "GGML 16T",
        "LlamaCppGGMLT1": "GGML 1T",
        "LlamaCppGGMLT16": "GGML 16T",
        "CUDADeviceMatMul": "LiteNN CUDA DeviceMatMul",
        "CUDACPUFallbackRunInto": "LiteNN CUDA CPU Fallback",
        "CUDANativeRunInto": "LiteNN CUDA Native",
        "CUDANativeGraphRunInto": "LiteNN CUDA Graph",
        "VulkanNativeRunInto": "LiteNN Vulkan Native",
        "VulkanNativeGraphRunInto": "LiteNN Vulkan Graph",
        "VulkanNativeGraphDeviceLocalRunInto": "LiteNN Vulkan Graph DeviceLocal",
        "VulkanNativeManualPipeline": "LiteNN Vulkan ManualPipeline",
        "AOTRun": "LiteNN CPU AOT Run",
        "AOTRunInto": "LiteNN CPU AOT RunInto",
        "AOTFastPathRunIntoT1": "LiteNN CPU Fast T1",
        "AOTFastPathRunIntoT16": "LiteNN CPU Fast T16",
        "Interpreter": "LiteNN Interpreter",
    }
    return aliases.get(name, name)


def should_include_model(model: str, include_all_models: bool) -> bool:
    return include_all_models or model in DEFAULT_MODEL_SET


def read_litenn_json(path: Path, *, include_all_models: bool = False) -> list[Result]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results: list[Result] = []
    for entry in data.get("benchmarks", []):
        name = str(entry.get("name", ""))
        if not name.endswith("/real_time"):
            continue
        match = BENCH_RE.match(name)
        if match is None:
            continue
        model = match.group("model")
        if not should_include_model(model, include_all_models):
            continue
        time_unit = entry.get("time_unit", "ms")
        real_time = float(entry["real_time"])
        if time_unit == "us":
            real_time /= 1000.0
        elif time_unit == "ns":
            real_time /= 1_000_000.0
        elif time_unit != "ms":
            raise ValueError(f"Unsupported Google Benchmark time unit '{time_unit}' in {path}")
        results.append(
            Result(
                model=model,
                batch=int(match.group("batch")),
                backend=canonical_backend(match.group("backend")),
                ms=real_time,
            )
        )
    return results


def read_pytorch_text(path: Path, *, cpu1: bool = False, include_all_models: bool = False) -> list[Result]:
    results: list[Result] = []
    backend: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[CPU]"):
            backend = "PyTorch CPU 1T" if cpu1 else "PyTorch CPU"
            continue
        if line.startswith("[CUDA:"):
            backend = "PyTorch CUDA"
            continue
        match = PYTORCH_ROW_RE.match(line)
        if match is None or backend is None:
            continue
        model = match.group("model").strip()
        if not should_include_model(model, include_all_models):
            continue
        results.append(
            Result(
                model=model,
                batch=int(match.group("batch")),
                backend=backend,
                ms=float(match.group("ms")),
            )
        )
    return results


def pct_delta(value: float, baseline: float | None) -> float | None:
    if baseline is None or baseline == 0:
        return None
    return (value / baseline - 1.0) * 100.0


def format_ms(value: float | None) -> str:
    if value is None:
        return ""
    if value < 0.01:
        return f"{value:.4f}"
    if value < 1:
        return f"{value:.3f}"
    return f"{value:.2f}"


def format_pct(value: float | None) -> str:
    return "" if value is None else f"{value:+.1f}%"


def sort_key(row: tuple[str, int]) -> tuple[int, str, int]:
    model_order = {
        "Linear(784->10)": 0,
        "MLP(784->128->10)": 1,
        "MLP(784->512->256->10)": 2,
    }
    model, batch = row
    return (model_order.get(model, 100), model, batch)


def backend_sort_key(backend: str) -> tuple[int, str]:
    order = {
        "GGML 1T": 0,
        "GGML 16T": 1,
        "PyTorch CPU 1T": 2,
        "PyTorch CPU": 3,
        "PyTorch CUDA": 4,
        "LiteNN Interpreter": 5,
        "LiteNN CPU AOT Run": 6,
        "LiteNN CPU AOT RunInto": 7,
        "LiteNN CPU Fast T1": 8,
        "LiteNN CPU Fast T16": 9,
        "LiteNN CUDA CPU Fallback": 10,
        "LiteNN CUDA Native": 11,
        "LiteNN CUDA Graph": 12,
        "LiteNN CUDA DeviceMatMul": 13,
        "LiteNN Vulkan Native": 14,
        "LiteNN Vulkan Graph": 15,
        "LiteNN Vulkan Graph DeviceLocal": 16,
        "LiteNN Vulkan ManualPipeline": 17,
    }
    return (order.get(backend, 100), backend)


def build_table(results: list[Result]) -> tuple[list[tuple[str, int]], list[str], dict[tuple[str, int, str], float]]:
    values: dict[tuple[str, int, str], float] = {}
    rows: set[tuple[str, int]] = set()
    backends: set[str] = set()
    for result in results:
        key = (result.model, result.batch, result.backend)
        if key not in values or result.ms < values[key]:
            values[key] = result.ms
        rows.add((result.model, result.batch))
        backends.add(result.backend)
    return sorted(rows, key=sort_key), sorted(backends, key=backend_sort_key), values


def write_csv(path: Path, rows: list[tuple[str, int]], backends: list[str], values: dict[tuple[str, int, str], float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "batch",
                "backend",
                "ms_per_batch",
                "delta_vs_pytorch_cpu",
                "delta_vs_pytorch_cpu_1t",
                "delta_vs_pytorch_cuda",
                "delta_vs_ggml_1t",
            ]
        )
        for model, batch in rows:
            baselines = {
                "cpu": values.get((model, batch, "PyTorch CPU")),
                "cpu1": values.get((model, batch, "PyTorch CPU 1T")),
                "cuda": values.get((model, batch, "PyTorch CUDA")),
                "ggml": values.get((model, batch, "GGML 1T")),
            }
            for backend in backends:
                value = values.get((model, batch, backend))
                if value is None:
                    continue
                writer.writerow(
                    [
                        model,
                        batch,
                        backend,
                        f"{value:.9g}",
                        format_pct(pct_delta(value, baselines["cpu"])),
                        format_pct(pct_delta(value, baselines["cpu1"])),
                        format_pct(pct_delta(value, baselines["cuda"])),
                        format_pct(pct_delta(value, baselines["ggml"])),
                    ]
                )


def write_markdown(
    path: Path,
    rows: list[tuple[str, int]],
    backends: list[str],
    values: dict[tuple[str, int, str], float],
    sources: list[Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Backend Comparison",
        "",
        "- Values are wall-clock `ms/batch`; lower is better.",
        "- Percent deltas in the CSV use `(backend_ms / baseline_ms - 1) * 100`.",
        "- Source files:",
    ]
    lines.extend(f"  - `{source.as_posix()}`" for source in sources)
    lines.extend(["", "## ms/batch", ""])
    header = ["Model", "Batch", *backends]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|---|---:|" + "|".join("---:" for _ in backends) + "|")
    for model, batch in rows:
        cells = [model, str(batch)]
        cells.extend(format_ms(values.get((model, batch, backend))) for backend in backends)
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate backend comparison tables")
    parser.add_argument("--litenn-json", action="append", type=Path, default=[], help="Google Benchmark JSON file")
    parser.add_argument("--pytorch-text", action="append", type=Path, default=[], help="PyTorch benchmark text output")
    parser.add_argument(
        "--pytorch-cpu1-text",
        action="append",
        type=Path,
        default=[],
        help="PyTorch --threads 1 benchmark text output",
    )
    parser.add_argument("--out-md", type=Path, required=True, help="Markdown output path")
    parser.add_argument("--out-csv", type=Path, required=True, help="CSV output path")
    parser.add_argument(
        "--include-all-models",
        action="store_true",
        help="Include every parsed model-like benchmark row instead of only the standard inference model set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: list[Result] = []
    sources = [*args.litenn_json, *args.pytorch_text, *args.pytorch_cpu1_text]
    for path in args.litenn_json:
        results.extend(read_litenn_json(path, include_all_models=args.include_all_models))
    for path in args.pytorch_text:
        results.extend(read_pytorch_text(path, include_all_models=args.include_all_models))
    for path in args.pytorch_cpu1_text:
        results.extend(read_pytorch_text(path, cpu1=True, include_all_models=args.include_all_models))
    if not results:
        raise SystemExit("No benchmark rows were parsed")
    rows, backends, values = build_table(results)
    write_csv(args.out_csv, rows, backends, values)
    write_markdown(args.out_md, rows, backends, values, sources)
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
