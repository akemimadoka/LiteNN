#!/usr/bin/env python3
"""Benchmark the SDXL LiteNN smoke pipeline by stage.

The script measures the current import/AOT packaging path:
manifest generation, safetensors import, graph serialization, CPU AOT object
emission, shared-library link, DLL/SO load, and one Euler denoise step.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StepResult:
    name: str
    seconds: float
    returncode: int
    stdout_tail: str
    stderr_tail: str
    metrics: dict[str, object] | None = None


def tail(text: str, limit: int = 1200) -> str:
    return text[-limit:]


def run_step(
    name: str,
    command: list[str],
    *,
    cwd: Path | None = None,
    metrics_path: Path | None = None,
) -> StepResult:
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    elapsed = time.perf_counter() - start
    metrics = None
    if metrics_path is not None and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return StepResult(
        name=name,
        seconds=elapsed,
        returncode=completed.returncode,
        stdout_tail=tail(completed.stdout),
        stderr_tail=tail(completed.stderr),
        metrics=metrics,
    )


def write_markdown(path: Path, results: list[StepResult]) -> None:
    lines = [
        "# SDXL Smoke Benchmark",
        "",
        "| Stage | Status | Wall Time (ms) | Backend | Compile (ms) | Load (ms) | Run Mean (ms) | Artifact MiB | IO MiB |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        status = "ok" if result.returncode == 0 else f"failed ({result.returncode})"
        metrics = result.metrics or {}
        if metrics.get("status") == "skipped":
            status = "skipped"
        artifact_mib = (
            (float(metrics.get("rodata_bytes", 0)) + float(metrics.get("instruction_bytes", 0))) / (1024.0 * 1024.0)
            if metrics
            else 0.0
        )
        io_mib = (
            (float(metrics.get("input_bytes", 0)) + float(metrics.get("output_bytes", 0))) / (1024.0 * 1024.0)
            if metrics
            else 0.0
        )
        lines.append(
            "| {} | {} | {:.3f} | {} | {} | {} | {} | {} | {} |".format(
                result.name,
                status,
                result.seconds * 1000.0,
                metrics.get("backend", "") if metrics else "",
                f"{float(metrics.get('compile_ms', 0.0)):.3f}" if metrics else "",
                f"{float(metrics.get('load_ms', 0.0)):.3f}" if metrics else "",
                f"{float(metrics.get('run_mean_ms', 0.0)):.3f}" if metrics else "",
                f"{artifact_mib:.3f}" if metrics else "",
                f"{io_mib:.3f}" if metrics else "",
            )
        )
        if metrics.get("message"):
            lines.append(f"\nNote for `{result.name}`: {metrics['message']}\n")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, results: list[StepResult]) -> None:
    payload = [
        {
            "name": result.name,
            "seconds": result.seconds,
            "returncode": result.returncode,
            "stdout_tail": result.stdout_tail,
            "stderr_tail": result.stderr_tail,
            "metrics": result.metrics,
        }
        for result in results
    ]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path, help="Path to litenn_sdxl_example")
    parser.add_argument("--probe-script", default=Path(__file__).with_name("sdxl_manifest_probe.py"), type=Path)
    parser.add_argument("--input-script", default=Path(__file__).with_name("sdxl_write_inputs.py"), type=Path)
    parser.add_argument("--config", required=True, type=Path, help="SDXL generative-models YAML")
    parser.add_argument("--safetensors", required=True, type=Path, help="SDXL safetensors checkpoint")
    parser.add_argument("--workdir", required=True, type=Path, help="Output directory for generated benchmark artifacts")
    parser.add_argument("--probe", default="unet-euler-smoke")
    parser.add_argument("--height", default=64, type=int)
    parser.add_argument("--width", default=64, type=int)
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--context-tokens", default=77, type=int)
    parser.add_argument("--context-width", default=2048, type=int)
    parser.add_argument("--vector-width", default=2816, type=int)
    parser.add_argument("--latent-channels", default=4, type=int)
    parser.add_argument("--steps", default=1, type=int, help="Euler steps for the invocation benchmark")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--benchmark-devices", default="cpu,cuda", help="Comma-separated devices for AOT rows")
    parser.add_argument("--benchmark-warmup", default=1, type=int)
    parser.add_argument("--benchmark-iterations", default=1, type=int)
    parser.add_argument(
        "--vae-mid-attention-policy",
        choices=["auto", "force", "skip"],
        default="auto",
        help="Forwarded to sdxl_manifest_probe.py for vae-decode-full",
    )
    parser.add_argument(
        "--vae-attention-max-mib",
        default=512,
        type=int,
        help="Forwarded dense VAE mid-attention workspace limit in MiB",
    )
    parser.add_argument("--symbol-prefix", default="litenn_sdxl_module")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"), help="Compiler/linker used for shared library")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.vae_attention_max_mib <= 0:
        raise ValueError("--vae-attention-max-mib must be positive")
    args.workdir.mkdir(parents=True, exist_ok=True)

    manifest = args.workdir / f"{args.probe}_manifest.json"
    graph = args.workdir / f"{args.probe}.ltnn.json"
    inputs = args.workdir / f"{args.probe}_inputs.safetensors"
    obj = args.workdir / f"{args.probe}.obj"
    if os.name == "nt":
        library = args.workdir / f"{args.probe}.dll"
        def_file = args.workdir / f"{args.symbol_prefix}_exports.def"
        link_command = [args.cxx, "-shared", str(obj), str(def_file), "-o", str(library)]
    else:
        library = args.workdir / f"lib{args.probe}.so"
        link_command = [args.cxx, "-shared", str(obj), "-o", str(library)]

    results: list[StepResult] = []
    steps = [
        (
            "manifest",
            [
                sys.executable,
                str(args.probe_script),
                "--config",
                str(args.config),
                "--safetensors",
                str(args.safetensors),
                "--probe",
                args.probe,
                "--batch",
                str(args.batch),
                "--height",
                str(args.height),
                "--width",
                str(args.width),
                "--context-tokens",
                str(args.context_tokens),
                "--vae-mid-attention-policy",
                args.vae_mid_attention_policy,
                "--vae-attention-max-mib",
                str(args.vae_attention_max_mib),
                "--emit-probe-manifest",
                str(manifest),
            ],
        ),
        (
            "import+serialize",
            [str(args.exe), "--import-package", str(manifest), str(args.safetensors), str(graph), "--allow-extra-tensors"],
        ),
        (
            "write-inputs",
            [
                sys.executable,
                str(args.input_script),
                "--probe",
                args.probe,
                "--batch",
                str(args.batch),
                "--height",
                str(args.height),
                "--width",
                str(args.width),
                "--latent-channels",
                str(args.latent_channels),
                "--context-tokens",
                str(args.context_tokens),
                "--context-width",
                str(args.context_width),
                "--vector-width",
                str(args.vector_width),
                "--output",
                str(inputs),
            ],
        ),
        ("cpu-aot-object", [str(args.exe), "--compile-object", str(graph), str(obj), args.symbol_prefix]),
        ("shared-library-link", link_command),
        ("dll-load-smoke", [str(args.exe), "--load-dll", str(library), args.symbol_prefix]),
        (
            "one-denoise-step",
            [
                str(args.exe),
                "--sample-euler",
                str(library),
                args.symbol_prefix,
                "--steps",
                str(args.steps),
                "--seed",
                str(args.seed),
            ],
        ),
    ]

    devices = [device.strip() for device in args.benchmark_devices.split(",") if device.strip()]
    for device in devices:
        if device not in {"cpu", "cuda"}:
            raise ValueError(f"unsupported benchmark device {device!r}")
        metrics_path = args.workdir / f"{args.probe}_{device}_aot_benchmark.json"
        steps.insert(
            3 + devices.index(device),
            (
                f"{device}-aot-denoise-step",
                [
                    str(args.exe),
                    "--benchmark-model-with-inputs",
                    str(graph),
                    str(inputs),
                    "--device",
                    device,
                    "--warmup",
                    str(args.benchmark_warmup),
                    "--iterations",
                    str(args.benchmark_iterations),
                    "--json",
                    str(metrics_path),
                ],
                metrics_path,
            ),
        )

    for step in steps:
        if len(step) == 2:
            name, command = step
            metrics_path = None
        else:
            name, command, metrics_path = step
        result = run_step(name, command, metrics_path=metrics_path)
        results.append(result)
        print(f"{name}: {'ok' if result.returncode == 0 else 'failed'} {result.seconds * 1000.0:.3f} ms")
        if result.metrics:
            metric_status = result.metrics.get("status", "")
            metric_backend = result.metrics.get("backend", "")
            metric_run = result.metrics.get("run_mean_ms", 0.0)
            print(f"  metrics: status={metric_status} backend={metric_backend} run_mean_ms={metric_run}")
        if result.returncode != 0:
            if result.stderr_tail:
                print(result.stderr_tail)
            break

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.json_out, results)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.markdown_out, results)
    return 0 if all(result.returncode == 0 for result in results) and len(results) == len(steps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
