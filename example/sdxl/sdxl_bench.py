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


def tail(text: str, limit: int = 1200) -> str:
    return text[-limit:]


def run_step(name: str, command: list[str], *, cwd: Path | None = None) -> StepResult:
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    elapsed = time.perf_counter() - start
    return StepResult(
        name=name,
        seconds=elapsed,
        returncode=completed.returncode,
        stdout_tail=tail(completed.stdout),
        stderr_tail=tail(completed.stderr),
    )


def write_markdown(path: Path, results: list[StepResult]) -> None:
    lines = [
        "# SDXL Smoke Benchmark",
        "",
        "| Stage | Status | Time (ms) |",
        "| --- | ---: | ---: |",
    ]
    for result in results:
        status = "ok" if result.returncode == 0 else f"failed ({result.returncode})"
        lines.append(f"| {result.name} | {status} | {result.seconds * 1000.0:.3f} |")
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
        }
        for result in results
    ]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path, help="Path to litenn_sdxl_example")
    parser.add_argument("--probe-script", default=Path(__file__).with_name("sdxl_manifest_probe.py"), type=Path)
    parser.add_argument("--config", required=True, type=Path, help="SDXL generative-models YAML")
    parser.add_argument("--safetensors", required=True, type=Path, help="SDXL safetensors checkpoint")
    parser.add_argument("--workdir", required=True, type=Path, help="Output directory for generated benchmark artifacts")
    parser.add_argument("--probe", default="unet-euler-smoke")
    parser.add_argument("--height", default=64, type=int)
    parser.add_argument("--width", default=64, type=int)
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--steps", default=1, type=int, help="Euler steps for the invocation benchmark")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--symbol-prefix", default="litenn_sdxl_module")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"), help="Compiler/linker used for shared library")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    manifest = args.workdir / f"{args.probe}_manifest.json"
    graph = args.workdir / f"{args.probe}.ltnn"
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
                "--emit-probe-manifest",
                str(manifest),
            ],
        ),
        (
            "import+serialize",
            [str(args.exe), "--import", str(manifest), str(args.safetensors), str(graph), "--allow-extra-tensors"],
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

    for name, command in steps:
        result = run_step(name, command)
        results.append(result)
        print(f"{name}: {'ok' if result.returncode == 0 else 'failed'} {result.seconds * 1000.0:.3f} ms")
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
