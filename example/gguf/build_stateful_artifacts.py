#!/usr/bin/env python3
"""Lower stateful GGUF decode and emit separated CPU/CUDA artifacts."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


BACKEND_RE = re.compile(r"\bbackend=(?P<backend>[a-z_]+)\b")


def discover_litenn(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"LiteNN GGUF tool does not exist: {explicit}")
        return explicit
    for name in ("litenn_gguf_convert.exe", "litenn_gguf_convert"):
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)
    root = Path(__file__).resolve().parents[2]
    for candidate in (
        root / "build-release" / "tools" / "gguf" / "litenn_gguf_convert.exe",
        root / "build-release" / "tools" / "gguf" / "litenn_gguf_convert",
        root / "build" / "tools" / "gguf" / "litenn_gguf_convert.exe",
        root / "build" / "tools" / "gguf" / "litenn_gguf_convert",
    ):
        if candidate.is_file():
            return candidate
    raise SystemExit("litenn_gguf_convert was not found; pass --litenn")


def run_step(name: str, command: list[str], out_dir: Path) -> dict[str, object]:
    completed = subprocess.run(command, text=True, capture_output=True)
    stdout_path = out_dir / f"{name}.stdout.txt"
    stderr_path = out_dir / f"{name}.stderr.txt"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    match = BACKEND_RE.search(completed.stdout)
    return {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "actualBackend": match.group("backend") if match else None,
    }


def require_ok(step: dict[str, object]) -> None:
    if int(step["returncode"]) != 0:
        raise SystemExit(f"step failed: {step['name']} (see {step['stderr']})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--litenn", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--past-length", type=int, default=0)
    parser.add_argument("--max-cache-length", type=int, required=True)
    parser.add_argument("--symbol-prefix", default="litenn_llama_decode")
    parser.add_argument(
        "--cuda-policy",
        choices=("disabled", "optional", "bridge-allowed", "native-required"),
        default="native-required",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.past_length < 0:
        raise SystemExit("--past-length must be non-negative")
    if args.max_cache_length <= args.past_length:
        raise SystemExit("--max-cache-length must be greater than --past-length")
    if not args.model.is_file():
        raise SystemExit(f"GGUF model does not exist: {args.model}")
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    litenn = discover_litenn(args.litenn)
    package = out_dir / "decode.stateful.ltnn"
    weights = out_dir / "decode.stateful.weights.bin"
    cpu_dir = out_dir / "cpu-separated"
    cuda_dir = out_dir / "cuda-separated"
    steps: list[dict[str, object]] = []

    lower = run_step(
        "lower_stateful",
        [
            str(litenn),
            "--lower-llama-decode-stateful",
            str(args.model),
            str(package),
            str(weights),
            str(args.past_length),
            str(args.max_cache_length),
        ],
        out_dir,
    )
    steps.append(lower)
    require_ok(lower)

    cpu = run_step(
        "compile_cpu_separated",
        [str(litenn), "--compile-cpu-separated", str(package), str(cpu_dir), args.symbol_prefix],
        out_dir,
    )
    steps.append(cpu)
    require_ok(cpu)
    if cpu["actualBackend"] != "cpu_native":
        raise SystemExit("CPU separated compile did not report cpu_native")

    cuda: dict[str, object] | None = None
    if args.cuda_policy != "disabled":
        cuda = run_step(
            "compile_cuda_separated",
            [str(litenn), "--compile-cuda-separated", str(package), str(cuda_dir), args.symbol_prefix],
            out_dir,
        )
        steps.append(cuda)
        if int(cuda["returncode"]) != 0 and args.cuda_policy != "optional":
            require_ok(cuda)
        if int(cuda["returncode"]) == 0:
            backend = cuda["actualBackend"]
            if backend not in ("cuda_native", "cpu_native"):
                raise SystemExit("CUDA separated compile did not report a recognized backend")
            if args.cuda_policy == "native-required" and backend != "cuda_native":
                raise SystemExit("CUDA native artifact was required but the compiler emitted a CPU bridge artifact")

    cuda_backend = cuda.get("actualBackend") if cuda is not None and int(cuda["returncode"]) == 0 else None
    manifest = {
        "schema": "litenn.gguf_stateful_artifacts.v1",
        "model": str(args.model),
        "package": str(package),
        "externalWeights": str(weights),
        "pastLength": args.past_length,
        "maxCacheLength": args.max_cache_length,
        "symbolPrefix": args.symbol_prefix,
        "cpu": {"actualBackend": cpu["actualBackend"], "directory": str(cpu_dir)},
        "cuda": {
            "policy": args.cuda_policy,
            "available": cuda_backend is not None,
            "actualBackend": cuda_backend,
            "fallbackUsed": cuda_backend == "cpu_native",
            "directory": str(cuda_dir) if cuda_backend is not None else None,
        },
        "steps": steps,
    }
    manifest_path = out_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
