#!/usr/bin/env python3
"""Decide whether GGUF/Qwen evidence is sufficient for a production claim."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_step(report: dict[str, object], name: str) -> dict[str, object] | None:
    steps = report.get("steps")
    if not isinstance(steps, list):
        return None
    return next((step for step in steps if isinstance(step, dict) and step.get("name") == name), None)


def load_step_report(step: dict[str, object] | None, base: Path) -> dict[str, object] | None:
    if step is None or step.get("returncode") != 0:
        return None
    stdout = Path(str(step.get("stdout", "")))
    if not stdout.is_file() and not stdout.is_absolute():
        stdout = base / stdout
    if not stdout.is_file():
        return None
    try:
        return load_json(stdout)
    except (json.JSONDecodeError, OSError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-report", required=True, type=Path)
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--required-backend", default="cuda-native")
    parser.add_argument("--require-prefill", action="store_true")
    parser.add_argument("--require-decode", action="store_true")
    parser.add_argument("--require-text", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    smoke = load_json(args.smoke_report)
    if smoke.get("schema") not in ("litenn.gguf_qwen_smoke.v1", "litenn.gguf_qwen_smoke.v2"):
        raise SystemExit("unsupported Qwen smoke report schema")
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    analyze = find_step(smoke, "analyze")
    check("compatibility", analyze is not None and analyze.get("returncode") == 0, "analyze step must pass")
    backend = str(smoke.get("backend_policy", ""))
    check("backend", backend == args.required_backend, f"required={args.required_backend} actual={backend}")
    check("smokeFallback", smoke.get("fallback_used") is False, "smoke report must explicitly report no fallback")

    parity_requirements = (
        ("prefillParity", "compare_prefill_logits", args.require_prefill),
        ("decodeParity", "compare_decode_logits", args.require_decode),
        ("generatedText", "compare_generation_text", args.require_text),
    )
    for check_name, step_name, required in parity_requirements:
        if not required:
            continue
        nested = load_step_report(find_step(smoke, step_name), args.smoke_report.parent)
        check(check_name, nested is not None and nested.get("passed") is True, f"{step_name} must report passed=true")

    if args.required_backend == "cuda-native":
        if args.artifact_manifest is None:
            check("fallback", False, "CUDA production gate requires --artifact-manifest")
        else:
            artifact = load_json(args.artifact_manifest)
            cuda = artifact.get("cuda")
            valid_cuda = isinstance(cuda, dict) and cuda.get("actualBackend") == "cuda_native"
            no_fallback = isinstance(cuda, dict) and cuda.get("fallbackUsed") is False
            check("cudaArtifact", valid_cuda, "stateful artifact must report cuda_native")
            check("fallback", no_fallback, "CPU bridge fallback must be explicitly absent")

    passed = all(bool(item["passed"]) for item in checks)
    report = {
        "schema": "litenn.gguf_production_gate.v1",
        "smokeReport": str(args.smoke_report),
        "artifactManifest": str(args.artifact_manifest) if args.artifact_manifest is not None else None,
        "requiredBackend": args.required_backend,
        "passed": passed,
        "checks": checks,
    }
    output = args.output if args.output is not None else args.smoke_report.parent / "production_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
