#!/usr/bin/env python3
"""Compare LiteNN prefill logits against a llama.cpp golden capture.

Examples:
  python311 scripts/gguf_compare_llamacpp_logits.py \
    --manifest build/gguf_golden/hello/manifest.json \
    --litenn build-release/tools/gguf/litenn_gguf_convert.exe \
    --abs-tol 1e-4 --rel-tol 1e-4
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path


TOKEN_IDS_RE = re.compile(r"^token ids:\s*(?P<ids>[0-9,\s-]+)\s*$", re.MULTILINE)
LOGIT_RE = re.compile(r"^\s*(?P<index>\d+):\s*(?P<value>[-+0-9.eE]+)\s*$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover_litenn(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"LiteNN GGUF tool does not exist: {explicit}")
        return explicit
    for name in ("litenn_gguf_convert.exe", "litenn_gguf_convert"):
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)
    root = repo_root()
    for candidate in (
        root / "build-release" / "tools" / "gguf" / "litenn_gguf_convert.exe",
        root / "build-release" / "tools" / "gguf" / "litenn_gguf_convert",
        root / "build" / "tools" / "gguf" / "litenn_gguf_convert.exe",
        root / "build" / "tools" / "gguf" / "litenn_gguf_convert",
    ):
        if candidate.exists():
            return candidate
    raise SystemExit("litenn_gguf_convert executable was not found; pass --litenn")


def load_manifest(path: Path) -> tuple[dict[str, object], Path]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "litenn.llamacpp_golden_capture.v1":
        raise SystemExit(f"unsupported golden manifest schema: {data.get('schema')!r}")
    return data, path.parent


def unique_artifact(manifest: dict[str, object], base: Path, suffix: str, label: str) -> Path:
    artifacts = manifest.get("debugArtifacts")
    if not isinstance(artifacts, list):
        raise SystemExit("golden manifest is missing debugArtifacts")
    matches = [base / str(path) for path in artifacts if str(path).endswith(suffix)]
    if not matches:
        raise SystemExit(f"golden manifest does not contain a llama-debug {label} artifact")
    if len(matches) > 1:
        names = ", ".join(str(path) for path in matches)
        raise SystemExit(f"golden manifest contains multiple {label} artifacts; pass a trimmed manifest: {names}")
    if not matches[0].exists():
        raise SystemExit(f"{label} artifact does not exist: {matches[0]}")
    return matches[0]


def find_logits_artifact(manifest: dict[str, object], base: Path) -> Path:
    artifacts = manifest.get("debugArtifacts")
    if not isinstance(artifacts, list):
        raise SystemExit("golden manifest is missing debugArtifacts")
    matches = []
    for raw in artifacts:
        text = str(raw)
        if not text.endswith(".txt"):
            continue
        if text.endswith("-prompt.txt") or text.endswith("-embeddings.txt"):
            continue
        matches.append(base / text)
    if not matches:
        raise SystemExit("golden manifest does not contain a llama-debug logits .txt artifact")
    if len(matches) > 1:
        names = ", ".join(str(path) for path in matches)
        raise SystemExit(f"golden manifest contains multiple logits artifacts; pass a trimmed manifest: {names}")
    if not matches[0].exists():
        raise SystemExit(f"logits artifact does not exist: {matches[0]}")
    return matches[0]


def parse_token_ids(prompt_file: Path) -> list[int]:
    text = prompt_file.read_text(encoding="utf-8")
    match = TOKEN_IDS_RE.search(text)
    if match is None:
        raise SystemExit(f"prompt artifact does not contain a 'token ids:' line: {prompt_file}")
    ids = [int(part.strip()) for part in match.group("ids").split(",") if part.strip()]
    if not ids or any(token_id < 0 for token_id in ids):
        raise SystemExit(f"prompt artifact contains invalid token ids: {prompt_file}")
    return ids


def parse_logits(path: Path) -> list[float]:
    values: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LOGIT_RE.match(line)
        if match is None:
            continue
        index = int(match.group("index"))
        value = float(match.group("value"))
        if not math.isfinite(value):
            raise SystemExit(f"logits file contains non-finite value at index {index}: {path}")
        values[index] = value
    if not values:
        raise SystemExit(f"logits file contains no 'index: value' entries: {path}")
    expected = max(values) + 1
    if len(values) != expected:
        raise SystemExit(f"logits file has sparse indices: {path}")
    return [values[index] for index in range(expected)]


def compare_logits(reference: list[float], candidate: list[float]) -> dict[str, object]:
    if len(reference) != len(candidate):
        raise SystemExit(f"logit length mismatch: reference={len(reference)} candidate={len(candidate)}")
    max_abs = 0.0
    max_rel = 0.0
    worst_index = 0
    mismatches: list[dict[str, float | int]] = []
    for index, (ref, got) in enumerate(zip(reference, candidate)):
        abs_error = abs(got - ref)
        denom = max(abs(ref), 1.0)
        rel_error = abs_error / denom
        if abs_error > max_abs or rel_error > max_rel:
            max_abs = max(max_abs, abs_error)
            max_rel = max(max_rel, rel_error)
            worst_index = index
        mismatches.append(
            {
                "index": index,
                "reference": ref,
                "candidate": got,
                "absError": abs_error,
                "relError": rel_error,
            }
        )
    mismatches.sort(key=lambda item: (float(item["absError"]), float(item["relError"])), reverse=True)
    return {
        "count": len(reference),
        "maxAbsError": max_abs,
        "maxRelError": max_rel,
        "worstIndex": worst_index,
        "topMismatches": mismatches[:10],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="llama.cpp capture manifest.json")
    parser.add_argument("--litenn", type=Path, help="Path to litenn_gguf_convert")
    parser.add_argument("--output-dir", type=Path, help="Directory for LiteNN dump and comparison report")
    parser.add_argument("--abs-tol", type=float, default=1e-4)
    parser.add_argument("--rel-tol", type=float, default=1e-4)
    parser.add_argument("--position-offset", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest, base = load_manifest(args.manifest)
    model = Path(str(manifest.get("model", "")))
    if not model:
        raise SystemExit("golden manifest is missing model")
    prompt_file = unique_artifact(manifest, base, "-prompt.txt", "prompt")
    reference_logits_file = find_logits_artifact(manifest, base)
    token_ids = parse_token_ids(prompt_file)
    out_dir = args.output_dir if args.output_dir is not None else base
    out_dir.mkdir(parents=True, exist_ok=True)
    litenn_logits_file = out_dir / "litenn_last_logits.txt"
    litenn = discover_litenn(args.litenn)
    command = [
        str(litenn),
        "--dump-llama-token-id-logits",
        str(model),
        ",".join(str(token_id) for token_id in token_ids),
        str(litenn_logits_file),
    ]
    if args.position_offset != 0:
        command.append(str(args.position_offset))
    completed = subprocess.run(command, text=True, capture_output=True)
    (out_dir / "litenn_logits.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (out_dir / "litenn_logits.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        return completed.returncode

    reference = parse_logits(reference_logits_file)
    candidate = parse_logits(litenn_logits_file)
    metrics = compare_logits(reference, candidate)
    passed = metrics["maxAbsError"] <= args.abs_tol or metrics["maxRelError"] <= args.rel_tol
    report = {
        "schema": "litenn.llamacpp_logits_compare.v1",
        "sourceManifest": str(args.manifest),
        "promptArtifact": str(prompt_file),
        "referenceLogits": str(reference_logits_file),
        "candidateLogits": str(litenn_logits_file),
        "tokenIds": token_ids,
        "absTol": args.abs_tol,
        "relTol": args.rel_tol,
        "passed": passed,
        "metrics": metrics,
        "command": command,
    }
    report_path = out_dir / "logits_compare.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
