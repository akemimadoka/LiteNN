#!/usr/bin/env python3
"""Compare per-step LiteNN decode logits with exact-token llama.cpp references.

The reference manifest must use schema ``litenn.llamacpp_decode_logits.v1`` and
list full-vocabulary ``index: value`` files in ``logitsArtifacts``. Each entry
contains ``decodeStep`` (1 is the logits after the first generated token) and
``path``. This explicit token-level contract avoids tokenizer round-trip drift.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


LOGIT_RE = re.compile(r"^\s*(?P<index>\d+):\s*(?P<value>[-+0-9.eE]+)\s*$")


def load_json(path: Path, schema: str) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != schema:
        raise SystemExit(f"unsupported manifest schema in {path}: {data.get('schema')!r}")
    return data


def resolve_artifacts(manifest: dict[str, object], base: Path) -> dict[int, Path]:
    raw_artifacts = manifest.get("logitsArtifacts")
    if not isinstance(raw_artifacts, list):
        raise SystemExit("manifest is missing logitsArtifacts")
    artifacts: dict[int, Path] = {}
    for raw in raw_artifacts:
        if not isinstance(raw, dict) or raw.get("decodeStep") is None or raw.get("path") is None:
            continue
        step = int(raw["decodeStep"])
        path = Path(str(raw["path"]))
        if not path.is_absolute():
            path = base / path
        if step <= 0 or step in artifacts:
            raise SystemExit(f"invalid or duplicate decodeStep {step} in manifest")
        if not path.exists():
            raise SystemExit(f"decode logits artifact does not exist: {path}")
        artifacts[step] = path
    if not artifacts:
        raise SystemExit("manifest contains no decode logits artifacts")
    return artifacts


def parse_logits(path: Path) -> list[float]:
    values: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LOGIT_RE.match(line)
        if match is None:
            continue
        index = int(match.group("index"))
        value = float(match.group("value"))
        if not math.isfinite(value):
            raise SystemExit(f"non-finite logit at index {index}: {path}")
        values[index] = value
    if not values or len(values) != max(values) + 1:
        raise SystemExit(f"logits file is empty or sparse: {path}")
    return [values[index] for index in range(len(values))]


def compare(reference: list[float], candidate: list[float]) -> dict[str, object]:
    if len(reference) != len(candidate):
        raise SystemExit(f"logit length mismatch: reference={len(reference)} candidate={len(candidate)}")
    rows = []
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        abs_error = abs(actual - expected)
        rel_error = abs_error / max(abs(expected), 1.0)
        rows.append(
            {
                "index": index,
                "reference": expected,
                "candidate": actual,
                "absError": abs_error,
                "relError": rel_error,
            }
        )
    rows.sort(key=lambda row: (float(row["absError"]), float(row["relError"])), reverse=True)
    return {
        "count": len(reference),
        "maxAbsError": max(float(row["absError"]) for row in rows),
        "maxRelError": max(float(row["relError"]) for row in rows),
        "topMismatches": rows[:10],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", required=True, type=Path)
    parser.add_argument("--replay-manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Defaults next to the replay manifest")
    parser.add_argument("--abs-tol", type=float, default=1e-4)
    parser.add_argument("--rel-tol", type=float, default=1e-4)
    parser.add_argument("--max-decode-steps", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_decode_steps is not None and args.max_decode_steps <= 0:
        raise SystemExit("--max-decode-steps must be positive")
    reference_manifest = load_json(args.reference_manifest, "litenn.llamacpp_decode_logits.v1")
    replay_manifest = load_json(args.replay_manifest, "litenn.golden_replay.v2")
    if reference_manifest.get("promptTokenIds") != replay_manifest.get("tokenIds"):
        raise SystemExit("reference and replay prompt token ids differ")
    reference_generated_ids = reference_manifest.get("generatedTokenIds")
    replay_generated_ids = replay_manifest.get("generatedTokenIds")
    if not isinstance(reference_generated_ids, list) or not isinstance(replay_generated_ids, list):
        raise SystemExit("reference or replay manifest is missing generatedTokenIds")
    reference = resolve_artifacts(reference_manifest, args.reference_manifest.parent)
    candidate = resolve_artifacts(replay_manifest, args.replay_manifest.parent)
    common_steps = sorted(reference.keys() & candidate.keys())
    if args.max_decode_steps is not None:
        common_steps = common_steps[: args.max_decode_steps]
    if not common_steps:
        raise SystemExit("reference and replay manifests have no common decode steps")
    compared_prefix = max(common_steps)
    if reference_generated_ids[:compared_prefix] != replay_generated_ids[:compared_prefix]:
        raise SystemExit("reference and replay generated token ids diverge before the compared decode steps")

    step_reports = []
    passed = True
    for step in common_steps:
        metrics = compare(parse_logits(reference[step]), parse_logits(candidate[step]))
        step_passed = metrics["maxAbsError"] <= args.abs_tol or metrics["maxRelError"] <= args.rel_tol
        passed = passed and step_passed
        step_reports.append(
            {
                "decodeStep": step,
                "referenceLogits": str(reference[step]),
                "candidateLogits": str(candidate[step]),
                "passed": step_passed,
                "metrics": metrics,
            }
        )

    report = {
        "schema": "litenn.llamacpp_decode_logits_compare.v1",
        "referenceManifest": str(args.reference_manifest),
        "replayManifest": str(args.replay_manifest),
        "absTol": args.abs_tol,
        "relTol": args.rel_tol,
        "comparedDecodeSteps": common_steps,
        "passed": passed,
        "steps": step_reports,
    }
    output = args.output if args.output is not None else args.replay_manifest.parent / "decode_logits_compare.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
