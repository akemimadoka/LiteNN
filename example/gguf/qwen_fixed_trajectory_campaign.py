#!/usr/bin/env python3
"""Replay a reference Qwen trajectory and evaluate every LiteNN distribution on identical contexts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gguf_generation_quality_gate as quality_gate


QUALITY_SCHEMA = "litenn.natural_generation_quality.v1"
GENERATION_SCHEMA = "litenn.natural_generation.v1"


def load_json(path: Path, schema: str) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != schema:
        raise SystemExit(f"unsupported schema in {path}")
    return document


def resolve_existing_path(raw: object, base: Path) -> Path:
    path = Path(str(raw))
    candidates = [path] if path.is_absolute() else [base / path, ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise SystemExit(f"referenced manifest does not exist: {path}")


def token_ids(document: dict[str, object], key: str, source: Path) -> list[int]:
    raw = document.get(key)
    if not isinstance(raw, list) or any(not isinstance(token, int) or token < 0 for token in raw):
        raise SystemExit(f"{source} has invalid {key}")
    return raw


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-report", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--litenn", required=True, type=Path)
    parser.add_argument("--workdir", type=Path, default=Path("build/qwen_fixed_trajectory_campaign"))
    parser.add_argument("--max-cache-length", type=int)
    parser.add_argument("--aot-cache-dir", type=Path)
    parser.add_argument("--require-aot-cache-hit", action="store_true")
    parser.add_argument("--no-aot-cache-write", action="store_true")
    parser.add_argument("--llvm-opt-level", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--cpu-aot-threads", type=int)
    parser.add_argument("--reuse-artifacts", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--minimum-top1-agreement", type=float, default=0.95)
    parser.add_argument("--minimum-top-k-overlap", type=float, default=0.95)
    parser.add_argument("--minimum-centered-cosine", type=float, default=0.999)
    parser.add_argument("--maximum-jensen-shannon", type=float, default=0.001)
    return parser


def run_candidate(
    args: argparse.Namespace,
    name: str,
    prompt: list[int],
    trajectory: list[int],
    case_dir: Path,
) -> Path:
    candidate_dir = case_dir / "candidate"
    smoke_dir = case_dir / "smoke"
    command = [
        sys.executable,
        str(Path(__file__).with_name("qwen_smoke.py")),
        "--model",
        str(args.model.resolve()),
        "--litenn",
        str(args.litenn.resolve()),
        "--token-ids",
        ",".join(str(token) for token in prompt),
        "--stateful",
        "--max-tokens",
        str(len(trajectory)),
        "--ignore-eos",
        "--forced-generated-token-ids",
        ",".join(str(token) for token in trajectory),
        "--capture-natural-generation",
        "--natural-generation-dir",
        str(candidate_dir),
        "--workdir",
        str(smoke_dir),
        "--llvm-opt-level",
        str(args.llvm_opt_level),
        "--memory-sample-interval-ms",
        "0",
    ]
    if args.max_cache_length is not None:
        command.extend(["--max-cache-length", str(args.max_cache_length)])
    if args.aot_cache_dir is not None:
        command.extend(["--aot-cache-dir", str(args.aot_cache_dir.resolve())])
    if args.require_aot_cache_hit:
        command.append("--require-aot-cache-hit")
    if args.no_aot_cache_write:
        command.append("--no-aot-cache-write")
    if args.cpu_aot_threads is not None:
        command.extend(["--cpu-aot-threads", str(args.cpu_aot_threads)])

    case_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fixed trajectory] running {name}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    (case_dir / "driver.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (case_dir / "driver.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise SystemExit(
            f"fixed-trajectory case {name} failed with return code {completed.returncode}; "
            f"see {case_dir / 'driver.stderr.txt'}"
        )
    return candidate_dir / "manifest.json"


def main() -> int:
    args = build_parser().parse_args()
    if args.max_cache_length is not None and args.max_cache_length <= 0:
        raise SystemExit("--max-cache-length must be positive")
    if args.require_aot_cache_hit and args.aot_cache_dir is None:
        raise SystemExit("--require-aot-cache-hit requires --aot-cache-dir")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    for name, value in (
        ("--minimum-top1-agreement", args.minimum_top1_agreement),
        ("--minimum-top-k-overlap", args.minimum_top_k_overlap),
        ("--minimum-centered-cosine", args.minimum_centered_cosine),
    ):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} must be in [0, 1]")
    if args.maximum_jensen_shannon < 0.0:
        raise SystemExit("--maximum-jensen-shannon must be non-negative")

    quality_path = args.quality_report.resolve()
    quality = load_json(quality_path, QUALITY_SCHEMA)
    raw_cases = quality.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SystemExit("quality report contains no cases")
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    total_tokens = 0
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict) or raw_case.get("passedIntegrity") is not True:
            raise SystemExit("quality report contains an invalid case")
        name = raw_case.get("name")
        if not isinstance(name, str) or not name:
            raise SystemExit("quality report contains an unnamed case")
        reference_path = resolve_existing_path(raw_case.get("referenceManifest"), quality_path.parent)
        reference = load_json(reference_path, GENERATION_SCHEMA)
        prompt = token_ids(reference, "promptTokenIds", reference_path)
        trajectory = token_ids(reference, "generatedTokenIds", reference_path)
        if not prompt or not trajectory:
            raise SystemExit(f"quality case {name} has an empty prompt or trajectory")
        case_dir = workdir / "cases" / name
        candidate_path = case_dir / "candidate" / "manifest.json"
        if not args.reuse_artifacts:
            candidate_path = run_candidate(args, name, prompt, trajectory, case_dir)
        if not candidate_path.is_file():
            raise SystemExit(f"fixed-trajectory case {name} is missing {candidate_path}")
        cases.append(
            {
                "name": name,
                "comparisonMode": quality_gate.FIXED_REFERENCE_COMPARISON,
                "referenceManifest": relative_or_absolute(reference_path, workdir),
                "candidateManifest": relative_or_absolute(candidate_path, workdir),
            }
        )
        total_tokens += len(trajectory)

    campaign = {
        "schema": quality_gate.CAMPAIGN_SCHEMA,
        "thresholds": {
            "topK": args.top_k,
            "minimumCaseCount": len(cases),
            "minimumTotalReferenceTokens": total_tokens,
            "minimumFixedTrajectoryTop1Agreement": args.minimum_top1_agreement,
            "minimumFixedTrajectoryTopKOverlap": args.minimum_top_k_overlap,
            "minimumFixedTrajectoryCenteredCosine": args.minimum_centered_cosine,
            "maximumFixedTrajectoryJensenShannon": args.maximum_jensen_shannon,
        },
        "cases": cases,
    }
    campaign_path = workdir / "campaign.json"
    campaign_path.write_text(json.dumps(campaign, indent=2) + "\n", encoding="utf-8")
    try:
        report = quality_gate.evaluate_campaign(campaign_path)
    except quality_gate.QualityError as error:
        raise SystemExit(str(error)) from error
    report_path = workdir / "fixed_trajectory_quality_report.json"
    markdown_path = workdir / "fixed_trajectory_quality_report.md"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(quality_gate.markdown_report(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"fixed_trajectory_gate={'PASS' if report['passed'] else 'FAIL'} report={report_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
