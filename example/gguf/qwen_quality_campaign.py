#!/usr/bin/env python3
"""Run a reproducible multi-prompt LiteNN/llama.cpp natural-generation gate."""

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


PROMPT_SCHEMA = "litenn.qwen_quality_prompts.v1"


def load_prompts(path: Path) -> list[dict[str, str]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != PROMPT_SCHEMA:
        raise SystemExit(f"unsupported prompt schema in {path}")
    raw_prompts = document.get("prompts")
    if not isinstance(raw_prompts, list) or len(raw_prompts) < 2:
        raise SystemExit("quality campaign requires at least two prompts")
    prompts: list[dict[str, str]] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_prompts):
        if not isinstance(raw, dict):
            raise SystemExit(f"prompt {index} must be an object")
        name = raw.get("name")
        prompt = raw.get("prompt")
        if not isinstance(name, str) or not name or name in names or any(ch in name for ch in "\\/:"):
            raise SystemExit(f"prompt {index} has an invalid or duplicate name")
        if not isinstance(prompt, str) or not prompt:
            raise SystemExit(f"prompt {name} is empty")
        names.add(name)
        prompts.append({"name": name, "prompt": prompt})
    return prompts


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--litenn", required=True, type=Path)
    parser.add_argument("--llamacpp-tokenizer-tool", required=True, type=Path)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path(__file__).with_name("qwen_quality_prompts.json"),
    )
    parser.add_argument("--workdir", type=Path, default=Path("build/qwen_quality_campaign"))
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-cache-length", type=int)
    parser.add_argument("--aot-cache-dir", type=Path)
    parser.add_argument("--require-aot-cache-hit", action="store_true")
    parser.add_argument("--llvm-opt-level", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--cpu-aot-threads", type=int)
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument(
        "--reuse-artifacts",
        action="store_true",
        help="Skip runtime execution and rebuild the campaign from existing per-case smoke reports",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--minimum-prefix-agreement", type=float, default=0.95)
    parser.add_argument("--minimum-same-context-top-k-overlap", type=float, default=0.90)
    return parser


def run_case(args: argparse.Namespace, name: str, prompt: str, case_dir: Path) -> None:
    command = [
        sys.executable,
        str(Path(__file__).with_name("qwen_smoke.py")),
        "--model",
        str(args.model),
        "--litenn",
        str(args.litenn),
        "--llamacpp-tokenizer-tool",
        str(args.llamacpp_tokenizer_tool),
        "--prompt",
        prompt,
        "--stateful",
        "--max-tokens",
        str(args.max_tokens),
        "--workdir",
        str(case_dir),
        "--llvm-opt-level",
        str(args.llvm_opt_level),
        "--memory-sample-interval-ms",
        "0",
        "--capture-natural-generation",
        "--capture-natural-generation-reference",
    ]
    if args.raw_prompt:
        command.append("--raw-prompt")
    if args.max_cache_length is not None:
        command.extend(["--max-cache-length", str(args.max_cache_length)])
    if args.aot_cache_dir is not None:
        command.extend(["--aot-cache-dir", str(args.aot_cache_dir)])
    if args.require_aot_cache_hit:
        command.append("--require-aot-cache-hit")
    if args.cpu_aot_threads is not None:
        command.extend(["--cpu-aot-threads", str(args.cpu_aot_threads)])

    print(f"[quality campaign] running {name}", flush=True)
    completed = subprocess.run(command, text=True, capture_output=True)
    (case_dir / "campaign_driver.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (case_dir / "campaign_driver.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise SystemExit(
            f"quality case {name} failed with return code {completed.returncode}; "
            f"see {case_dir / 'campaign_driver.stderr.txt'}"
        )


def main() -> int:
    args = build_parser().parse_args()
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")
    if args.max_cache_length is not None and args.max_cache_length <= 0:
        raise SystemExit("--max-cache-length must be positive")
    if args.require_aot_cache_hit and args.aot_cache_dir is None:
        raise SystemExit("--require-aot-cache-hit requires --aot-cache-dir")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if not 0.0 <= args.minimum_prefix_agreement <= 1.0:
        raise SystemExit("--minimum-prefix-agreement must be in [0, 1]")
    if not 0.0 <= args.minimum_same_context_top_k_overlap <= 1.0:
        raise SystemExit("--minimum-same-context-top-k-overlap must be in [0, 1]")
    prompts = load_prompts(args.prompts)
    workdir: Path = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    cases = []
    for prompt in prompts:
        name = prompt["name"]
        case_dir = workdir / "cases" / name
        case_dir.mkdir(parents=True, exist_ok=True)
        if not args.reuse_artifacts:
            run_case(args, name, prompt["prompt"], case_dir)
        smoke_report_path = case_dir / "qwen_smoke_report.json"
        if not smoke_report_path.is_file():
            raise SystemExit(f"quality case {name} is missing {smoke_report_path}")
        smoke_report = json.loads(smoke_report_path.read_text(encoding="utf-8"))
        reference = smoke_report.get("natural_generation_reference_manifest")
        candidate = smoke_report.get("natural_generation_manifest")
        if not isinstance(reference, str) or not isinstance(candidate, str):
            raise SystemExit(f"quality case {name} smoke report is missing natural-generation manifests")
        cases.append(
            {
                "name": name,
                "referenceManifest": relative_or_absolute(Path(reference), workdir),
                "candidateManifest": relative_or_absolute(Path(candidate), workdir),
            }
        )

    campaign = {
        "schema": quality_gate.CAMPAIGN_SCHEMA,
        "thresholds": {
            "topK": args.top_k,
            "minimumCaseCount": len(prompts),
            "minimumTotalReferenceTokens": args.max_tokens * len(prompts),
            "minimumPrefixAgreement": args.minimum_prefix_agreement,
            "minimumSameContextTopKOverlap": args.minimum_same_context_top_k_overlap,
        },
        "cases": cases,
    }
    campaign_path = workdir / "campaign.json"
    campaign_path.write_text(json.dumps(campaign, indent=2) + "\n", encoding="utf-8")
    try:
        report = quality_gate.evaluate_campaign(campaign_path)
    except quality_gate.QualityError as error:
        raise SystemExit(str(error)) from error
    report_path = workdir / "generation_quality_report.json"
    markdown_path = workdir / "generation_quality_report.md"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(quality_gate.markdown_report(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"quality_gate={'PASS' if report['passed'] else 'FAIL'} report={report_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
