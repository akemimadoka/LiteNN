#!/usr/bin/env python3
"""Run a fixed public-text teacher-forced corpus loss campaign for Qwen GGUF models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gguf_corpus_loss_gate as loss_gate


CORPUS_SCHEMA = "litenn.corpus_text_slice.v1"
TOKEN_SCHEMA = "litenn.llamacpp_tokens.v1"
GENERATION_SCHEMA = "litenn.natural_generation.v1"


def load_json(path: Path, schema: str) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"failed to read JSON {path}: {error}") from error
    if not isinstance(document, dict) or document.get("schema") != schema:
        raise SystemExit(f"unsupported schema in {path}")
    return document


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def load_corpus_slice(path: Path) -> tuple[dict[str, object], list[dict[str, str]], int]:
    document = load_json(path, CORPUS_SCHEMA)
    for key in ("name", "license", "provenance", "normalization"):
        if not isinstance(document.get(key), str) or not document[key]:
            raise SystemExit(f"corpus slice {path} has invalid {key}")
    default_target_count = document.get("defaultTargetTokensPerSample")
    if not isinstance(default_target_count, int) or isinstance(default_target_count, bool) or default_target_count <= 0:
        raise SystemExit(f"corpus slice {path} has invalid defaultTargetTokensPerSample")
    raw_samples = document.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise SystemExit(f"corpus slice {path} has no samples")
    samples: list[dict[str, str]] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, dict):
            raise SystemExit(f"corpus sample {index} must be an object")
        sample: dict[str, str] = {}
        for key in ("name", "category", "sha256", "text"):
            value = raw.get(key)
            if not isinstance(value, str) or not value:
                raise SystemExit(f"corpus sample {index} has invalid {key}")
            sample[key] = value
        if sample["name"] in names:
            raise SystemExit(f"duplicate corpus sample name {sample['name']}")
        names.add(sample["name"])
        actual_digest = hashlib.sha256(sample["text"].encode("utf-8")).hexdigest()
        if actual_digest != sample["sha256"]:
            raise SystemExit(
                f"corpus sample {sample['name']} SHA-256 mismatch: expected={sample['sha256']} actual={actual_digest}"
            )
        samples.append(sample)
    return document, samples, default_target_count


def run_step(name: str, command: list[str], workdir: Path, environment: dict[str, str] | None = None) -> dict[str, object]:
    print(f"[corpus loss] {name}...", flush=True)
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=environment)
    elapsed = time.perf_counter() - start
    stdout_path = workdir / f"{name}.stdout.txt"
    stderr_path = workdir / f"{name}.stderr.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    print(f"[corpus loss] {name}: {elapsed:.3f}s returncode={completed.returncode}", flush=True)
    if completed.returncode != 0:
        raise SystemExit(f"{name} failed with return code {completed.returncode}; see {stderr_path}")
    return {
        "name": name,
        "command": command,
        "elapsedSeconds": elapsed,
        "returncode": completed.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def tokenize_sample(
    sample: dict[str, str], model: Path, adapter: Path, case_dir: Path, environment: dict[str, str]
) -> tuple[list[int], dict[str, object]]:
    input_path = case_dir / "source.txt"
    output_path = case_dir / "tokens.json"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(sample["text"].encode("utf-8"))
    step = run_step(
        f"tokenize_{sample['name']}",
        [str(adapter), "tokenize-file", str(model), str(input_path), str(output_path)],
        case_dir,
        environment,
    )
    document = load_json(output_path, TOKEN_SCHEMA)
    raw_tokens = document.get("tokenIds")
    if not isinstance(raw_tokens, list) or any(not isinstance(token, int) or token < 0 for token in raw_tokens):
        raise SystemExit(f"tokenizer returned invalid token ids for {sample['name']}")
    if len(raw_tokens) < 2:
        raise SystemExit(f"corpus sample {sample['name']} tokenized to fewer than two tokens")
    return raw_tokens, step


def normalize_candidate_manifest(source_path: Path, output_path: Path, targets: list[int]) -> Path:
    source = load_json(source_path, GENERATION_SCHEMA)
    prompt = source.get("promptTokenIds")
    generated = source.get("generatedTokenIds")
    if not isinstance(prompt, list) or not prompt or any(not isinstance(token, int) or token < 0 for token in prompt):
        raise SystemExit(f"candidate manifest {source_path} has invalid promptTokenIds")
    if generated != targets:
        raise SystemExit(f"candidate manifest {source_path} did not preserve the teacher-forced targets")
    if source.get("sampling") != "forced-reference-trajectory":
        raise SystemExit(f"candidate manifest {source_path} is not teacher-forced")
    if source.get("fallbackUsed") is not False:
        raise SystemExit(f"candidate manifest {source_path} used fallback")
    raw_artifacts = source.get("logitsArtifacts")
    if not isinstance(raw_artifacts, list):
        raise SystemExit(f"candidate manifest {source_path} is missing logitsArtifacts")
    artifacts: list[dict[str, object]] = []
    for expected_step, raw in enumerate(raw_artifacts):
        if not isinstance(raw, dict) or raw.get("decisionStep") != expected_step:
            raise SystemExit(f"candidate manifest {source_path} has shifted logits steps")
        if expected_step >= len(targets):
            raise SystemExit(f"candidate manifest {source_path} contains extra logits steps")
        if raw.get("position") != len(prompt) + expected_step:
            raise SystemExit(f"candidate manifest {source_path} has shifted logits positions")
        if raw.get("selectedTokenId") != targets[expected_step]:
            raise SystemExit(f"candidate manifest {source_path} has shifted selected tokens")
        raw_path = Path(str(raw.get("path")))
        artifact_path = raw_path if raw_path.is_absolute() else source_path.parent / raw_path
        if not artifact_path.is_file():
            raise SystemExit(f"candidate logits artifact does not exist: {artifact_path}")
        artifacts.append(
            {
                "decisionStep": expected_step,
                "position": len(prompt) + expected_step,
                "targetTokenId": targets[expected_step],
                "path": relative_or_absolute(artifact_path, output_path.parent),
            }
        )
    if len(artifacts) != len(targets):
        raise SystemExit(
            f"candidate manifest {source_path} logits coverage mismatch: expected={len(targets)} actual={len(artifacts)}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "schema": loss_gate.MANIFEST_SCHEMA,
        "producer": "LiteNN",
        "runtime": "cpu_aot",
        "captureBoundary": "pre-target",
        "promptTokenIds": prompt,
        "targetTokenIds": targets,
        "fallbackUsed": False,
        "logitsArtifacts": artifacts,
    }
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--litenn", required=True, type=Path)
    parser.add_argument("--llamacpp-adapter", required=True, type=Path)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).with_name("qwen_corpus_loss_slice.json"),
    )
    parser.add_argument("--workdir", type=Path, default=Path("build/qwen_corpus_loss"))
    parser.add_argument("--target-tokens-per-sample", type=int)
    parser.add_argument("--max-cache-length", type=int)
    parser.add_argument("--aot-cache-dir", type=Path)
    parser.add_argument("--require-aot-cache-hit", action="store_true")
    parser.add_argument("--no-aot-cache-write", action="store_true")
    parser.add_argument("--llvm-opt-level", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--cpu-aot-threads", type=int)
    parser.add_argument("--reuse-artifacts", action="store_true")
    parser.add_argument("--maximum-cross-entropy-regression-nats", type=float, default=0.02)
    parser.add_argument("--maximum-relative-perplexity-regression", type=float, default=0.02)
    parser.add_argument("--maximum-worst-sample-regression-nats", type=float, default=0.10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for label, path in (
        ("model", args.model),
        ("LiteNN tool", args.litenn),
        ("llama.cpp adapter", args.llamacpp_adapter),
        ("corpus", args.corpus),
    ):
        if not path.is_file():
            raise SystemExit(f"{label} does not exist: {path}")
    if args.require_aot_cache_hit and args.aot_cache_dir is None:
        raise SystemExit("--require-aot-cache-hit requires --aot-cache-dir")
    if args.no_aot_cache_write and args.aot_cache_dir is None:
        raise SystemExit("--no-aot-cache-write requires --aot-cache-dir")
    for label, value in (
        ("--maximum-cross-entropy-regression-nats", args.maximum_cross_entropy_regression_nats),
        ("--maximum-relative-perplexity-regression", args.maximum_relative_perplexity_regression),
        ("--maximum-worst-sample-regression-nats", args.maximum_worst_sample_regression_nats),
    ):
        if value < 0.0:
            raise SystemExit(f"{label} must be non-negative")

    corpus_path = args.corpus.resolve()
    corpus, samples, default_target_count = load_corpus_slice(corpus_path)
    target_limit = args.target_tokens_per_sample or default_target_count
    if target_limit <= 0:
        raise SystemExit("--target-tokens-per-sample must be positive")
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    runtime_paths = [args.llamacpp_adapter.resolve().parent, args.llamacpp_adapter.resolve().parent / "bin"]
    environment["PATH"] = os.pathsep.join(str(path) for path in runtime_paths) + os.pathsep + environment.get("PATH", "")
    steps: list[dict[str, object]] = []
    tokenized: list[tuple[dict[str, str], list[int], Path]] = []
    for sample in samples:
        case_dir = workdir / "cases" / sample["name"]
        if args.reuse_artifacts:
            token_document = load_json(case_dir / "tokens.json", TOKEN_SCHEMA)
            raw_tokens = token_document.get("tokenIds")
            if not isinstance(raw_tokens, list) or any(not isinstance(token, int) or token < 0 for token in raw_tokens):
                raise SystemExit(f"cached tokenizer output is invalid for {sample['name']}")
            tokens = raw_tokens
        else:
            tokens, token_step = tokenize_sample(
                sample, args.model.resolve(), args.llamacpp_adapter.resolve(), case_dir, environment
            )
            steps.append(token_step)
        if len(tokens) < 2:
            raise SystemExit(f"corpus sample {sample['name']} has fewer than two tokens")
        targets = tokens[1 : 1 + target_limit]
        if len(targets) != target_limit:
            raise SystemExit(
                f"corpus sample {sample['name']} provides only {len(targets)} targets; {target_limit} required"
            )
        tokenized.append((sample, [tokens[0], *targets], case_dir))

    maximum_required_cache = max(len(tokens) for _, tokens, _ in tokenized)
    max_cache_length = args.max_cache_length or maximum_required_cache
    if max_cache_length < maximum_required_cache:
        raise SystemExit(
            f"--max-cache-length {max_cache_length} is smaller than required sequence length {maximum_required_cache}"
        )
    cache_dir = args.aot_cache_dir.resolve() if args.aot_cache_dir is not None else workdir / "aot_cache"
    cases: list[dict[str, object]] = []
    for sample, tokens, case_dir in tokenized:
        prompt = tokens[:1]
        targets = tokens[1:]
        prompt_text = ",".join(str(token) for token in prompt)
        target_text = ",".join(str(token) for token in targets)
        reference_dir = case_dir / "reference"
        candidate_raw_dir = case_dir / "candidate_raw"
        candidate_smoke_dir = case_dir / "candidate_smoke"
        candidate_manifest = case_dir / "candidate" / "manifest.json"
        if not args.reuse_artifacts:
            steps.append(
                run_step(
                    f"reference_{sample['name']}",
                    [
                        str(args.llamacpp_adapter.resolve()),
                        "teacher-forced-logits",
                        str(args.model.resolve()),
                        prompt_text,
                        target_text,
                        str(reference_dir),
                    ],
                    case_dir,
                    environment,
                )
            )
            candidate_command = [
                sys.executable,
                str(Path(__file__).with_name("qwen_smoke.py")),
                "--model",
                str(args.model.resolve()),
                "--litenn",
                str(args.litenn.resolve()),
                "--token-ids",
                prompt_text,
                "--stateful",
                "--max-tokens",
                str(len(targets)),
                "--max-cache-length",
                str(max_cache_length),
                "--ignore-eos",
                "--forced-generated-token-ids",
                target_text,
                "--capture-natural-generation",
                "--natural-generation-dir",
                str(candidate_raw_dir),
                "--workdir",
                str(candidate_smoke_dir),
                "--aot-cache-dir",
                str(cache_dir),
                "--llvm-opt-level",
                str(args.llvm_opt_level),
                "--memory-sample-interval-ms",
                "0",
            ]
            if args.require_aot_cache_hit:
                candidate_command.append("--require-aot-cache-hit")
            if args.no_aot_cache_write:
                candidate_command.append("--no-aot-cache-write")
            if args.cpu_aot_threads is not None:
                candidate_command.extend(["--cpu-aot-threads", str(args.cpu_aot_threads)])
            steps.append(run_step(f"candidate_{sample['name']}", candidate_command, case_dir))
            normalize_candidate_manifest(candidate_raw_dir / "manifest.json", candidate_manifest, targets)
        reference_manifest = reference_dir / "manifest.json"
        if not reference_manifest.is_file() or not candidate_manifest.is_file():
            raise SystemExit(f"corpus case {sample['name']} is missing reference or candidate artifacts")
        cases.append(
            {
                "name": sample["name"],
                "category": sample["category"],
                "sourceSha256": sample["sha256"],
                "referenceManifest": relative_or_absolute(reference_manifest, workdir),
                "candidateManifest": relative_or_absolute(candidate_manifest, workdir),
            }
        )

    campaign = {
        "schema": loss_gate.CAMPAIGN_SCHEMA,
        "corpus": {
            "name": corpus["name"],
            "version": corpus.get("version"),
            "license": corpus["license"],
            "provenance": corpus["provenance"],
            "normalization": corpus["normalization"],
            "manifest": relative_or_absolute(corpus_path, workdir),
            "targetTokensPerSample": target_limit,
        },
        "thresholds": {
            "minimumCaseCount": len(cases),
            "minimumTokenCount": len(cases) * target_limit,
            "maximumCrossEntropyRegressionNats": args.maximum_cross_entropy_regression_nats,
            "maximumRelativePerplexityRegression": args.maximum_relative_perplexity_regression,
            "maximumWorstSampleCrossEntropyRegressionNats": args.maximum_worst_sample_regression_nats,
        },
        "cases": cases,
    }
    campaign_path = workdir / "campaign.json"
    campaign_path.write_text(json.dumps(campaign, indent=2) + "\n", encoding="utf-8")
    try:
        report = loss_gate.evaluate_campaign(campaign_path)
    except loss_gate.CorpusLossError as error:
        raise SystemExit(str(error)) from error
    report["run"] = {
        "maxCacheLength": max_cache_length,
        "llvmOptLevel": args.llvm_opt_level,
        "requireAOTCacheHit": args.require_aot_cache_hit,
        "aotCacheWrite": not args.no_aot_cache_write,
        "steps": steps,
    }
    report_path = workdir / "corpus_loss_report.json"
    markdown_path = workdir / "corpus_loss_report.md"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(loss_gate.markdown_report(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, allow_nan=False))
    print(f"corpus_loss_gate={'PASS' if report['passed'] else 'FAIL'} report={report_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
