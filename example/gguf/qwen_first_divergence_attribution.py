#!/usr/bin/env python3
"""Capture and compare same-context hidden states at natural-generation divergences."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import compare_layer_checkpoints as layer_compare


QUALITY_SCHEMA = "litenn.natural_generation_quality.v1"
GENERATION_SCHEMA = "litenn.natural_generation.v1"
REPORT_SCHEMA = "litenn.qwen_first_divergence_attribution.v1"
NRMSE_THRESHOLDS = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)
SUB_LAYER_BOUNDARIES = (
    "attention_norm",
    "query_rotated",
    "key_rotated",
    "value",
    "attention_context",
    "attention_output",
    "attention_residual",
    "ffn_norm",
    "ffn_gate",
    "ffn_up",
    "ffn_swiglu",
    "ffn_down",
    "post_ffn",
)


def load_json(path: Path, schema: str) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"failed to read JSON {path}: {error}") from error
    if not isinstance(document, dict) or document.get("schema") != schema:
        actual = document.get("schema") if isinstance(document, dict) else type(document).__name__
        raise RuntimeError(f"unsupported schema in {path}: {actual!r}")
    return document


def resolve_existing_path(raw: object, anchor: Path) -> Path:
    path = Path(str(raw))
    candidates = [path] if path.is_absolute() else [path, anchor / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(f"recorded artifact does not exist: {path}")


def token_ids(document: dict[str, object], key: str, source: Path) -> list[int]:
    raw = document.get(key)
    if not isinstance(raw, list) or any(not isinstance(token, int) or token < 0 for token in raw):
        raise RuntimeError(f"{source} has invalid {key}")
    return raw


def case_context(case: dict[str, object], quality_report: Path) -> dict[str, object]:
    name = case.get("name")
    divergence = case.get("firstDivergenceDecisionStep")
    if not isinstance(name, str) or not name or case.get("passedIntegrity") is not True:
        raise RuntimeError("quality report contains an invalid case")
    if not isinstance(divergence, int) or divergence < 0:
        raise RuntimeError(f"quality case {name} has no finite first divergence")
    reference_path = resolve_existing_path(case.get("referenceManifest"), quality_report.parent)
    reference = load_json(reference_path, GENERATION_SCHEMA)
    prompt = token_ids(reference, "promptTokenIds", reference_path)
    generated = token_ids(reference, "generatedTokenIds", reference_path)
    if not prompt or divergence >= len(generated):
        raise RuntimeError(f"quality case {name} cannot provide the first-divergence context")
    return {
        "name": name,
        "divergence": divergence,
        "prompt": prompt,
        "generatedPrefix": generated[:divergence],
        "forcedTrajectory": generated[: divergence + 1],
    }


def run_logged(label: str, command: list[str], log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[first-divergence] {label}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    (log_dir / f"{label}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (log_dir / f"{label}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed with return code {completed.returncode}; see {log_dir / f'{label}.stderr.txt'}"
        )


def compact_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "layer": row["layer"],
        "normalizedRmsError": row["normalized_rms_error"],
        "cosineDistance": 1.0 - float(row["cosine_similarity"]),
        "maxAbsoluteError": row["max_absolute_error"],
        "meanAbsoluteError": row["mean_absolute_error"],
    }


def summarize_comparison(comparison: dict[str, object], generated_index: int) -> dict[str, object]:
    rows = [
        row
        for row in comparison.get("rows", [])
        if isinstance(row, dict) and row.get("generated_index") == generated_index
    ]
    rows.sort(key=lambda row: int(row["layer"]))
    if not rows:
        raise RuntimeError(f"comparison has no rows for generated index {generated_index}")
    numeric_rows = [row for row in rows if row.get("normalized_rms_error") is not None]
    if not numeric_rows:
        raise RuntimeError("comparison has no finite NRMSE rows")

    representative_indices = sorted({0, len(rows) // 2, len(rows) - 1})
    first_by_threshold: dict[str, int | None] = {}
    for threshold in NRMSE_THRESHOLDS:
        first = next(
            (
                int(row["layer"])
                for row in numeric_rows
                if float(row["normalized_rms_error"]) >= threshold
            ),
            None,
        )
        first_by_threshold[f"{threshold:.0e}"] = first

    peak = max(numeric_rows, key=lambda row: float(row["normalized_rms_error"]))
    previous_nrmse = 0.0
    increases: list[tuple[float, dict[str, object]]] = []
    for row in numeric_rows:
        nrmse = float(row["normalized_rms_error"])
        increases.append((nrmse - previous_nrmse, row))
        previous_nrmse = nrmse
    largest_increase, largest_increase_row = max(increases, key=lambda item: item[0])
    first_failing = comparison.get("first_failing_layer_by_generated_index", {}).get(str(generated_index))
    return {
        "layerCount": len(rows),
        "firstExactFailingLayer": first_failing,
        "firstLayerByNrmseThreshold": first_by_threshold,
        "peakNrmse": compact_row(peak),
        "largestPositiveNrmseIncrease": {
            **compact_row(largest_increase_row),
            "increase": largest_increase,
        },
        "representativeLayers": [compact_row(rows[index]) for index in representative_indices],
    }


def summarize_sub_layers(
    reference_root: Path,
    candidate_root: Path,
    generated_index: int,
    blocks: list[int],
    absolute_tolerance: float,
    relative_tolerance: float,
    output_dir: Path,
) -> dict[str, object]:
    boundary_reports: dict[str, dict[str, object]] = {}
    coordinates: list[dict[str, object]] = []
    for boundary in SUB_LAYER_BOUNDARIES:
        reference_manifest = reference_root / boundary / "manifest.tsv"
        candidate_manifest = candidate_root / boundary / "manifest.tsv"
        if not reference_manifest.is_file() or not candidate_manifest.is_file():
            raise RuntimeError(f"sub-layer boundary {boundary} is missing a manifest")
        comparison = layer_compare.compare_manifests(
            reference_manifest,
            candidate_manifest,
            absolute_tolerance,
            relative_tolerance,
            {generated_index},
        )
        boundary_reports[boundary] = comparison
        for row in comparison["rows"]:
            coordinate = compact_row(row)
            coordinate["boundary"] = boundary
            coordinates.append(coordinate)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "boundary_comparisons.json").write_text(
        json.dumps(boundary_reports, indent=2) + "\n", encoding="utf-8"
    )
    per_block: list[dict[str, object]] = []
    for block in blocks:
        rows = [row for row in coordinates if row["layer"] == block]
        rows.sort(key=lambda row: SUB_LAYER_BOUNDARIES.index(str(row["boundary"])))
        if len(rows) != len(SUB_LAYER_BOUNDARIES):
            raise RuntimeError(f"sub-layer block {block} does not cover every boundary")
        first_by_threshold: dict[str, str | None] = {}
        for threshold in NRMSE_THRESHOLDS:
            first = next(
                (
                    str(row["boundary"])
                    for row in rows
                    if row["normalizedRmsError"] is not None
                    and float(row["normalizedRmsError"]) >= threshold
                ),
                None,
            )
            first_by_threshold[f"{threshold:.0e}"] = first
        peak = max(rows, key=lambda row: float(row["normalizedRmsError"] or 0.0))
        previous = 0.0
        increases: list[tuple[float, dict[str, object]]] = []
        for row in rows:
            nrmse = float(row["normalizedRmsError"] or 0.0)
            increases.append((nrmse - previous, row))
            previous = nrmse
        increase, increase_row = max(increases, key=lambda item: item[0])
        per_block.append(
            {
                "block": block,
                "firstBoundaryByNrmseThreshold": first_by_threshold,
                "peakNrmse": peak,
                "largestPositiveNrmseIncrease": {**increase_row, "increase": increase},
                "boundaries": rows,
            }
        )
    return {"boundaryCount": len(SUB_LAYER_BOUNDARIES), "blocks": per_block}


def markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Qwen First-Divergence Layer Attribution",
        "",
        "Each row compares LiteNN and llama.cpp after replaying the exact reference token history.",
        "",
        "| Case | Decision | Layers | First NRMSE >=1e-4 | Peak layer | Peak NRMSE | First sub-layer >=1e-4 | Peak sub-layer |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for case in report["cases"]:
        summary = case["summary"]
        first_material = summary["firstLayerByNrmseThreshold"]["1e-04"]
        sub_layer = case.get("subLayerSummary")
        first_sub_layer = "n/a"
        peak_sub_layer = "n/a"
        if isinstance(sub_layer, dict) and sub_layer.get("blocks"):
            first_block = sub_layer["blocks"][0]
            first_sub_layer = first_block["firstBoundaryByNrmseThreshold"]["1e-04"]
            peak_sub_layer = first_block["peakNrmse"]["boundary"]
        lines.append(
            f"| {case['name']} | {case['firstDivergenceDecisionStep']} | {summary['layerCount']} | "
            f"{first_material} | {summary['peakNrmse']['layer']} | "
            f"{summary['peakNrmse']['normalizedRmsError']:.6g} | {first_sub_layer} | {peak_sub_layer} |"
        )
    lines.extend(
        [
            "",
            "The threshold ladder and complete per-layer comparison artifacts are retained in the JSON report and each case directory.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-report", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--litenn", required=True, type=Path)
    parser.add_argument("--llamacpp-tokenizer-tool", required=True, type=Path)
    parser.add_argument("--workdir", type=Path, default=Path("build/qwen_first_divergence_attribution"))
    parser.add_argument("--max-cache-length", type=int, default=256)
    parser.add_argument("--aot-cache-dir", type=Path)
    parser.add_argument("--require-aot-cache-hit", action="store_true")
    parser.add_argument("--no-aot-cache-write", action="store_true")
    parser.add_argument("--llvm-opt-level", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--cpu-aot-threads", type=int)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-5)
    parser.add_argument(
        "--sub-layer-blocks",
        default="0",
        help="Comma-separated blocks for internal attribution; empty disables sub-layer capture",
    )
    parser.add_argument("--reuse-artifacts", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_cache_length <= 0:
        raise SystemExit("--max-cache-length must be positive")
    if args.require_aot_cache_hit and args.aot_cache_dir is None:
        raise SystemExit("--require-aot-cache-hit requires --aot-cache-dir")
    if args.absolute_tolerance < 0 or args.relative_tolerance < 0:
        raise SystemExit("checkpoint tolerances must be non-negative")
    try:
        sub_layer_blocks = sorted({int(value) for value in args.sub_layer_blocks.split(",") if value})
    except ValueError as error:
        raise SystemExit("--sub-layer-blocks must contain comma-separated non-negative integers") from error
    if any(block < 0 for block in sub_layer_blocks):
        raise SystemExit("--sub-layer-blocks must contain comma-separated non-negative integers")

    quality_report = args.quality_report.resolve()
    quality = load_json(quality_report, QUALITY_SCHEMA)
    raw_cases = quality.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SystemExit("quality report contains no cases")
    contexts = [case_context(case, quality_report) for case in raw_cases if isinstance(case, dict)]
    if len(contexts) != len(raw_cases):
        raise SystemExit("quality report contains a non-object case")

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    case_reports: list[dict[str, object]] = []
    for context in contexts:
        name = str(context["name"])
        divergence = int(context["divergence"])
        prompt = [int(value) for value in context["prompt"]]
        generated_prefix = [int(value) for value in context["generatedPrefix"]]
        forced_trajectory = [int(value) for value in context["forcedTrajectory"]]
        case_dir = workdir / "cases" / name
        reference_dir = case_dir / "reference_checkpoints"
        candidate_dir = case_dir / "litenn_checkpoints"
        comparison_dir = case_dir / "comparison"
        if not args.reuse_artifacts:
            shutil.rmtree(reference_dir, ignore_errors=True)
            shutil.rmtree(candidate_dir, ignore_errors=True)
            prompt_text = ",".join(str(token) for token in prompt)
            prefix_text = ",".join(str(token) for token in generated_prefix) or "-"
            reference_command = [
                str(args.llamacpp_tokenizer_tool.resolve()),
                "decode-layer-checkpoints",
                str(args.model.resolve()),
                prompt_text,
                prefix_text,
                str(divergence),
                str(reference_dir),
            ]
            run_logged(f"{name}_reference", reference_command, case_dir / "logs")

            candidate_command = [
                sys.executable,
                str(ROOT / "example" / "gguf" / "qwen_smoke.py"),
                "--model",
                str(args.model.resolve()),
                "--litenn",
                str(args.litenn.resolve()),
                "--token-ids",
                prompt_text,
                "--stateful",
                "--max-tokens",
                str(divergence + 1),
                "--ignore-eos",
                "--forced-generated-token-ids",
                ",".join(str(token) for token in forced_trajectory),
                "--layer-checkpoint-dir",
                str(candidate_dir),
                "--layer-checkpoint-generated-indices",
                str(divergence),
                "--max-cache-length",
                str(args.max_cache_length),
                "--llvm-opt-level",
                str(args.llvm_opt_level),
                "--memory-sample-interval-ms",
                "0",
                "--workdir",
                str(case_dir / "litenn_smoke"),
            ]
            if args.aot_cache_dir is not None:
                candidate_command.extend(["--aot-cache-dir", str(args.aot_cache_dir.resolve())])
            if args.require_aot_cache_hit:
                candidate_command.append("--require-aot-cache-hit")
            if args.no_aot_cache_write:
                candidate_command.append("--no-aot-cache-write")
            if args.cpu_aot_threads is not None:
                candidate_command.extend(["--cpu-aot-threads", str(args.cpu_aot_threads)])
            run_logged(f"{name}_litenn", candidate_command, case_dir / "logs")

        comparison = layer_compare.compare_manifests(
            reference_dir / "manifest.tsv",
            candidate_dir / "manifest.tsv",
            args.absolute_tolerance,
            args.relative_tolerance,
            {divergence},
        )
        comparison_dir.mkdir(parents=True, exist_ok=True)
        comparison_json = comparison_dir / "layer_checkpoint_comparison.json"
        comparison_json.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
        (comparison_dir / "layer_checkpoint_comparison.md").write_text(
            layer_compare.markdown_report(comparison), encoding="utf-8"
        )
        case_report: dict[str, object] = {
            "name": name,
            "firstDivergenceDecisionStep": divergence,
            "promptTokenCount": len(prompt),
            "referencePrefixTokenCount": len(generated_prefix),
            "comparisonReport": str(comparison_json.relative_to(workdir)),
            "summary": summarize_comparison(comparison, divergence),
        }
        if sub_layer_blocks:
            sub_reference_dir = case_dir / "reference_sub_layers"
            sub_candidate_dir = case_dir / "litenn_sub_layers"
            sub_comparison_dir = case_dir / "sub_layer_comparison"
            if not args.reuse_artifacts:
                shutil.rmtree(sub_reference_dir, ignore_errors=True)
                shutil.rmtree(sub_candidate_dir, ignore_errors=True)
                sub_reference_command = [
                    str(args.llamacpp_tokenizer_tool.resolve()),
                    "decode-sub-layer-checkpoints",
                    str(args.model.resolve()),
                    prompt_text,
                    prefix_text,
                    str(divergence),
                    ",".join(str(block) for block in sub_layer_blocks),
                    str(sub_reference_dir),
                ]
                run_logged(f"{name}_sub_layer_reference", sub_reference_command, case_dir / "logs")

                sub_candidate_command = candidate_command.copy()
                sub_candidate_command[sub_candidate_command.index("--layer-checkpoint-dir") + 1] = str(
                    sub_candidate_dir
                )
                sub_candidate_command[sub_candidate_command.index("--workdir") + 1] = str(
                    case_dir / "litenn_sub_layer_smoke"
                )
                sub_candidate_command.extend(
                    [
                        "--sub-layer-checkpoint-blocks",
                        ",".join(str(block) for block in sub_layer_blocks),
                        "--paged-reference-decode",
                    ]
                )
                run_logged(f"{name}_sub_layer_litenn", sub_candidate_command, case_dir / "logs")
            case_report["subLayerSummary"] = summarize_sub_layers(
                sub_reference_dir,
                sub_candidate_dir,
                divergence,
                sub_layer_blocks,
                args.absolute_tolerance,
                args.relative_tolerance,
                sub_comparison_dir,
            )
        case_reports.append(case_report)

    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "qualityReport": str(quality_report),
        "sameContextReference": "llama.cpp reference token history",
        "cases": case_reports,
    }
    output = workdir / "first_divergence_attribution.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = workdir / "first_divergence_attribution.md"
    markdown.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report), end="")
    print(f"report={output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
