#!/usr/bin/env python3
"""Run paired baseline/profile stage controls across llama.cpp builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

try:
    from .run_llama_cpp_completion_control import host_metadata, sha256_file
    from .run_paired_gguf_decode_control import (
        power_policy,
        redact_text,
        run_monitored,
        series_statistics,
    )
except ImportError:
    from run_llama_cpp_completion_control import host_metadata, sha256_file
    from run_paired_gguf_decode_control import (
        power_policy,
        redact_text,
        run_monitored,
        series_statistics,
    )


SUMMARY_RE = re.compile(
    r"^mode=(?P<mode>[\w.-]+) threads=(?P<threads>\d+) warmup=(?P<warmup>\d+) steps=(?P<steps>\d+) "
    r"mean_decode_ms=(?P<mean_ms>[0-9.]+) tokens_per_second=(?P<tps>[0-9.]+)\r?$",
    re.MULTILINE,
)
STAGE_RE = re.compile(
    r"^stage=(?P<stage>\S+) ms_per_token=(?P<ms>[0-9.]+) calls_per_token=(?P<calls>[0-9.]+) "
    r"percent_of_decode=(?P<percent>[0-9.]+)\r?$",
    re.MULTILINE,
)
NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")
AGGREGATE_STAGES = {"attention", "ffn.gate_up", "ffn.down", "logits"}


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def non_negative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return value


def token_ids(raw: str) -> list[int]:
    try:
        values = [int(value) for value in raw.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("token ids must be comma-separated integers") from exc
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("token ids must be non-empty and non-negative")
    return values


def token_ids_digest(values: list[int]) -> str:
    return hashlib.sha256(",".join(str(value) for value in values).encode("ascii")).hexdigest()


def parse_registry(entries: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for entry in entries:
        name, separator, value = entry.partition("=")
        name = name.strip()
        value = value.strip()
        if not separator or not NAME_RE.fullmatch(name) or not value:
            raise SystemExit(f"invalid --binary value {entry!r}; expected NAME=PATH")
        if name in result:
            raise SystemExit(f"duplicate binary name {name!r}")
        path = Path(value).resolve()
        if not path.is_file():
            raise SystemExit(f"stage profiler not found: {path}")
        result[name] = path
    return result


def parse_output(stdout: str, expected_mode: str) -> dict[str, object]:
    stdout = stdout.replace("\r\n", "\n").replace("\r", "\n")
    match = SUMMARY_RE.search(stdout)
    if match is None:
        raise RuntimeError("stage profiler summary line was not found")
    if match.group("mode") != expected_mode:
        raise RuntimeError(f"expected mode {expected_mode!r}, got {match.group('mode')!r}")
    stages: dict[str, object] = {}
    for stage_match in STAGE_RE.finditer(stdout):
        name = stage_match.group("stage")
        if name in stages:
            raise RuntimeError(f"duplicate stage {name!r}")
        stages[name] = {
            "ms_per_token": float(stage_match.group("ms")),
            "calls_per_token": float(stage_match.group("calls")),
            "percent_of_decode": float(stage_match.group("percent")),
        }
    if expected_mode != "baseline" and not stages:
        raise RuntimeError(f"profile mode {expected_mode!r} reported no stages")
    return {
        "mode": match.group("mode"),
        "threads": int(match.group("threads")),
        "warmup": int(match.group("warmup")),
        "steps": int(match.group("steps")),
        "mean_decode_ms": float(match.group("mean_ms")),
        "tokens_per_second": float(match.group("tps")),
        "stages": stages,
    }


def frequency_value(process: dict[str, object]) -> float | None:
    frequency = process["frequency"]
    if not frequency["available"]:  # type: ignore[index]
        return None
    value = frequency.get("weighted_actual_mhz_median")  # type: ignore[union-attr]
    return float(value) if value is not None else None


def run_profile(
    binary: Path,
    model: Path,
    mode: str,
    args: argparse.Namespace,
    artifact_prefix: Path,
    replacements: dict[str, str],
) -> dict[str, object]:
    command = [
        str(binary),
        str(model),
        mode,
        str(args.threads),
        str(args.warmup),
        str(args.steps),
    ]
    if args.prefill_token_ids is not None:
        command.extend(
            [
                "--prefill-token-ids",
                ",".join(str(value) for value in args.prefill_token_ids),
                "--decode-token-ids",
                ",".join(str(value) for value in args.decode_token_ids),
            ]
        )
    process, stdout, stderr = run_monitored(command, artifact_prefix, replacements, args.monitor_interval_seconds)
    if process["returncode"] != 0:
        raise RuntimeError(
            f"stage profiler failed with {process['returncode']}: "
            f"{redact_text(stderr[-1000:], replacements)}"
        )
    metrics = parse_output(stdout, mode)
    return {"process": process, "metrics": metrics}


def normalize_pair(baseline: dict[str, object], profile: dict[str, object]) -> dict[str, object]:
    baseline_ms = float(baseline["metrics"]["mean_decode_ms"])  # type: ignore[index]
    profile_ms = float(profile["metrics"]["mean_decode_ms"])  # type: ignore[index]
    scale = baseline_ms / profile_ms
    stages = profile["metrics"]["stages"]  # type: ignore[index]
    normalized_stages = {
        name: {
            "ms_per_token": float(stage["ms_per_token"]) * scale,
            "calls_per_token": stage["calls_per_token"],
            "raw_ms_per_token": stage["ms_per_token"],
        }
        for name, stage in stages.items()  # type: ignore[union-attr]
    }
    return {
        "baseline": baseline,
        "profile": profile,
        "profile_overhead_percent": (profile_ms / baseline_ms - 1.0) * 100.0,
        "normalization_scale": scale,
        "normalized_stages": normalized_stages,
        "stage_coverage_percent": 100.0
        * sum(float(stage["raw_ms_per_token"]) for stage in normalized_stages.values())
        / profile_ms,
    }


def summarize_pairs(name: str, mode: str, pairs: list[dict[str, object]]) -> dict[str, object]:
    baseline_values = [float(pair["baseline"]["metrics"]["mean_decode_ms"]) for pair in pairs]  # type: ignore[index]
    profile_values = [float(pair["profile"]["metrics"]["mean_decode_ms"]) for pair in pairs]  # type: ignore[index]
    overhead_values = [float(pair["profile_overhead_percent"]) for pair in pairs]
    baseline_frequency = [
        value
        for pair in pairs
        if (value := frequency_value(pair["baseline"]["process"])) is not None  # type: ignore[index]
    ]
    profile_frequency = [
        value
        for pair in pairs
        if (value := frequency_value(pair["profile"]["process"])) is not None  # type: ignore[index]
    ]
    stage_names = sorted(
        set.intersection(*(set(pair["normalized_stages"]) for pair in pairs))  # type: ignore[arg-type]
    )
    stages = {
        stage: series_statistics(
            [float(pair["normalized_stages"][stage]["ms_per_token"]) for pair in pairs]  # type: ignore[index]
        )
        for stage in stage_names
    }
    coverage_values = [float(pair["stage_coverage_percent"]) for pair in pairs]
    return {
        "binary": name,
        "mode": mode,
        "baseline_ms_per_token": series_statistics(baseline_values),
        "profile_ms_per_token": series_statistics(profile_values),
        "profile_overhead_percent": series_statistics(overhead_values),
        "baseline_weighted_actual_mhz": series_statistics(baseline_frequency) if baseline_frequency else None,
        "profile_weighted_actual_mhz": series_statistics(profile_frequency) if profile_frequency else None,
        "normalized_stages": stages,
        "stage_coverage_percent": series_statistics(coverage_values),
    }


def write_markdown(path: Path, document: dict[str, object]) -> None:
    configuration = document["configuration"]
    gate = document["gate"]
    lines = [
        "# llama.cpp Paired Stage Control",
        "",
        f"- Host: `{document['host']['cpu_model']}`",  # type: ignore[index]
        f"- Threads: `{configuration['threads']}`",  # type: ignore[index]
        f"- Warmup/steps: `{configuration['warmup']}/{configuration['steps']}`",  # type: ignore[index]
        f"- Repetitions: `{configuration['repetitions']}`",  # type: ignore[index]
        f"- Accepted: `{gate['accepted']}`",  # type: ignore[index]
        "",
        "Binaries:",
        "",
    ]
    for name, binary in document["binaries"].items():  # type: ignore[union-attr]
        lines.append(
            f"- `{name}`: baseline `{binary['baseline_sha256']}`, profile `{binary['profile_sha256']}`"
        )
    lines.extend(
        [
            "",
            "| Binary | Mode | Baseline ms | Profile ms | Overhead | Coverage | Baseline MHz | Profile MHz |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for summary in document["summaries"]:  # type: ignore[union-attr]
        baseline_frequency = summary["baseline_weighted_actual_mhz"]
        profile_frequency = summary["profile_weighted_actual_mhz"]
        lines.append(
            f"| {summary['binary']} | {summary['mode']} | "
            f"{summary['baseline_ms_per_token']['median']:.3f} | "
            f"{summary['profile_ms_per_token']['median']:.3f} | "
            f"{summary['profile_overhead_percent']['median']:.2f}% | "
            f"{summary['stage_coverage_percent']['median']:.2f}% | "
            f"{baseline_frequency['median']:.0f} | "
            f"{profile_frequency['median']:.0f} |"
            if baseline_frequency is not None and profile_frequency is not None
            else f"| {summary['binary']} | {summary['mode']} | "
            f"{summary['baseline_ms_per_token']['median']:.3f} | "
            f"{summary['profile_ms_per_token']['median']:.3f} | "
            f"{summary['profile_overhead_percent']['median']:.2f}% | "
            f"{summary['stage_coverage_percent']['median']:.2f}% | n/a | n/a |"
        )
    for summary in document["summaries"]:  # type: ignore[union-attr]
        lines.extend(
            [
                "",
                f"## {summary['binary']} / {summary['mode']}",
                "",
                "| Normalized stage | Median ms/token | CV |",
                "| --- | ---: | ---: |",
            ]
        )
        for stage, stats in summary["normalized_stages"].items():
            lines.append(
                f"| {stage} | {stats['median']:.3f} | {stats['coefficient_of_variation_percent']:.2f}% |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--binary", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument(
        "--baseline-binary",
        action="append",
        metavar="NAME=PATH",
        help="Optional clean baseline binary for each instrumented --binary name",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--mode", action="append", help="coarse, ffn, layer-N, or scan-layer-N; may be repeated"
    )
    parser.add_argument("--threads", default=2, type=positive_int)
    parser.add_argument("--warmup", default=9, type=non_negative_int)
    parser.add_argument("--steps", default=15, type=positive_int)
    parser.add_argument("--repetitions", default=3, type=positive_int)
    parser.add_argument("--prefill-token-ids", type=token_ids)
    parser.add_argument("--decode-token-ids", type=token_ids)
    parser.add_argument("--variance-threshold-percent", default=3.0, type=positive_float)
    parser.add_argument("--stage-variance-threshold-percent", default=15.0, type=positive_float)
    parser.add_argument("--overhead-threshold-percent", default=3.0, type=positive_float)
    parser.add_argument("--minimum-stage-coverage-percent", default=95.0, type=positive_float)
    parser.add_argument("--maximum-stage-coverage-percent", default=102.0, type=positive_float)
    parser.add_argument("--monitor-interval-seconds", default=0.25, type=positive_float)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.repetitions < 3:
        raise SystemExit("stage control requires at least three repetitions")
    if (args.prefill_token_ids is None) != (args.decode_token_ids is None):
        raise SystemExit("--prefill-token-ids and --decode-token-ids must be supplied together")
    if args.decode_token_ids is not None and (
        args.warmup != 0 or len(args.decode_token_ids) != args.steps
    ):
        raise SystemExit("exact token replay requires --warmup 0 and one decode token per measured step")
    binaries = parse_registry(args.binary)
    baseline_binaries = parse_registry(args.baseline_binary) if args.baseline_binary else binaries
    if set(baseline_binaries) != set(binaries):
        raise SystemExit("--baseline-binary names must exactly match --binary names")
    model = args.model.resolve()
    if not model.is_file():
        raise SystemExit(f"model file not found: {model}")
    modes = args.mode or ["coarse", "ffn"]
    invalid_modes = [
        mode
        for mode in modes
        if mode not in {"aggregate", "coarse", "ffn"}
        and not re.fullmatch(r"(?:scan-)?layer-[1-9]\d*", mode)
    ]
    if invalid_modes:
        raise SystemExit(f"invalid profile modes: {', '.join(invalid_modes)}")
    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    replacements = {str(model): "<model>", str(Path.cwd()): "<repo>"}
    if args.prefill_token_ids is not None:
        replacements[",".join(str(value) for value in args.prefill_token_ids)] = "<prefill-token-ids>"
        replacements[",".join(str(value) for value in args.decode_token_ids)] = "<decode-token-ids>"
    replacements.update({str(path): f"<stage-profiler:{name}>" for name, path in binaries.items()})
    replacements.update({str(path.parent): f"<stage-profiler-dir:{name}>" for name, path in binaries.items()})
    replacements.update(
        {str(path): f"<baseline-stage-profiler:{name}>" for name, path in baseline_binaries.items()}
    )
    replacements.update(
        {str(path.parent): f"<baseline-stage-profiler-dir:{name}>" for name, path in baseline_binaries.items()}
    )

    document: dict[str, object] = {
        "schema_version": 1,
        "tool": "llama-cpp-paired-stage-control",
        "status": "running",
        "host": host_metadata(),
        "power_policy": power_policy(),
        "binaries": {
            name: {
                "baseline_path": f"<baseline-stage-profiler:{name}>",
                "baseline_sha256": sha256_file(baseline_binaries[name]),
                "profile_path": f"<stage-profiler:{name}>",
                "profile_sha256": sha256_file(path),
            }
            for name, path in binaries.items()
        },
        "model": {"filename": "<model>", "size_bytes": model.stat().st_size},
        "configuration": {
            "threads": args.threads,
            "warmup": args.warmup,
            "steps": args.steps,
            "repetitions": args.repetitions,
            "modes": modes,
            "variance_threshold_percent": args.variance_threshold_percent,
            "stage_variance_threshold_percent": args.stage_variance_threshold_percent,
            "overhead_threshold_percent": args.overhead_threshold_percent,
            "minimum_stage_coverage_percent": args.minimum_stage_coverage_percent,
            "maximum_stage_coverage_percent": args.maximum_stage_coverage_percent,
            "exact_token_replay": (
                {
                    "prefill_count": len(args.prefill_token_ids),
                    "prefill_sha256": token_ids_digest(args.prefill_token_ids),
                    "decode_count": len(args.decode_token_ids),
                    "decode_sha256": token_ids_digest(args.decode_token_ids),
                }
                if args.prefill_token_ids is not None
                else None
            ),
        },
        "pairs": [],
    }
    output_json.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    try:
        for repetition in range(1, args.repetitions + 1):
            binary_names = list(binaries)
            mode_order = list(modes)
            if repetition % 2 == 0:
                binary_names.reverse()
                mode_order.reverse()
            for binary_name in binary_names:
                for mode in mode_order:
                    execution_order = ["baseline", mode] if repetition % 2 == 1 else [mode, "baseline"]
                    print(
                        f"[stage-control] repetition={repetition}/{args.repetitions} binary={binary_name} "
                        f"mode={mode} order={'->'.join(execution_order)}",
                        flush=True,
                    )
                    runs: dict[str, dict[str, object]] = {}
                    for executed_mode in execution_order:
                        artifact_prefix = (
                            output_json.parent
                            / f"rep_{repetition:02d}_{binary_name}_{mode}_{executed_mode}"
                        )
                        runs[executed_mode] = run_profile(
                            baseline_binaries[binary_name]
                            if executed_mode == "baseline"
                            else binaries[binary_name],
                            model,
                            executed_mode,
                            args,
                            artifact_prefix,
                            replacements,
                        )
                    pair = normalize_pair(runs["baseline"], runs[mode])
                    pair.update(
                        {
                            "repetition": repetition,
                            "binary": binary_name,
                            "mode": mode,
                            "order": execution_order,
                        }
                    )
                    document["pairs"].append(pair)  # type: ignore[union-attr]
                    output_json.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        document["status"] = "failed"
        document["error"] = redact_text(str(exc), replacements)
        output_json.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        raise

    summaries = []
    for binary_name in binaries:
        for mode in modes:
            pairs = [
                pair
                for pair in document["pairs"]  # type: ignore[union-attr]
                if pair["binary"] == binary_name and pair["mode"] == mode
            ]
            summaries.append(summarize_pairs(binary_name, mode, pairs))
    variance_ok = all(
        summary[metric]["coefficient_of_variation_percent"] <= args.variance_threshold_percent
        for summary in summaries
        for metric in ("baseline_ms_per_token", "profile_ms_per_token")
    )
    overhead_ok = all(
        abs(summary["profile_overhead_percent"]["median"]) <= args.overhead_threshold_percent
        for summary in summaries
    )
    stage_variance_ok = all(
        stats["coefficient_of_variation_percent"] <= args.stage_variance_threshold_percent
        for summary in summaries
        for stats in summary["normalized_stages"].values()
    )
    aggregate_shape_ok = all(
        summary["mode"] != "aggregate" or set(summary["normalized_stages"]) == AGGREGATE_STAGES
        for summary in summaries
    )
    stage_coverage_ok = all(
        args.minimum_stage_coverage_percent
        <= summary["stage_coverage_percent"]["median"]
        <= args.maximum_stage_coverage_percent
        for summary in summaries
    )
    document["summaries"] = summaries
    document["gate"] = {
        "variance": variance_ok,
        "stage_variance": stage_variance_ok,
        "profile_overhead": overhead_ok,
        "aggregate_shape": aggregate_shape_ok,
        "stage_coverage": stage_coverage_ok,
        "accepted": variance_ok
        and stage_variance_ok
        and overhead_ok
        and aggregate_shape_ok
        and stage_coverage_ok,
    }
    document["status"] = "complete"
    document["power_policy_after"] = power_policy()
    output_json.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    if args.output_md is not None:
        write_markdown(args.output_md.resolve(), document)
    return 0 if document["gate"]["accepted"] else 1  # type: ignore[index]


if __name__ == "__main__":
    raise SystemExit(main())
