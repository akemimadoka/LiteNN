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
        process_power_policy_stable,
        redact_text,
        run_monitored,
        series_statistics,
    )
except ImportError:
    from run_llama_cpp_completion_control import host_metadata, sha256_file
    from run_paired_gguf_decode_control import (
        power_policy,
        process_power_policy_stable,
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
DECODE_STEP_RE = re.compile(
    r"^decode_step=(?P<step>\d+) decode_ms=(?P<ms>[0-9.]+)\r?$", re.MULTILINE
)
STAGE_STEP_RE = re.compile(
    r"^stage_step=(?P<step>\d+) stage=(?P<stage>\S+) stage_ms=(?P<ms>[0-9.]+) "
    r"calls=(?P<calls>\d+)\r?$",
    re.MULTILINE,
)
NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")
AGGREGATE_STAGES = {"attention", "ffn.gate_up", "ffn.activation", "ffn.down", "logits"}


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


def position_bins(raw: str) -> list[tuple[int, int]]:
    bins: list[tuple[int, int]] = []
    expected_start = 1
    for value in raw.split(","):
        match = re.fullmatch(r"(?P<start>[1-9]\d*)-(?P<end>[1-9]\d*)", value.strip())
        if match is None:
            raise argparse.ArgumentTypeError("position bins must use START-END comma-separated ranges")
        start = int(match.group("start"))
        end = int(match.group("end"))
        if start != expected_start or end < start:
            raise argparse.ArgumentTypeError("position bins must be ordered, contiguous, and start at 1")
        bins.append((start, end))
        expected_start = end + 1
    return bins


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
    decode_steps = {
        int(step_match.group("step")): float(step_match.group("ms"))
        for step_match in DECODE_STEP_RE.finditer(stdout)
    }
    if decode_steps and sorted(decode_steps) != list(range(1, int(match.group("steps")) + 1)):
        raise RuntimeError("decode-step records are not contiguous")
    stage_steps: dict[int, dict[str, dict[str, float | int]]] = {}
    for step_match in STAGE_STEP_RE.finditer(stdout):
        step = int(step_match.group("step"))
        stage = step_match.group("stage")
        step_stages = stage_steps.setdefault(step, {})
        if stage in step_stages:
            raise RuntimeError(f"duplicate stage-step record for step {step}, stage {stage!r}")
        step_stages[stage] = {
            "ms": float(step_match.group("ms")),
            "calls": int(step_match.group("calls")),
        }
    if stage_steps:
        expected_steps = list(range(1, int(match.group("steps")) + 1))
        if sorted(stage_steps) != expected_steps:
            raise RuntimeError("stage-step records are not contiguous")
        if any(set(step_stages) != AGGREGATE_STAGES for step_stages in stage_steps.values()):
            raise RuntimeError("stage-step records do not contain the aggregate stage set")
    return {
        "mode": match.group("mode"),
        "threads": int(match.group("threads")),
        "warmup": int(match.group("warmup")),
        "steps": int(match.group("steps")),
        "mean_decode_ms": float(match.group("mean_ms")),
        "tokens_per_second": float(match.group("tps")),
        "stages": stages,
        "decode_steps": [decode_steps[step] for step in sorted(decode_steps)],
        "stage_steps": [
            {"step": step, "stages": stage_steps[step]} for step in sorted(stage_steps)
        ],
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


def normalize_position_bins(
    baseline: dict[str, object],
    profile: dict[str, object],
    bins: list[tuple[int, int]],
) -> list[dict[str, object]]:
    baseline_steps = baseline["metrics"]["decode_steps"]  # type: ignore[index]
    profile_steps = profile["metrics"]["decode_steps"]  # type: ignore[index]
    profile_stage_steps = profile["metrics"]["stage_steps"]  # type: ignore[index]
    if not baseline_steps or not profile_steps or not profile_stage_steps:
        raise RuntimeError("position bins require decode-step and aggregate stage-step records")
    if len(baseline_steps) != len(profile_steps) or len(profile_steps) != len(profile_stage_steps):
        raise RuntimeError("baseline/profile position-step counts differ")
    if bins[-1][1] != len(baseline_steps):
        raise RuntimeError("position bins must cover every measured decode step")

    result: list[dict[str, object]] = []
    for start, end in bins:
        count = end - start + 1
        baseline_ms = sum(float(value) for value in baseline_steps[start - 1 : end]) / count
        profile_ms = sum(float(value) for value in profile_steps[start - 1 : end]) / count
        scale = baseline_ms / profile_ms
        stage_names = set.intersection(
            *(
                set(step["stages"])  # type: ignore[index]
                for step in profile_stage_steps[start - 1 : end]
            )
        )
        stages: dict[str, object] = {}
        for stage in sorted(stage_names):
            raw_ms = (
                sum(
                    float(step["stages"][stage]["ms"])  # type: ignore[index]
                    for step in profile_stage_steps[start - 1 : end]
                )
                / count
            )
            calls = (
                sum(
                    float(step["stages"][stage]["calls"])  # type: ignore[index]
                    for step in profile_stage_steps[start - 1 : end]
                )
                / count
            )
            stages[stage] = {
                "ms_per_token": raw_ms * scale,
                "raw_ms_per_token": raw_ms,
                "calls_per_token": calls,
            }
        result.append(
            {
                "start": start,
                "end": end,
                "baseline_ms_per_token": baseline_ms,
                "profile_ms_per_token": profile_ms,
                "profile_overhead_percent": (profile_ms / baseline_ms - 1.0) * 100.0,
                "normalization_scale": scale,
                "normalized_stages": stages,
                "stage_coverage_percent": 100.0
                * sum(float(stage["raw_ms_per_token"]) for stage in stages.values())
                / profile_ms,
            }
        )
    return result


def normalize_pair(
    baseline: dict[str, object],
    profile: dict[str, object],
    bins: list[tuple[int, int]] | None = None,
) -> dict[str, object]:
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
    result = {
        "baseline": baseline,
        "profile": profile,
        "profile_overhead_percent": (profile_ms / baseline_ms - 1.0) * 100.0,
        "normalization_scale": scale,
        "normalized_stages": normalized_stages,
        "stage_coverage_percent": 100.0
        * sum(float(stage["raw_ms_per_token"]) for stage in normalized_stages.values())
        / profile_ms,
    }
    result["position_bins"] = normalize_position_bins(baseline, profile, bins) if bins else []
    return result


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
    position_bin_summaries: list[dict[str, object]] = []
    if pairs[0]["position_bins"]:
        bin_count = len(pairs[0]["position_bins"])  # type: ignore[arg-type]
        if any(len(pair["position_bins"]) != bin_count for pair in pairs):  # type: ignore[arg-type]
            raise RuntimeError("position-bin count differs across repetitions")
        for bin_index in range(bin_count):
            bin_values = [pair["position_bins"][bin_index] for pair in pairs]  # type: ignore[index]
            stage_names_for_bin = sorted(
                set.intersection(*(set(value["normalized_stages"]) for value in bin_values))  # type: ignore[arg-type]
            )
            position_bin_summaries.append(
                {
                    "start": bin_values[0]["start"],  # type: ignore[index]
                    "end": bin_values[0]["end"],  # type: ignore[index]
                    "baseline_ms_per_token": series_statistics(
                        [float(value["baseline_ms_per_token"]) for value in bin_values]  # type: ignore[index]
                    ),
                    "profile_ms_per_token": series_statistics(
                        [float(value["profile_ms_per_token"]) for value in bin_values]  # type: ignore[index]
                    ),
                    "profile_overhead_percent": series_statistics(
                        [float(value["profile_overhead_percent"]) for value in bin_values]  # type: ignore[index]
                    ),
                    "stage_coverage_percent": series_statistics(
                        [float(value["stage_coverage_percent"]) for value in bin_values]  # type: ignore[index]
                    ),
                    "normalized_stages": {
                        stage: series_statistics(
                            [
                                float(value["normalized_stages"][stage]["ms_per_token"])  # type: ignore[index]
                                for value in bin_values
                            ]
                        )
                        for stage in stage_names_for_bin
                    },
                }
            )
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
        "position_bins": position_bin_summaries,
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
        f"- Power-policy stability: `{gate['power_policy_stability']}`",  # type: ignore[index]
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
        if summary["position_bins"]:
            lines.extend(
                [
                    "",
                    "### Position bins",
                    "",
                    "| Positions | Baseline ms | Profile ms | Overhead | Coverage |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for position_bin in summary["position_bins"]:
                lines.append(
                    f"| {position_bin['start']}-{position_bin['end']} | "
                    f"{position_bin['baseline_ms_per_token']['median']:.3f} | "
                    f"{position_bin['profile_ms_per_token']['median']:.3f} | "
                    f"{position_bin['profile_overhead_percent']['median']:.2f}% | "
                    f"{position_bin['stage_coverage_percent']['median']:.2f}% |"
                )
            lines.extend(
                [
                    "",
                    "| Positions | Stage | Median ms/token | CV |",
                    "| --- | --- | ---: | ---: |",
                ]
            )
            for position_bin in summary["position_bins"]:
                for stage, stats in position_bin["normalized_stages"].items():
                    lines.append(
                        f"| {position_bin['start']}-{position_bin['end']} | {stage} | "
                        f"{stats['median']:.3f} | {stats['coefficient_of_variation_percent']:.2f}% |"
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
    parser.add_argument(
        "--position-bins",
        type=position_bins,
        help="Contiguous one-based decode-step ranges, for example 1-16,17-48,49-80,81-112,113-128",
    )
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
    if args.position_bins is not None:
        if modes != ["aggregate"]:
            raise SystemExit("--position-bins requires exactly one --mode aggregate")
        if args.position_bins[-1][1] != args.steps:
            raise SystemExit("--position-bins must cover exactly --steps decode steps")
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
            "position_bins": (
                [{"start": start, "end": end} for start, end in args.position_bins]
                if args.position_bins is not None
                else []
            ),
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
                    pair = normalize_pair(runs["baseline"], runs[mode], args.position_bins)
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
    position_bin_variance_ok = all(
        stats["coefficient_of_variation_percent"] <= args.variance_threshold_percent
        for summary in summaries
        for position_bin in summary["position_bins"]
        for stats in (position_bin["baseline_ms_per_token"], position_bin["profile_ms_per_token"])
    )
    position_bin_stage_variance_ok = all(
        stats["coefficient_of_variation_percent"] <= args.stage_variance_threshold_percent
        for summary in summaries
        for position_bin in summary["position_bins"]
        for stats in position_bin["normalized_stages"].values()
    )
    position_bin_overhead_ok = all(
        abs(position_bin["profile_overhead_percent"]["median"]) <= args.overhead_threshold_percent
        for summary in summaries
        for position_bin in summary["position_bins"]
    )
    position_bin_coverage_ok = all(
        args.minimum_stage_coverage_percent
        <= position_bin["stage_coverage_percent"]["median"]
        <= args.maximum_stage_coverage_percent
        for summary in summaries
        for position_bin in summary["position_bins"]
    )
    power_policy_stability_ok = all(
        process_power_policy_stable(pair[run]["process"])
        for pair in document["pairs"]  # type: ignore[union-attr]
        for run in ("baseline", "profile")
    )
    document["summaries"] = summaries
    document["gate"] = {
        "variance": variance_ok,
        "stage_variance": stage_variance_ok,
        "profile_overhead": overhead_ok,
        "aggregate_shape": aggregate_shape_ok,
        "stage_coverage": stage_coverage_ok,
        "position_bin_variance": position_bin_variance_ok,
        "position_bin_stage_variance": position_bin_stage_variance_ok,
        "position_bin_profile_overhead": position_bin_overhead_ok,
        "position_bin_stage_coverage": position_bin_coverage_ok,
        "power_policy_stability": power_policy_stability_ok,
        "accepted": variance_ok
        and stage_variance_ok
        and overhead_ok
        and aggregate_shape_ok
        and stage_coverage_ok
        and position_bin_variance_ok
        and position_bin_stage_variance_ok
        and position_bin_overhead_ok
        and position_bin_coverage_ok
        and power_policy_stability_ok,
    }
    document["status"] = "complete"
    document["power_policy_after"] = power_policy()
    output_json.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    if args.output_md is not None:
        write_markdown(args.output_md.resolve(), document)
    return 0 if document["gate"]["accepted"] else 1  # type: ignore[index]


if __name__ == "__main__":
    raise SystemExit(main())
