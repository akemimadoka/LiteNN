#!/usr/bin/env python3
"""Run paired clean/profile LiteNN GGUF decode controls across position bins."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

try:
    from .profile_bundle import GGUFHelperEvent, LogEvidence, parse_gguf_decode_logs
    from .run_llama_cpp_completion_control import host_metadata
    from .run_llama_cpp_stage_control import position_bins
    from .run_paired_gguf_decode_control import (
        binary_identity,
        load_litenn_generated_token_ids,
        parse_forced_replay_metrics,
        power_policy,
        process_power_policy_stable,
        redact_text,
        run_monitored,
        series_statistics,
        token_ids_identity,
    )
except ImportError:
    from profile_bundle import GGUFHelperEvent, LogEvidence, parse_gguf_decode_logs
    from run_llama_cpp_completion_control import host_metadata
    from run_llama_cpp_stage_control import position_bins
    from run_paired_gguf_decode_control import (
        binary_identity,
        load_litenn_generated_token_ids,
        parse_forced_replay_metrics,
        power_policy,
        process_power_policy_stable,
        redact_text,
        run_monitored,
        series_statistics,
        token_ids_identity,
    )


STAGE_ORDER = (
    "embedding",
    "attention.qkv",
    "attention.rope",
    "attention.kv_append",
    "attention.core",
    "attention.output",
    "ffn.gate_up",
    "ffn.activation",
    "ffn.down",
    "ffn.swiglu_down_fused",
    "normalization",
    "logits",
    "helper.other",
    "module.residual",
)
STAGE_SORT_KEY = {name: index for index, name in enumerate(STAGE_ORDER)}
SUB_MILLISECOND_STAGE_LIMIT_MS = 1.0
METRIC_RE = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^\s]+)")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return value


def non_negative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return value


def comma_token_ids(raw: str) -> list[int]:
    try:
        values = [int(part) for part in raw.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("token ids must be comma-separated integers") from error
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("token ids must be non-empty and non-negative")
    return values


def load_token_ids_file(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"token-id file is empty: {path}")
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        return comma_token_ids(text)
    values: object = document
    if isinstance(document, dict):
        for key in ("tokenIds", "generatedTokenIds"):
            if key in document:
                values = document[key]
                break
    if not isinstance(values, list) or not values or any(
        not isinstance(value, int) or value < 0 for value in values
    ):
        raise RuntimeError(f"token-id file contains no valid token-id array: {path}")
    return values


def token_ids_digest(values: list[int]) -> str:
    return hashlib.sha256(",".join(str(value) for value in values).encode("ascii")).hexdigest()


def stage_for_helper(event: GGUFHelperEvent) -> str:
    if "swiglu_ggml_block_matmul" in event.helper:
        return "ffn.swiglu_down_fused"
    key = (event.operator, event.role)
    return {
        ("embedding", "token_lookup"): "embedding",
        ("projection", "qkv_grouped"): "attention.qkv",
        ("position_encoding", "rope"): "attention.rope",
        ("kv_update", "append"): "attention.kv_append",
        ("attention", "active_prefix"): "attention.core",
        ("attention", "paged"): "attention.core",
        ("projection", "hidden_or_output"): "attention.output",
        ("projection", "ffn_gate_up_grouped"): "ffn.gate_up",
        ("projection", "ffn_gate_or_up"): "ffn.gate_up",
        ("activation", "swiglu"): "ffn.activation",
        ("projection", "ffn_down"): "ffn.down",
        ("normalization", "norm"): "normalization",
        ("projection", "logits"): "logits",
    }.get(key, "helper.other")


def parse_decode_metrics(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise RuntimeError(f"LiteNN decode output is missing its metrics row: {path}")
    return {match.group("name"): match.group("value") for match in METRIC_RE.finditer(lines[2])}


def parse_run_logs(stdout: Path, stderr: Path, expected_steps: int, profile: bool) -> dict[str, object]:
    analysis = parse_gguf_decode_logs([LogEvidence(name="litenn", stdout=stdout, stderr=stderr)])
    generation_steps = [step for step in analysis.steps if step.phase == "generation"]
    if len(generation_steps) != expected_steps:
        raise RuntimeError(
            f"expected {expected_steps} generation stream-stat records, found {len(generation_steps)}"
        )
    if [step.generated_tokens for step in generation_steps] != list(range(1, expected_steps + 1)):
        raise RuntimeError("generation stream-stat records do not have contiguous generated-token positions")

    helpers_by_step: dict[int, list[GGUFHelperEvent]] = {}
    for helper in analysis.helpers:
        helpers_by_step.setdefault(helper.step, []).append(helper)

    step_records: list[dict[str, object]] = []
    for position, step in enumerate(generation_steps, start=1):
        if step.module_run_ms is None:
            raise RuntimeError(f"generation position {position} has no module_run_ms")
        record: dict[str, object] = {
            "position": position,
            "runtime_step": step.step,
            "step_ms": step.step_ms,
            "module_ms": step.module_run_ms,
            "sampling_ms": step.sampling_ms,
            "helper_profile_emit_ms": step.helper_profile_emit_ms,
            "stages": {},
        }
        if profile:
            if not step.helper_profile_enabled or step.helper_total_ms is None or step.module_non_helper_ms is None:
                raise RuntimeError(f"profile generation position {position} has no helper accounting")
            events = helpers_by_step.get(step.step, [])
            if not events:
                raise RuntimeError(f"profile generation position {position} has no helper events")
            stages: dict[str, dict[str, float | int]] = {}
            for event in events:
                stage = stage_for_helper(event)
                aggregate = stages.setdefault(stage, {"ms": 0.0, "calls": 0})
                aggregate["ms"] = float(aggregate["ms"]) + event.total_ms
                aggregate["calls"] = int(aggregate["calls"]) + event.calls
            stages["module.residual"] = {"ms": step.module_non_helper_ms, "calls": 0}
            record["stages"] = stages
            record["helper_total_ms"] = step.helper_total_ms
            record["module_non_helper_ms"] = step.module_non_helper_ms
        step_records.append(record)

    mean_step_ms = sum(float(step["step_ms"]) for step in step_records) / len(step_records)
    mean_module_ms = sum(float(step["module_ms"]) for step in step_records) / len(step_records)
    return {
        "mean_step_ms": mean_step_ms,
        "mean_module_ms": mean_module_ms,
        "tokens_per_second": 1000.0 / mean_step_ms,
        "steps": step_records,
    }


def build_command(
    args: argparse.Namespace,
    model: Path,
    litenn: Path,
    prompt_ids: list[int],
    decode_ids: list[int],
    workdir: Path,
    output: Path,
    profile: bool,
) -> list[str]:
    command = [
        args.python,
        str(repo_root() / "example" / "gguf" / "qwen_smoke.py"),
        "--model",
        str(model),
        "--litenn",
        str(litenn),
        "--token-ids",
        ",".join(str(value) for value in prompt_ids),
        "--max-tokens",
        str(len(decode_ids)),
        "--forced-generated-token-ids",
        ",".join(str(value) for value in decode_ids),
        "--output",
        str(output),
        "--workdir",
        str(workdir),
        "--stateful",
        "--ignore-eos",
        "--stream-stats",
        "--aot-cache-dir",
        str(args.aot_cache_dir),
        "--require-aot-cache-hit",
        "--llvm-opt-level",
        str(args.llvm_opt_level),
        "--cpu-aot-threads",
        str(args.threads),
        "--cpu-aot-worker-wait",
        args.worker_wait,
        "--cpu-aot-ggml-prepacked-weight-policy",
        args.prepacked_weight_policy,
        "--cpu-aot-ggml-prepacked-weight-layout",
        args.prepacked_weight_layout,
    ]
    if args.affinity != "default":
        command.extend(("--cpu-aot-affinity", args.affinity))
    if args.max_cache_length is not None:
        command.extend(("--max-cache-length", str(args.max_cache_length)))
    if profile:
        command.append("--profile-helpers")
    return command


def run_litenn(
    args: argparse.Namespace,
    model: Path,
    litenn: Path,
    prompt_ids: list[int],
    decode_ids: list[int],
    profile: bool,
    artifact_prefix: Path,
    replacements: dict[str, str],
) -> dict[str, object]:
    mode = "profile" if profile else "clean"
    workdir = artifact_prefix.parent / f"{artifact_prefix.name}_{mode}_workdir"
    output = workdir / "generated_tokens.txt"
    command = build_command(args, model, litenn, prompt_ids, decode_ids, workdir, output, profile)
    process, _, stderr = run_monitored(
        command, artifact_prefix, replacements, args.monitor_interval_seconds
    )
    if process["returncode"] != 0:
        raise RuntimeError(
            f"LiteNN {mode} run failed with {process['returncode']}: "
            f"{redact_text(stderr[-2000:], replacements)}"
        )
    generated = load_litenn_generated_token_ids(output, len(prompt_ids))
    replay = parse_forced_replay_metrics(output)
    metrics_row = parse_decode_metrics(output)
    if generated != decode_ids:
        raise RuntimeError(f"LiteNN {mode} run did not preserve the forced token trajectory")
    if metrics_row.get("fallback_count") != "0":
        raise RuntimeError(f"LiteNN {mode} run reported fallback_count={metrics_row.get('fallback_count')}")
    stdout_path = artifact_prefix.with_suffix(".stdout.txt")
    stderr_path = artifact_prefix.with_suffix(".stderr.txt")
    metrics = parse_run_logs(stdout_path, stderr_path, len(decode_ids), profile)
    return {
        "process": process,
        "metrics": metrics,
        "tokens": token_ids_identity(generated),
        "forced_replay": replay,
        "fallback_count": 0,
        "aot_cache_hit_required": True,
    }


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def position_slice(steps: list[dict[str, object]], start: int, end: int) -> list[dict[str, object]]:
    return steps[start - 1 : end]


def normalized_stages(
    profile_steps: list[dict[str, object]], scale: float
) -> tuple[dict[str, dict[str, float]], float]:
    stage_names = set.intersection(*(set(step["stages"]) for step in profile_steps))  # type: ignore[arg-type]
    if any(set(step["stages"]) != stage_names for step in profile_steps):  # type: ignore[arg-type]
        raise RuntimeError("profile stage shape changes across generated positions")
    stages: dict[str, dict[str, float]] = {}
    for stage in sorted(stage_names, key=lambda value: STAGE_SORT_KEY.get(value, len(STAGE_SORT_KEY))):
        raw_ms = average([float(step["stages"][stage]["ms"]) for step in profile_steps])  # type: ignore[index]
        call_values = [float(step["stages"][stage]["calls"]) for step in profile_steps]  # type: ignore[index]
        if any(value != call_values[0] for value in call_values[1:]):
            raise RuntimeError(f"profile call count changes across generated positions for stage {stage!r}")
        calls = call_values[0]
        stages[stage] = {
            "ms_per_token": raw_ms * scale,
            "raw_ms_per_token": raw_ms,
            "calls_per_token": calls,
        }
    profile_module_ms = average([float(step["module_ms"]) for step in profile_steps])
    coverage = 100.0 * sum(stage["raw_ms_per_token"] for stage in stages.values()) / profile_module_ms
    return stages, coverage


def normalize_window(
    clean_steps: list[dict[str, object]], profile_steps: list[dict[str, object]]
) -> dict[str, object]:
    clean_step_ms = average([float(step["step_ms"]) for step in clean_steps])
    profile_step_ms = average([float(step["step_ms"]) for step in profile_steps])
    clean_module_ms = average([float(step["module_ms"]) for step in clean_steps])
    profile_module_ms = average([float(step["module_ms"]) for step in profile_steps])
    scale = clean_module_ms / profile_module_ms
    stages, coverage = normalized_stages(profile_steps, scale)
    return {
        "clean_step_ms_per_token": clean_step_ms,
        "profile_step_ms_per_token": profile_step_ms,
        "step_profile_overhead_percent": (profile_step_ms / clean_step_ms - 1.0) * 100.0,
        "clean_module_ms_per_token": clean_module_ms,
        "profile_module_ms_per_token": profile_module_ms,
        "module_profile_overhead_percent": (profile_module_ms / clean_module_ms - 1.0) * 100.0,
        "normalization_scale": scale,
        "normalized_stages": stages,
        "stage_coverage_percent": coverage,
    }


def normalize_pair(
    clean: dict[str, object], profile: dict[str, object], bins: list[tuple[int, int]]
) -> dict[str, object]:
    clean_steps = clean["metrics"]["steps"]  # type: ignore[index]
    profile_steps = profile["metrics"]["steps"]  # type: ignore[index]
    if len(clean_steps) != len(profile_steps) or bins[-1][1] != len(clean_steps):  # type: ignore[arg-type]
        raise RuntimeError("clean/profile step counts or position-bin coverage differ")
    result = normalize_window(clean_steps, profile_steps)  # type: ignore[arg-type]
    result["clean"] = clean
    result["profile"] = profile
    result["position_bins"] = [
        {
            "start": start,
            "end": end,
            **normalize_window(
                position_slice(clean_steps, start, end),  # type: ignore[arg-type]
                position_slice(profile_steps, start, end),  # type: ignore[arg-type]
            ),
        }
        for start, end in bins
    ]
    first = result["position_bins"][0]  # type: ignore[index]
    last = result["position_bins"][-1]  # type: ignore[index]
    growth: dict[str, object] = {
        "clean_module_ms": float(last["clean_module_ms_per_token"])
        - float(first["clean_module_ms_per_token"]),
        "clean_module_percent": (
            float(last["clean_module_ms_per_token"]) / float(first["clean_module_ms_per_token"]) - 1.0
        )
        * 100.0,
        "profile_module_ms": float(last["profile_module_ms_per_token"])
        - float(first["profile_module_ms_per_token"]),
        "profile_module_percent": (
            float(last["profile_module_ms_per_token"]) / float(first["profile_module_ms_per_token"]) - 1.0
        )
        * 100.0,
        "raw_stages": {},
    }
    common_stages = set(first["normalized_stages"]) & set(last["normalized_stages"])  # type: ignore[arg-type]
    for stage in sorted(common_stages, key=lambda value: STAGE_SORT_KEY.get(value, len(STAGE_SORT_KEY))):
        first_ms = float(first["normalized_stages"][stage]["raw_ms_per_token"])  # type: ignore[index]
        last_ms = float(last["normalized_stages"][stage]["raw_ms_per_token"])  # type: ignore[index]
        growth["raw_stages"][stage] = {  # type: ignore[index]
            "first_ms_per_token": first_ms,
            "last_ms_per_token": last_ms,
            "delta_ms_per_token": last_ms - first_ms,
            "delta_percent": (last_ms / first_ms - 1.0) * 100.0 if first_ms > 0.0 else 0.0,
        }
    result["first_to_last_growth"] = growth
    return result


def summarize_pairs(pairs: list[dict[str, object]]) -> dict[str, object]:
    scalar_metrics = (
        "clean_step_ms_per_token",
        "profile_step_ms_per_token",
        "step_profile_overhead_percent",
        "clean_module_ms_per_token",
        "profile_module_ms_per_token",
        "module_profile_overhead_percent",
        "stage_coverage_percent",
    )
    result: dict[str, object] = {
        metric: series_statistics([float(pair[metric]) for pair in pairs]) for metric in scalar_metrics
    }
    stage_names = set.intersection(*(set(pair["normalized_stages"]) for pair in pairs))  # type: ignore[arg-type]
    result["normalized_stages"] = {
        stage: {
            "ms_per_token": series_statistics(
                [float(pair["normalized_stages"][stage]["ms_per_token"]) for pair in pairs]  # type: ignore[index]
            ),
            "raw_ms_per_token": series_statistics(
                [float(pair["normalized_stages"][stage]["raw_ms_per_token"]) for pair in pairs]  # type: ignore[index]
            ),
            "calls_per_token": series_statistics(
                [float(pair["normalized_stages"][stage]["calls_per_token"]) for pair in pairs]  # type: ignore[index]
            ),
        }
        for stage in sorted(stage_names, key=lambda value: STAGE_SORT_KEY.get(value, len(STAGE_SORT_KEY)))
    }
    result["position_bins"] = []
    for index in range(len(pairs[0]["position_bins"])):  # type: ignore[arg-type]
        values = [pair["position_bins"][index] for pair in pairs]  # type: ignore[index]
        stages = set.intersection(*(set(value["normalized_stages"]) for value in values))  # type: ignore[arg-type]
        summary = {
            "start": values[0]["start"],  # type: ignore[index]
            "end": values[0]["end"],  # type: ignore[index]
            **{
                metric: series_statistics([float(value[metric]) for value in values])
                for metric in scalar_metrics
            },
            "normalized_stages": {
                stage: {
                    "ms_per_token": series_statistics(
                        [float(value["normalized_stages"][stage]["ms_per_token"]) for value in values]  # type: ignore[index]
                    ),
                    "raw_ms_per_token": series_statistics(
                        [float(value["normalized_stages"][stage]["raw_ms_per_token"]) for value in values]  # type: ignore[index]
                    ),
                    "calls_per_token": series_statistics(
                        [float(value["normalized_stages"][stage]["calls_per_token"]) for value in values]  # type: ignore[index]
                    ),
                }
                for stage in sorted(stages, key=lambda value: STAGE_SORT_KEY.get(value, len(STAGE_SORT_KEY)))
            },
        }
        result["position_bins"].append(summary)  # type: ignore[union-attr]
    result["first_to_last_growth"] = {
        "clean_module_ms": series_statistics(
            [float(pair["first_to_last_growth"]["clean_module_ms"]) for pair in pairs]  # type: ignore[index]
        ),
        "clean_module_percent": series_statistics(
            [float(pair["first_to_last_growth"]["clean_module_percent"]) for pair in pairs]  # type: ignore[index]
        ),
        "profile_module_ms": series_statistics(
            [float(pair["first_to_last_growth"]["profile_module_ms"]) for pair in pairs]  # type: ignore[index]
        ),
        "profile_module_percent": series_statistics(
            [float(pair["first_to_last_growth"]["profile_module_percent"]) for pair in pairs]  # type: ignore[index]
        ),
        "raw_stages": {
            stage: {
                metric: series_statistics(
                    [
                        float(pair["first_to_last_growth"]["raw_stages"][stage][metric])  # type: ignore[index]
                        for pair in pairs
                    ]
                )
                for metric in (
                    "first_ms_per_token",
                    "last_ms_per_token",
                    "delta_ms_per_token",
                    "delta_percent",
                )
            }
            for stage in sorted(stage_names, key=lambda value: STAGE_SORT_KEY.get(value, len(STAGE_SORT_KEY)))
        },
    }
    return result


def stage_variance_passes(stats: dict[str, object], relative_limit: float, absolute_limit: float) -> bool:
    return (
        float(stats["coefficient_of_variation_percent"]) <= relative_limit
        or (
            float(stats["mean"]) <= SUB_MILLISECOND_STAGE_LIMIT_MS
            and float(stats["standard_deviation"]) <= absolute_limit
        )
    )


def qwen_stage_shape_passes(summary: dict[str, object]) -> bool:
    stages = summary["normalized_stages"]  # type: ignore[index]
    names = set(stages)
    required = {
        "embedding",
        "attention.qkv",
        "attention.rope",
        "attention.kv_append",
        "attention.core",
        "attention.output",
        "ffn.gate_up",
        "logits",
        "module.residual",
    }
    activation_shape = (
        {"ffn.activation", "ffn.down"}.issubset(names)
        or "ffn.swiglu_down_fused" in names
    )
    calls_stable = all(float(stage["calls_per_token"]["standard_deviation"]) == 0.0 for stage in stages.values())
    return required.issubset(names) and activation_shape and "helper.other" not in names and calls_stable


def frequency_median(pair: dict[str, object], mode: str) -> float | None:
    frequency = pair[mode]["process"]["frequency"]  # type: ignore[index]
    if not frequency.get("available"):  # type: ignore[union-attr]
        return None
    value = frequency.get("weighted_actual_mhz_median")  # type: ignore[union-attr]
    return float(value) if value is not None else None


def write_markdown(path: Path, document: dict[str, object]) -> None:
    summary = document["summary"]
    gate = document["gate"]
    configuration = document["configuration"]
    lines = [
        "# LiteNN Position-Binned Stage Control",
        "",
        f"- Host: `{document['host']['cpu_model']}`",  # type: ignore[index]
        f"- Threads: `{configuration['threads']}`",  # type: ignore[index]
        f"- Generated tokens: `{configuration['decode_tokens']['count']}`",  # type: ignore[index]
        f"- Repetitions: `{configuration['repetitions']}`",  # type: ignore[index]
        f"- Power-policy stability: `{gate['power_policy_stability']}`",  # type: ignore[index]
        f"- Accepted: `{gate['accepted']}`",  # type: ignore[index]
        "",
        "| Boundary | Clean ms/token | Profile ms/token | Overhead | Coverage |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Step | {summary['clean_step_ms_per_token']['median']:.3f} | "  # type: ignore[index]
        f"{summary['profile_step_ms_per_token']['median']:.3f} | "  # type: ignore[index]
        f"{summary['step_profile_overhead_percent']['median']:.2f}% | n/a |",  # type: ignore[index]
        f"| Module | {summary['clean_module_ms_per_token']['median']:.3f} | "  # type: ignore[index]
        f"{summary['profile_module_ms_per_token']['median']:.3f} | "  # type: ignore[index]
        f"{summary['module_profile_overhead_percent']['median']:.2f}% | "  # type: ignore[index]
        f"{summary['stage_coverage_percent']['median']:.2f}% |",  # type: ignore[index]
        "",
        "## Whole Window",
        "",
        "| Stage | Normalized ms/token | Raw ms/token | Calls/token | CV |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for stage, values in summary["normalized_stages"].items():  # type: ignore[union-attr]
        lines.append(
            f"| {stage} | {values['ms_per_token']['median']:.3f} | "
            f"{values['raw_ms_per_token']['median']:.3f} | {values['calls_per_token']['median']:.1f} | "
            f"{values['ms_per_token']['coefficient_of_variation_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Position Bins",
            "",
            "| Positions | Clean module ms | Profile module ms | Module overhead | Coverage |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for position_bin in summary["position_bins"]:  # type: ignore[union-attr]
        lines.append(
            f"| {position_bin['start']}-{position_bin['end']} | "
            f"{position_bin['clean_module_ms_per_token']['median']:.3f} | "
            f"{position_bin['profile_module_ms_per_token']['median']:.3f} | "
            f"{position_bin['module_profile_overhead_percent']['median']:.2f}% | "
            f"{position_bin['stage_coverage_percent']['median']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "| Positions | Stage | Raw median ms/token | Normalized median ms/token | CV |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for position_bin in summary["position_bins"]:  # type: ignore[union-attr]
        for stage, values in position_bin["normalized_stages"].items():
            lines.append(
                f"| {position_bin['start']}-{position_bin['end']} | {stage} | "
                f"{values['raw_ms_per_token']['median']:.3f} | {values['ms_per_token']['median']:.3f} | "
                f"{values['ms_per_token']['coefficient_of_variation_percent']:.2f}% |"
            )
    growth = summary["first_to_last_growth"]  # type: ignore[index]
    lines.extend(
        [
            "",
            "## First-To-Last Growth",
            "",
            f"- Clean module: `{growth['clean_module_ms']['median']:+.3f} ms/token` "
            f"(`{growth['clean_module_percent']['median']:+.2f}%`).",
            f"- Profile module: `{growth['profile_module_ms']['median']:+.3f} ms/token` "
            f"(`{growth['profile_module_percent']['median']:+.2f}%`).",
            "",
            "| Raw stage | First ms/token | Last ms/token | Delta ms/token | Delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for stage, values in growth["raw_stages"].items():
        lines.append(
            f"| {stage} | {values['first_ms_per_token']['median']:.3f} | "
            f"{values['last_ms_per_token']['median']:.3f} | "
            f"{values['delta_ms_per_token']['median']:+.3f} | {values['delta_percent']['median']:+.2f}% |"
        )
    lines.extend(["", "## Gates", ""])
    for name, value in gate.items():  # type: ignore[union-attr]
        lines.append(f"- `{name}`: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--litenn", required=True, type=Path)
    parser.add_argument("--aot-cache-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt-token-ids", type=comma_token_ids)
    prompt.add_argument("--prompt-token-ids-file", type=Path)
    decode = parser.add_mutually_exclusive_group(required=True)
    decode.add_argument("--decode-token-ids", type=comma_token_ids)
    decode.add_argument("--decode-token-ids-file", type=Path)
    parser.add_argument("--position-bins", required=True, type=position_bins)
    parser.add_argument("--repetitions", default=3, type=positive_int)
    parser.add_argument("--threads", default=8, type=positive_int)
    parser.add_argument("--affinity", choices=("default", "none", "compact", "spread"), default="default")
    parser.add_argument("--worker-wait", choices=("adaptive", "low-power", "latency"), default="adaptive")
    parser.add_argument("--llvm-opt-level", choices=(0, 1, 2, 3), default=0, type=int)
    parser.add_argument("--max-cache-length", type=positive_int)
    parser.add_argument(
        "--prepacked-weight-policy", choices=("disabled", "profitable", "all"), default="all"
    )
    parser.add_argument(
        "--prepacked-weight-layout",
        choices=("expanded-v1", "compact-v3", "field-interleaved-v4"),
        default="field-interleaved-v4",
    )
    parser.add_argument("--variance-threshold-percent", default=3.0, type=positive_float)
    parser.add_argument("--stage-variance-threshold-percent", default=15.0, type=positive_float)
    parser.add_argument(
        "--stage-absolute-standard-deviation-ms", default=0.05, type=non_negative_float
    )
    parser.add_argument("--overhead-threshold-percent", default=3.0, type=positive_float)
    parser.add_argument("--minimum-stage-coverage-percent", default=99.0, type=positive_float)
    parser.add_argument("--maximum-stage-coverage-percent", default=101.0, type=positive_float)
    parser.add_argument("--monitor-interval-seconds", default=0.25, type=positive_float)
    parser.add_argument("--python", default=sys.executable)
    return parser


def resolve_file(path: Path, label: str) -> Path:
    result = path.resolve()
    if not result.is_file():
        raise SystemExit(f"{label} not found: {path}")
    return result


def main() -> int:
    args = build_parser().parse_args()
    if args.repetitions < 3:
        raise SystemExit("position-stage control requires at least three repetitions")
    model = resolve_file(args.model, "model")
    litenn = resolve_file(args.litenn, "LiteNN GGUF tool")
    cache_dir = args.aot_cache_dir.resolve()
    if not cache_dir.is_dir():
        raise SystemExit(f"AOT cache directory not found: {args.aot_cache_dir}")
    args.aot_cache_dir = cache_dir
    prompt_ids = (
        args.prompt_token_ids
        if args.prompt_token_ids is not None
        else load_token_ids_file(args.prompt_token_ids_file.resolve())
    )
    decode_ids = (
        args.decode_token_ids
        if args.decode_token_ids is not None
        else load_token_ids_file(args.decode_token_ids_file.resolve())
    )
    if args.position_bins[-1][1] != len(decode_ids):
        raise SystemExit("--position-bins must cover every decode token")
    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md = args.output_md.resolve() if args.output_md is not None else None
    prompt_text = ",".join(str(value) for value in prompt_ids)
    decode_text = ",".join(str(value) for value in decode_ids)
    replacements = {
        str(model): "<model>",
        str(litenn): "<litenn-gguf>",
        str(cache_dir): "<aot-cache>",
        str(repo_root()): "<repo>",
        str(Path.cwd().resolve()): "<repo>",
        prompt_text: "<prompt-token-ids>",
        decode_text: "<decode-token-ids>",
    }
    document: dict[str, object] = {
        "schema_version": 1,
        "tool": "litenn-position-stage-control",
        "status": "running",
        "host": host_metadata(),
        "power_policy": power_policy(),
        "litenn_binary": binary_identity(litenn),
        "model": {"filename": "<model>", "size_bytes": model.stat().st_size},
        "configuration": {
            "repetitions": args.repetitions,
            "threads": args.threads,
            "affinity": args.affinity,
            "worker_wait": args.worker_wait,
            "llvm_opt_level": args.llvm_opt_level,
            "max_cache_length": args.max_cache_length,
            "prepacked_weight_policy": args.prepacked_weight_policy,
            "prepacked_weight_layout": args.prepacked_weight_layout,
            "prompt_tokens": {"count": len(prompt_ids), "sha256": token_ids_digest(prompt_ids)},
            "decode_tokens": {"count": len(decode_ids), "sha256": token_ids_digest(decode_ids)},
            "position_bins": [{"start": start, "end": end} for start, end in args.position_bins],
            "variance_threshold_percent": args.variance_threshold_percent,
            "stage_variance_threshold_percent": args.stage_variance_threshold_percent,
            "stage_absolute_standard_deviation_ms": args.stage_absolute_standard_deviation_ms,
            "overhead_threshold_percent": args.overhead_threshold_percent,
            "minimum_stage_coverage_percent": args.minimum_stage_coverage_percent,
            "maximum_stage_coverage_percent": args.maximum_stage_coverage_percent,
            "run_order": "odd=clean_then_profile,even=profile_then_clean",
        },
        "pairs": [],
    }

    def checkpoint() -> None:
        output_json.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    checkpoint()
    try:
        for repetition in range(1, args.repetitions + 1):
            order = ["clean", "profile"] if repetition % 2 == 1 else ["profile", "clean"]
            print(
                f"[LiteNN position stage] repetition={repetition}/{args.repetitions} "
                f"order={'->'.join(order)}",
                flush=True,
            )
            runs: dict[str, dict[str, object]] = {}
            for mode in order:
                print(f"[LiteNN position stage] repetition={repetition} mode={mode} starting", flush=True)
                prefix = output_json.parent / f"rep_{repetition:02d}_{mode}"
                runs[mode] = run_litenn(
                    args,
                    model,
                    litenn,
                    prompt_ids,
                    decode_ids,
                    mode == "profile",
                    prefix,
                    replacements,
                )
                print(f"[LiteNN position stage] repetition={repetition} mode={mode} finished", flush=True)
            pair = normalize_pair(runs["clean"], runs["profile"], args.position_bins)
            pair.update({"repetition": repetition, "order": order})
            document["pairs"].append(pair)  # type: ignore[union-attr]
            checkpoint()
    except Exception as error:
        document["status"] = "failed"
        document["error"] = redact_text(str(error), replacements)
        checkpoint()
        raise

    pairs = document["pairs"]
    summary = summarize_pairs(pairs)  # type: ignore[arg-type]
    whole_variance = all(
        float(summary[name]["coefficient_of_variation_percent"]) <= args.variance_threshold_percent
        for name in (
            "clean_step_ms_per_token",
            "profile_step_ms_per_token",
            "clean_module_ms_per_token",
            "profile_module_ms_per_token",
        )
    )
    whole_stage_variance = all(
        stage_variance_passes(values["ms_per_token"], args.stage_variance_threshold_percent,
                              args.stage_absolute_standard_deviation_ms)
        for values in summary["normalized_stages"].values()  # type: ignore[union-attr]
    )
    whole_overhead = all(
        abs(float(summary[name]["median"])) <= args.overhead_threshold_percent
        for name in ("step_profile_overhead_percent", "module_profile_overhead_percent")
    )
    whole_coverage = (
        args.minimum_stage_coverage_percent
        <= float(summary["stage_coverage_percent"]["median"])
        <= args.maximum_stage_coverage_percent
    )
    bin_variance = all(
        float(position_bin[name]["coefficient_of_variation_percent"]) <= args.variance_threshold_percent
        for position_bin in summary["position_bins"]  # type: ignore[union-attr]
        for name in (
            "clean_step_ms_per_token",
            "profile_step_ms_per_token",
            "clean_module_ms_per_token",
            "profile_module_ms_per_token",
        )
    )
    bin_stage_variance = all(
        stage_variance_passes(values["ms_per_token"], args.stage_variance_threshold_percent,
                              args.stage_absolute_standard_deviation_ms)
        for position_bin in summary["position_bins"]  # type: ignore[union-attr]
        for values in position_bin["normalized_stages"].values()
    )
    bin_overhead = all(
        abs(float(position_bin[name]["median"])) <= args.overhead_threshold_percent
        for position_bin in summary["position_bins"]  # type: ignore[union-attr]
        for name in ("step_profile_overhead_percent", "module_profile_overhead_percent")
    )
    bin_coverage = all(
        args.minimum_stage_coverage_percent
        <= float(position_bin["stage_coverage_percent"]["median"])
        <= args.maximum_stage_coverage_percent
        for position_bin in summary["position_bins"]  # type: ignore[union-attr]
    )
    shape = qwen_stage_shape_passes(summary)
    trajectory = all(
        pair[mode]["tokens"] == token_ids_identity(decode_ids)  # type: ignore[index]
        and bool(pair[mode]["forced_replay"]["enabled"])  # type: ignore[index]
        for pair in pairs  # type: ignore[union-attr]
        for mode in ("clean", "profile")
    )
    no_fallback = all(
        int(pair[mode]["fallback_count"]) == 0  # type: ignore[index]
        for pair in pairs  # type: ignore[union-attr]
        for mode in ("clean", "profile")
    )
    power_stable = all(
        process_power_policy_stable(pair[mode]["process"])  # type: ignore[index]
        for pair in pairs  # type: ignore[union-attr]
        for mode in ("clean", "profile")
    )
    gate = {
        "whole_variance": whole_variance,
        "whole_stage_variance": whole_stage_variance,
        "whole_profile_overhead": whole_overhead,
        "whole_stage_coverage": whole_coverage,
        "position_bin_variance": bin_variance,
        "position_bin_stage_variance": bin_stage_variance,
        "position_bin_profile_overhead": bin_overhead,
        "position_bin_stage_coverage": bin_coverage,
        "qwen_stage_shape": shape,
        "fixed_trajectory": trajectory,
        "no_fallback": no_fallback,
        "aot_cache_hit_required": True,
        "power_policy_stability": power_stable,
    }
    gate["accepted"] = all(gate.values())
    document["summary"] = summary
    document["gate"] = gate
    document["status"] = "complete"
    document["power_policy_after"] = power_policy()
    frequency: dict[str, object] = {}
    for mode in ("clean", "profile"):
        values = []
        for pair in pairs:  # type: ignore[union-attr]
            value = frequency_median(pair, mode)
            if value is not None:
                values.append(value)
        frequency[mode] = series_statistics(values) if values else None
    document["frequency"] = frequency
    checkpoint()
    if output_md is not None:
        write_markdown(output_md, document)
    return 0 if gate["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
