#!/usr/bin/env python3
"""Run an alternating, redacted llama.cpp actual-completion configuration sweep."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

from run_llama_cpp_completion_control import (
    build_command,
    finite_float,
    host_metadata,
    is_absolute_path,
    parse_metadata,
    parse_perf_output,
    poll_level,
    positive_int,
    priority,
    prompt_metadata,
    resolve_completion,
    version_metadata,
)
from run_paired_gguf_decode_control import (
    positive_float,
    power_policy,
    redact_command,
    run_monitored,
    series_statistics,
    text_identity,
)


PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
PROFILE_FIELDS = {
    "threads",
    "threads_batch",
    "cpu_mask",
    "cpu_strict",
    "poll",
    "priority",
    "mmap",
    "repack",
    "warmup",
    "cache_type_k",
    "cache_type_v",
}


def parse_choice(value: str, choices: set[str], field: str) -> str:
    if value not in choices:
        raise argparse.ArgumentTypeError(f"{field} must be one of {sorted(choices)}")
    return value


def parse_profile(raw: str, defaults: argparse.Namespace) -> dict[str, object]:
    parts = [part.strip() for part in raw.split(";")]
    name = parts[0]
    if not name or not PROFILE_NAME_RE.fullmatch(name):
        raise argparse.ArgumentTypeError(
            "profile name must contain only ASCII letters, digits, '.', '_' or '-'"
        )
    values: dict[str, object] = {
        "name": name,
        "threads": defaults.default_threads,
        "threads_batch": defaults.default_threads_batch,
        "cpu_mask": defaults.default_cpu_mask,
        "cpu_strict": defaults.default_cpu_strict,
        "poll": defaults.default_poll,
        "priority": defaults.default_priority,
        "mmap": defaults.default_mmap,
        "repack": defaults.default_repack,
        "warmup": defaults.default_warmup,
        "cache_type_k": defaults.default_cache_type_k,
        "cache_type_v": defaults.default_cache_type_v,
    }
    seen = set()
    for assignment in parts[1:]:
        key, separator, value = assignment.partition("=")
        key = key.strip()
        value = value.strip()
        if not separator or key not in PROFILE_FIELDS:
            raise argparse.ArgumentTypeError(f"invalid profile assignment {assignment!r}")
        if key in seen:
            raise argparse.ArgumentTypeError(f"duplicate profile field {key!r}")
        seen.add(key)
        try:
            if key in ("threads", "threads_batch"):
                values[key] = positive_int(value)
            elif key == "poll":
                values[key] = poll_level(value)
            elif key == "priority":
                values[key] = priority(value)
            elif key == "cpu_strict":
                values[key] = parse_choice(value, {"0", "1"}, key)
            elif key in ("mmap", "repack", "warmup"):
                values[key] = parse_choice(value, {"on", "off"}, key)
            else:
                values[key] = value
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(f"invalid {key} value {value!r}") from exc
    if values["threads_batch"] is None:
        values["threads_batch"] = values["threads"]
    return values


def profile_namespace(args: argparse.Namespace, profile: dict[str, object]) -> argparse.Namespace:
    result = copy.copy(args)
    for key in PROFILE_FIELDS:
        setattr(result, key, profile[key])
    return result


def summarize_profile(profile: dict[str, object], runs: list[dict[str, object]]) -> dict[str, object]:
    profile_runs = [run for run in runs if run["profile"] == profile["name"]]
    throughputs = [float(run["metrics"]["eval_tokens_per_second"]) for run in profile_runs]  # type: ignore[index]
    latencies = [float(run["metrics"]["eval_ms_per_token"]) for run in profile_runs]  # type: ignore[index]
    prompt_counts = {int(run["metrics"]["prompt_count"]) for run in profile_runs}  # type: ignore[index]
    eval_counts = {int(run["metrics"]["eval_count"]) for run in profile_runs}  # type: ignore[index]
    text_hashes = {str(run["text"]["sha256"]) for run in profile_runs}  # type: ignore[index]
    weighted_frequencies = [
        float(value)
        for run in profile_runs
        if (
            value := run["process"]["frequency"].get("weighted_actual_mhz_median")  # type: ignore[index]
        )
        is not None
    ]
    if len(prompt_counts) != 1 or len(eval_counts) != 1 or len(text_hashes) != 1:
        raise ValueError(f"profile {profile['name']} produced inconsistent output windows")
    return {
        "name": profile["name"],
        "configuration": profile,
        "prompt_tokens": next(iter(prompt_counts)),
        "eval_tokens": next(iter(eval_counts)),
        "text_sha256": next(iter(text_hashes)),
        "tokens_per_second": series_statistics(throughputs),
        "ms_per_token": series_statistics(latencies),
        "weighted_actual_mhz": series_statistics(weighted_frequencies) if weighted_frequencies else None,
    }


def write_markdown(path: Path, document: dict[str, object]) -> None:
    binary = document["binary"]
    host = document["host"]
    gate = document["gate"]
    lines = [
        "# llama.cpp Actual-Completion Configuration Sweep",
        "",
        f"- Host: `{host['cpu_model']}` ({host['logical_cpus']} logical CPUs)",  # type: ignore[index]
        f"- Binary SHA-256: `{binary['sha256']}`",  # type: ignore[index]
        f"- Prompt SHA-256: `{document['prompt']['sha256']}`",  # type: ignore[index]
        f"- Repetitions: `{document['configuration']['repetitions']}`",  # type: ignore[index]
        f"- Variance threshold: `{document['configuration']['variance_threshold_percent']}%`",  # type: ignore[index]
        f"- Power policy: `{document['power_policy']['value']}`",  # type: ignore[index]
        "",
        "| Rank | Profile | Threads | Mask/strict | Poll | Priority | MHz | Median ms/token | Median t/s | CV |",
        "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, summary in enumerate(document["ranking"], 1):  # type: ignore[union-attr]
        config = summary["configuration"]
        throughput = summary["tokens_per_second"]
        latency = summary["ms_per_token"]
        frequency = summary["weighted_actual_mhz"]
        mask = config["cpu_mask"] or "default"
        frequency_text = f"{frequency['median']:.0f}" if frequency is not None else "n/a"
        lines.append(
            f"| {rank} | {summary['name']} | {config['threads']} | {mask}/{config['cpu_strict']} | "
            f"{config['poll']} | {config['priority']} | {frequency_text} | {latency['median']:.3f} | "
            f"{throughput['median']:.3f} | {throughput['coefficient_of_variation_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Gates",
            "",
            f"- Token-window parity: `{gate['token_window_parity']}`",  # type: ignore[index]
            f"- Text parity: `{gate['text_parity']}`",  # type: ignore[index]
            f"- Variance: `{gate['variance']}`",  # type: ignore[index]
            f"- Accepted: `{gate['accepted']}`",  # type: ignore[index]
            "",
            "## Commands",
            "",
        ]
    )
    for name, command in document["commands"].items():  # type: ignore[union-attr]
        lines.extend((f"`{name}`:", "", "```text", " ".join(command), "```", ""))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--llama-completion", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--profile",
        action="append",
        required=True,
        help="NAME;threads=N;cpu_mask=HEX;cpu_strict=0|1;poll=0..100;priority=-1..3;...",
    )
    parser.add_argument("--repetitions", default=3, type=positive_int)
    parser.add_argument("--variance-threshold-percent", default=3.0, type=positive_float)
    parser.add_argument("--require-variance-gate", action="store_true")
    parser.add_argument("--monitor-interval", default=0.25, type=positive_float)
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--keep-prompt", action="store_true")
    parser.add_argument("--predict", default=32, type=positive_int)
    parser.add_argument("--context-size", default=256, type=positive_int)
    parser.add_argument("--conversation-mode", choices=["raw", "chat"], default="chat")
    parser.add_argument("--batch-size", default=2048, type=positive_int)
    parser.add_argument("--ubatch-size", default=512, type=positive_int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--temperature", default=0.0, type=finite_float)
    parser.add_argument("--respect-eos", dest="ignore_eos", action="store_false")
    parser.set_defaults(ignore_eos=True)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--build-metadata", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--default-threads", default=2, type=positive_int)
    parser.add_argument("--default-threads-batch", type=positive_int)
    parser.add_argument("--default-cpu-mask", default="")
    parser.add_argument("--default-cpu-strict", choices=["0", "1"], default="0")
    parser.add_argument("--default-poll", default=50, type=poll_level)
    parser.add_argument("--default-priority", default=0, type=priority)
    parser.add_argument("--default-mmap", choices=["on", "off"], default="on")
    parser.add_argument("--default-repack", choices=["on", "off"], default="on")
    parser.add_argument("--default-warmup", choices=["on", "off"], default="on")
    parser.add_argument("--default-cache-type-k", default="f16")
    parser.add_argument("--default-cache-type-v", default="f16")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.repetitions < 3:
        raise SystemExit("configuration sweep requires at least three repetitions")
    try:
        profiles = [parse_profile(raw, args) for raw in args.profile]
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc
    profile_names = [str(profile["name"]) for profile in profiles]
    if len(set(profile_names)) != len(profile_names):
        raise SystemExit("profile names must be unique")

    completion = resolve_completion(args.llama_completion)
    model = args.model.resolve()
    if not model.is_file():
        raise SystemExit(f"model file not found: {args.model}")
    output_json = args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)

    replacements = {
        str(model): "<model>",
        str(completion): "<llama-completion>",
        str(Path.cwd()): "<repo>",
        args.prompt: "<prompt>",
    }
    replacements.update({argument: "<path>" for argument in args.extra_arg if is_absolute_path(argument)})
    commands: dict[str, list[str]] = {}
    raw_commands: dict[str, list[str]] = {}
    for profile in profiles:
        namespace = profile_namespace(args, profile)
        command = build_command(namespace, completion, model, int(profile["threads"]))
        raw_commands[str(profile["name"])] = command
        commands[str(profile["name"])] = redact_command(command, replacements)

    binary = version_metadata(completion)
    binary["cmake"].update(parse_metadata(args.build_metadata))  # type: ignore[union-attr]
    document: dict[str, object] = {
        "schema_version": 1,
        "tool": "llama-completion-configuration-sweep",
        "status": "running",
        "host": host_metadata(),
        "power_policy": power_policy(),
        "binary": binary,
        "model": {"filename": "<model>", "size_bytes": model.stat().st_size},
        "prompt": prompt_metadata(args.prompt, args.keep_prompt),
        "configuration": {
            "repetitions": args.repetitions,
            "run_order": "odd=forward,even=reverse",
            "predict": args.predict,
            "context_size": args.context_size,
            "conversation_mode": args.conversation_mode,
            "batch_size": args.batch_size,
            "ubatch_size": args.ubatch_size,
            "seed": args.seed,
            "temperature": args.temperature,
            "ignore_eos": args.ignore_eos,
            "variance_threshold_percent": args.variance_threshold_percent,
            "monitor_interval_seconds": args.monitor_interval,
            "gpu_layers": 0,
            "device": "none",
            "flash_attention": "off",
        },
        "profiles": profiles,
        "commands": commands,
        "runs": [],
    }

    def checkpoint() -> None:
        output_json.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    checkpoint()
    runs = document["runs"]
    assert isinstance(runs, list)
    for repetition in range(1, args.repetitions + 1):
        ordered_profiles = profiles if repetition % 2 == 1 else list(reversed(profiles))
        for profile in ordered_profiles:
            name = str(profile["name"])
            print(
                f"[llama.cpp sweep] {name} repetition {repetition}/{args.repetitions} starting",
                file=sys.stderr,
                flush=True,
            )
            process_record, stdout_text, stderr_text = run_monitored(
                raw_commands[name],
                output_json.with_suffix("").parent / f"{output_json.stem}.{name}.r{repetition}",
                replacements,
                args.monitor_interval,
            )
            if process_record["returncode"] != 0:
                document["status"] = "failed"
                document["failure"] = f"profile {name} repetition {repetition} failed"
                checkpoint()
                raise SystemExit(document["failure"])
            try:
                metrics = parse_perf_output(f"{stdout_text}\n{stderr_text}")
            except ValueError as exc:
                document["status"] = "failed"
                document["failure"] = f"profile {name} repetition {repetition}: {exc}"
                checkpoint()
                raise SystemExit(document["failure"]) from exc
            combined = f"{stdout_text}\n{stderr_text}"
            system_info = next((line.strip() for line in combined.splitlines() if "system_info:" in line), "")
            run = {
                "profile": name,
                "repetition": repetition,
                "process": process_record,
                "metrics": metrics,
                "text": text_identity(stdout_text.strip()),
                "system_info": system_info,
            }
            runs.append(run)
            checkpoint()
            print(
                f"[llama.cpp sweep] {name} repetition {repetition}/{args.repetitions} finished: "
                f"{metrics['eval_ms_per_token']:.3f} ms/token, "
                f"{metrics['eval_tokens_per_second']:.3f} t/s",
                file=sys.stderr,
                flush=True,
            )

    summaries = [summarize_profile(profile, runs) for profile in profiles]
    ranking = sorted(
        summaries,
        key=lambda summary: float(summary["tokens_per_second"]["median"]),  # type: ignore[index]
        reverse=True,
    )
    prompt_windows = {int(summary["prompt_tokens"]) for summary in summaries}
    eval_windows = {int(summary["eval_tokens"]) for summary in summaries}
    text_hashes = {str(summary["text_sha256"]) for summary in summaries}
    variance_passed = all(
        float(summary["tokens_per_second"]["coefficient_of_variation_percent"])  # type: ignore[index]
        <= args.variance_threshold_percent
        for summary in summaries
    )
    gate = {
        "token_window_parity": len(prompt_windows) == 1 and len(eval_windows) == 1,
        "text_parity": len(text_hashes) == 1,
        "variance": variance_passed,
    }
    gate["accepted"] = all(gate.values())
    document["summary"] = summaries
    document["ranking"] = ranking
    document["gate"] = gate
    document["status"] = "complete"
    document["power_policy_after"] = power_policy()
    checkpoint()
    if args.output_md is not None:
        write_markdown(args.output_md, document)
    best = ranking[0]
    print(
        f"[llama.cpp sweep] best={best['name']} "
        f"{best['tokens_per_second']['median']:.3f} t/s "  # type: ignore[index]
        f"variance_gate={variance_passed}",
        file=sys.stderr,
    )
    return 2 if args.require_variance_gate and not variance_passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
