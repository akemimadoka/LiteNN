#!/usr/bin/env python3
"""Run equal-thread LiteNN/llama.cpp GGUF decode scaling controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path


def positive_thread_counts(raw: str) -> list[int]:
    try:
        values = [int(value) for value in raw.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("thread counts must be comma-separated integers") from error
    if not values or any(value <= 0 for value in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("thread counts must be unique positive integers")
    return sorted(values)


def _measured_windows(document: dict[str, object], runtime: str) -> list[dict[str, object]]:
    return [
        window
        for pair in document["pairs"]  # type: ignore[index]
        for window in pair[runtime]["in_process"]["windows"]  # type: ignore[index]
        if window["phase"] == "measured"
    ]


def summarize_runtime(document: dict[str, object], runtime: str, threads: int) -> dict[str, object]:
    windows = _measured_windows(document, runtime)
    wall_ms_per_token = [float(window["decodeWallMs"]) / int(window["decodeTokens"]) for window in windows]
    cpu_ms_per_token = [
        float(window["telemetry"]["processCPUTimeDeltaMs"]) / int(window["decodeTokens"])  # type: ignore[index]
        for window in windows
        if window["telemetry"]["processCPUTimeDeltaMs"] is not None  # type: ignore[index]
    ]
    cpu_complete = len(cpu_ms_per_token) == len(windows) and bool(windows)
    throughput = float(document["summary"][runtime]["median"])  # type: ignore[index]
    cpu_median = statistics.median(cpu_ms_per_token) if cpu_complete else None
    return {
        "runtime": runtime,
        "threads": threads,
        "process_median_tokens_per_second": throughput,
        "process_cv_percent": float(
            document["summary"][runtime]["coefficient_of_variation_percent"]  # type: ignore[index]
        ),
        "measured_window_count": len(windows),
        "wall_ms_per_token_median": statistics.median(wall_ms_per_token) if wall_ms_per_token else None,
        "process_cpu_ms_per_token_median": cpu_median,
        "tokens_per_cpu_second": 1000.0 / cpu_median if cpu_median is not None and cpu_median > 0.0 else None,
        "cpu_telemetry_complete": cpu_complete,
    }


def summarize_scaling_reports(reports: list[tuple[int, dict[str, object]]]) -> dict[str, object]:
    if not reports:
        raise ValueError("at least one scaling report is required")
    rows: list[dict[str, object]] = []
    for threads, report in reports:
        configuration = report["configuration"]
        if int(configuration["litenn_threads"]) != threads or int(configuration["llama_threads"]) != threads:  # type: ignore[index]
            raise ValueError(f"thread configuration mismatch for T{threads}")
        for runtime in ("llama_cpp", "litenn"):
            rows.append(summarize_runtime(report, runtime, threads))

    baseline_threads = min(threads for threads, _ in reports)
    for runtime in ("llama_cpp", "litenn"):
        baseline = next(
            float(row["process_median_tokens_per_second"])
            for row in rows
            if row["runtime"] == runtime and row["threads"] == baseline_threads
        )
        for row in rows:
            if row["runtime"] != runtime:
                continue
            speedup = float(row["process_median_tokens_per_second"]) / baseline
            row["speedup_vs_baseline"] = speedup
            row["parallel_efficiency_percent"] = (
                speedup * baseline_threads * 100.0 / int(row["threads"])
            )

    paired_rows = []
    for threads, report in reports:
        llama = next(row for row in rows if row["runtime"] == "llama_cpp" and row["threads"] == threads)
        litenn = next(row for row in rows if row["runtime"] == "litenn" and row["threads"] == threads)
        paired_rows.append(
            {
                "threads": threads,
                "litenn_vs_llama_wall_throughput_percent": (
                    float(litenn["process_median_tokens_per_second"])
                    / float(llama["process_median_tokens_per_second"])
                    - 1.0
                )
                * 100.0,
                "litenn_vs_llama_process_cpu_time_ratio": (
                    float(litenn["process_cpu_ms_per_token_median"])
                    / float(llama["process_cpu_ms_per_token_median"])
                    if litenn["process_cpu_ms_per_token_median"] is not None
                    and llama["process_cpu_ms_per_token_median"] is not None
                    else None
                ),
                "child_accepted": bool(report["gate"]["accepted"]),  # type: ignore[index]
            }
        )
    return {"baseline_threads": baseline_threads, "runtime_rows": rows, "paired_rows": paired_rows}


def write_markdown(path: Path, document: dict[str, object]) -> None:
    def metric(value: object) -> str:
        return f"{float(value):.3f}" if isinstance(value, (int, float)) else "n/a"

    lines = [
        "# Equal-Thread GGUF Decode Scaling Control",
        "",
        f"- Shared process CPU set: `{document['configuration']['process_cpu_set']}`",  # type: ignore[index]
        f"- Thread counts: `{document['configuration']['thread_counts']}`",  # type: ignore[index]
        "",
        "| Runtime | Threads | t/s | Process CV | Wall ms/token | CPU ms/token | "
        "tokens/CPU-s | Speedup | Parallel efficiency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in document["summary"]["runtime_rows"]:  # type: ignore[index]
        lines.append(
            f"| {row['runtime']} | {row['threads']} | {row['process_median_tokens_per_second']:.3f} | "
            f"{row['process_cv_percent']:.2f}% | {metric(row['wall_ms_per_token_median'])} | "
            f"{metric(row['process_cpu_ms_per_token_median'])} | {metric(row['tokens_per_cpu_second'])} | "
            f"{row['speedup_vs_baseline']:.3f}x | {row['parallel_efficiency_percent']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "| Threads | LiteNN wall delta | LiteNN/reference CPU-time ratio | Child accepted |",
            "| ---: | ---: | ---: | --- |",
        ]
    )
    for row in document["summary"]["paired_rows"]:  # type: ignore[index]
        lines.append(
            f"| {row['threads']} | {row['litenn_vs_llama_wall_throughput_percent']:+.2f}% | "
            f"{metric(row['litenn_vs_llama_process_cpu_time_ratio'])}x | {row['child_accepted']} |"
        )
    lines.extend(
        [
            "",
            "## Gates",
            "",
            f"- Exact equal-thread configuration: `{document['gate']['equal_thread_configuration']}`",  # type: ignore[index]
            f"- Shared process CPU set: `{document['gate']['shared_process_cpu_set']}`",  # type: ignore[index]
            f"- Complete process CPU telemetry: `{document['gate']['cpu_telemetry']}`",  # type: ignore[index]
            f"- All paired child controls accepted: `{document['gate']['child_controls']}`",  # type: ignore[index]
            f"- Accepted: `{document['gate']['accepted']}`",  # type: ignore[index]
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def option_present(arguments: list[str], name: str) -> bool:
    return any(argument == name or argument.startswith(name + "=") for argument in arguments)


def option_value(arguments: list[str], name: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument.startswith(name + "="):
            return argument[len(name) + 1 :]
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
    return None


def resume_input_facts(paired_arguments: list[str]) -> dict[str, dict[str, int]]:
    facts = {}
    for name in ("--model", "--litenn", "--llamacpp-tokenizer-tool", "--llama-completion"):
        value = option_value(paired_arguments, name)
        if value is None:
            continue
        path = Path(value).resolve()
        try:
            stat = path.stat()
        except OSError as error:
            raise SystemExit(f"cannot fingerprint resume input {name}: {error}") from error
        facts[name] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return facts


def resume_identity(
    thread_counts: list[int],
    python: str,
    paired_arguments: list[str],
    input_facts: dict[str, dict[str, int]] | None = None,
) -> str:
    payload = json.dumps(
        {
            "thread_counts": thread_counts,
            "python": python,
            "paired_arguments": paired_arguments,
            "input_facts": input_facts or {},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_completed_child_report(path: Path, threads: int) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        configuration = document["configuration"]
        gate = document["gate"]
        if (
            document.get("status") != "complete"
            or int(configuration["litenn_threads"]) != threads
            or int(configuration["llama_threads"]) != threads
            or not isinstance(document.get("pairs"), list)
            or not document["pairs"]
            or not isinstance(gate.get("accepted"), bool)
        ):
            return None
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return None
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-counts", default=[1, 2, 4, 8], type=positive_thread_counts)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--resume-complete",
        action="store_true",
        help="Reuse only complete thread controls from an identical scaling invocation",
    )
    parser.add_argument(
        "paired_arguments",
        nargs=argparse.REMAINDER,
        help="Arguments for run_paired_gguf_decode_control.py after --",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paired_arguments = args.paired_arguments
    if paired_arguments and paired_arguments[0] == "--":
        paired_arguments = paired_arguments[1:]
    required = ("--model", "--litenn", "--llamacpp-tokenizer-tool", "--llama-completion", "--aot-cache-dir")
    missing = [name for name in required if not option_present(paired_arguments, name)]
    if missing:
        raise SystemExit("missing paired runner arguments: " + ", ".join(missing))
    for forbidden in ("--output-dir", "--litenn-threads", "--llama-threads", "--require-variance-gate"):
        if option_present(paired_arguments, forbidden):
            raise SystemExit(f"{forbidden} is owned by the scaling controller")
    for required_control in ("--fixed-token-replay", "--in-process-windows", "--process-cpu-set"):
        if not option_present(paired_arguments, required_control):
            raise SystemExit(f"equal-thread scaling requires {required_control}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "gguf_decode_scaling_control.json"
    markdown_path = output_dir / "gguf_decode_scaling_control.md"
    invocation_identity = resume_identity(
        args.thread_counts,
        args.python,
        paired_arguments,
        resume_input_facts(paired_arguments),
    )
    resume_validated = False
    if args.resume_complete and report_path.is_file():
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            previous_configuration = previous["configuration"]
        except (KeyError, TypeError, json.JSONDecodeError, OSError) as error:
            raise SystemExit(f"cannot resume invalid scaling report: {error}") from error
        if previous_configuration.get("resume_identity") != invocation_identity:
            raise SystemExit("cannot resume scaling control: invocation identity differs")
        resume_validated = True
    document: dict[str, object] = {
        "schema": "litenn.gguf_decode_scaling_control.v1",
        "status": "running",
        "configuration": {
            "thread_counts": args.thread_counts,
            "resume_identity": invocation_identity,
        },
        "runs": [],
    }

    def checkpoint() -> None:
        report_path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    checkpoint()
    reports: list[tuple[int, dict[str, object]]] = []
    runner = Path(__file__).with_name("run_paired_gguf_decode_control.py")
    for threads in args.thread_counts:
        run_dir = output_dir / f"threads_{threads:02d}"
        child_report_path = run_dir / "paired_gguf_decode_control.json"
        child = load_completed_child_report(child_report_path, threads) if resume_validated else None
        resumed = child is not None
        command = [
            args.python,
            str(runner),
            *paired_arguments,
            "--litenn-threads",
            str(threads),
            "--llama-threads",
            str(threads),
            "--output-dir",
            str(run_dir),
        ]
        if child is None:
            print(f"[decode scaling] T{threads} starting", file=sys.stderr, flush=True)
            result = subprocess.run(command, check=False)
            if result.returncode != 0 or not child_report_path.is_file():
                document["status"] = "failed"
                document["failure"] = f"T{threads} paired control failed with return code {result.returncode}"
                checkpoint()
                return result.returncode or 1
            child = load_completed_child_report(child_report_path, threads)
            if child is None:
                document["status"] = "failed"
                document["failure"] = f"T{threads} paired control did not publish a complete report"
                checkpoint()
                return 1
        else:
            print(f"[decode scaling] T{threads} reusing complete report", file=sys.stderr, flush=True)
        reports.append((threads, child))
        document["runs"].append(  # type: ignore[union-attr]
            {
                "threads": threads,
                "paired_report": child_report_path.relative_to(output_dir).as_posix(),
                "accepted": bool(child["gate"]["accepted"]),
                "resumed": resumed,
            }
        )
        checkpoint()
        print(f"[decode scaling] T{threads} finished", file=sys.stderr, flush=True)

    summary = summarize_scaling_reports(reports)
    process_cpu_sets = [report["configuration"]["process_cpu_set"] for _, report in reports]  # type: ignore[index]
    equal_thread_configuration = all(
        int(report["configuration"]["litenn_threads"]) == threads  # type: ignore[index]
        and int(report["configuration"]["llama_threads"]) == threads  # type: ignore[index]
        for threads, report in reports
    )
    shared_process_cpu_set = bool(process_cpu_sets[0]) and all(value == process_cpu_sets[0] for value in process_cpu_sets)
    cpu_telemetry = all(bool(row["cpu_telemetry_complete"]) for row in summary["runtime_rows"])
    child_controls = all(bool(report["gate"]["accepted"]) for _, report in reports)  # type: ignore[index]
    document["configuration"]["process_cpu_set"] = process_cpu_sets[0]  # type: ignore[index]
    document["summary"] = summary
    document["gate"] = {
        "equal_thread_configuration": equal_thread_configuration,
        "shared_process_cpu_set": shared_process_cpu_set,
        "cpu_telemetry": cpu_telemetry,
        "child_controls": child_controls,
        "accepted": equal_thread_configuration and shared_process_cpu_set and cpu_telemetry and child_controls,
    }
    document["status"] = "complete"
    checkpoint()
    write_markdown(markdown_path, document)
    return 0 if document["gate"]["accepted"] else 2  # type: ignore[index]


if __name__ == "__main__":
    raise SystemExit(main())
