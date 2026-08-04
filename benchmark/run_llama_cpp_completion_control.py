#!/usr/bin/env python3
"""Run a redacted CPU-only llama.cpp actual-completion control matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath, PureWindowsPath


NUMBER = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|inf|nan)"
PERF_PATTERNS = {
    "sampling_ms": re.compile(rf"sampling time\s*=\s*(?P<value>{NUMBER})\s*ms", re.IGNORECASE),
    "load_ms": re.compile(rf"load time\s*=\s*(?P<value>{NUMBER})\s*ms", re.IGNORECASE),
    "prompt_eval": re.compile(
        rf"prompt eval time\s*=\s*(?P<total>{NUMBER})\s*ms\s*/\s*(?P<count>\d+)\s*tokens\s*"
        rf"\(\s*(?P<per_token>{NUMBER})\s*ms per token,\s*(?P<tps>{NUMBER})\s*tokens per second\)",
        re.IGNORECASE,
    ),
    "eval": re.compile(
        rf"(?<!prompt )eval time\s*=\s*(?P<total>{NUMBER})\s*ms\s*/\s*(?P<count>\d+)\s*runs\s*"
        rf"\(\s*(?P<per_token>{NUMBER})\s*ms per token,\s*(?P<tps>{NUMBER})\s*tokens per second\)",
        re.IGNORECASE,
    ),
    "total": re.compile(
        rf"total time\s*=\s*(?P<total>{NUMBER})\s*ms\s*/\s*(?P<count>\d+)\s*tokens", re.IGNORECASE
    ),
    "unaccounted": re.compile(
        rf"unaccounted time\s*=\s*(?P<total>{NUMBER})\s*ms\s*/\s*(?P<percent>{NUMBER})\s*%", re.IGNORECASE
    ),
    "graphs_reused": re.compile(r"graphs reused\s*=\s*(?P<count>\d+)", re.IGNORECASE),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def executable_name(name: str) -> str:
    return f"{name}.exe" if sys.platform == "win32" else name


def default_completion_candidates() -> list[Path]:
    root = repo_root()
    binary = executable_name("llama-completion")
    return [
        root / "third_party" / "llama.cpp" / "build" / "bin" / binary,
        root / "third_party" / "llama.cpp" / "build" / "bin" / "Release" / binary,
        root / "third_party" / "llama.cpp" / "build" / "bin" / "RelWithDebInfo" / binary,
    ]


def resolve_completion(path: Path | None) -> Path:
    if path is not None:
        resolved = path.resolve()
        if not resolved.is_file():
            raise SystemExit(f"llama-completion executable not found: {path}")
        return resolved
    for candidate in default_completion_candidates():
        if candidate.is_file():
            return candidate.resolve()
    raise SystemExit("llama-completion executable not found; pass --llama-completion explicitly")


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def poll_level(raw: str) -> int:
    value = int(raw)
    if value < 0 or value > 100:
        raise argparse.ArgumentTypeError("poll level must be in [0, 100]")
    return value


def priority(raw: str) -> int:
    value = int(raw)
    if value < -1 or value > 3:
        raise argparse.ArgumentTypeError("priority must be in [-1, 3]")
    return value


def finite_float(raw: str) -> float:
    value = float(raw)
    if value != value or value in (float("inf"), float("-inf")):
        raise argparse.ArgumentTypeError("value must be finite")
    return value


def parse_number(raw: str) -> float | None:
    value = float(raw)
    return value if math.isfinite(value) else None


def parse_perf_output(output: str) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for name in ("sampling_ms", "load_ms"):
        match = PERF_PATTERNS[name].search(output)
        if match:
            metrics[name] = parse_number(match.group("value"))

    for name, prefix in (("prompt_eval", "prompt"), ("eval", "eval")):
        match = PERF_PATTERNS[name].search(output)
        if match:
            metrics[f"{prefix}_total_ms"] = parse_number(match.group("total"))
            metrics[f"{prefix}_count"] = int(match.group("count"))
            metrics[f"{prefix}_ms_per_token"] = parse_number(match.group("per_token"))
            metrics[f"{prefix}_tokens_per_second"] = parse_number(match.group("tps"))

    match = PERF_PATTERNS["total"].search(output)
    if match:
        metrics["total_ms"] = parse_number(match.group("total"))
        metrics["total_tokens"] = int(match.group("count"))

    match = PERF_PATTERNS["unaccounted"].search(output)
    if match:
        metrics["unaccounted_ms"] = parse_number(match.group("total"))
        metrics["unaccounted_percent"] = parse_number(match.group("percent"))

    match = PERF_PATTERNS["graphs_reused"].search(output)
    if match:
        metrics["graphs_reused"] = int(match.group("count"))

    if "eval_tokens_per_second" not in metrics:
        raise ValueError("llama-completion output does not contain an eval timing record")
    return metrics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_metadata(prompt: str, keep_prompt: bool) -> dict[str, object]:
    encoded = prompt.encode("utf-8")
    return {
        "text": prompt if keep_prompt else "<prompt>",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "utf8_bytes": len(encoded),
    }


def redact_text(text: str, replacements: dict[str, str]) -> str:
    redacted = text
    candidates: dict[str, str] = {}
    for source, destination in replacements.items():
        if not source:
            continue
        candidates[source] = destination
        try:
            candidates[Path(source).as_posix()] = destination
            candidates[str(Path(source).resolve())] = destination
            candidates[Path(source).resolve().as_posix()] = destination
        except OSError:
            pass
    for source, destination in list(candidates.items()):
        candidates[json.dumps(source)[1:-1]] = destination
    for source in sorted(candidates, key=len, reverse=True):
        redacted = redacted.replace(source, candidates[source])
    return redacted


def is_absolute_path(value: str) -> bool:
    return Path(value).is_absolute() or PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def redact_command(command: list[str], completion: Path, model: Path, prompt: str) -> list[str]:
    replacements = {
        str(completion): "<llama-completion>",
        str(model): "<model>",
        prompt: "<prompt>",
    }
    redacted = [redact_text(argument, replacements) for argument in command]
    return ["<path>" if is_absolute_path(argument) else argument for argument in redacted]


def find_cmake_build_metadata(completion: Path) -> dict[str, str]:
    cache = next(
        (parent / "CMakeCache.txt" for parent in completion.parents if (parent / "CMakeCache.txt").is_file()),
        None,
    )
    if cache is None:
        return {}
    allowed = {
        "CMAKE_BUILD_TYPE",
        "CMAKE_GENERATOR",
        "GGML_AVX",
        "GGML_AVX2",
        "GGML_AVX512",
        "GGML_AVX512_VBMI",
        "GGML_AVX512_VNNI",
        "GGML_BLAS",
        "GGML_CUDA",
        "GGML_FMA",
        "GGML_LLAMAFILE",
        "GGML_NATIVE",
        "GGML_OPENMP",
    }
    metadata: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        typed_key, value = line.split("=", 1)
        key = typed_key.split(":", 1)[0]
        if key in allowed:
            metadata[key] = value
    return metadata


def version_metadata(completion: Path) -> dict[str, object]:
    process = subprocess.run(
        [str(completion), "--version"],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    output = "\n".join(part.strip() for part in (process.stdout, process.stderr) if part.strip())
    return {
        "sha256": sha256_file(completion),
        "version": redact_text(output, {str(completion): "<llama-completion>"}),
        "cmake": find_cmake_build_metadata(completion),
    }


def cpu_model() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except (OSError, ImportError):
            pass
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.is_file():
        for line in cpu_info.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in {"model name", "Hardware"}:
                return value.strip()
    return platform.processor() or "unknown"


def host_metadata() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
    }


def parse_metadata(entries: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for entry in entries:
        key, separator, value = entry.partition("=")
        if not separator or not key.strip():
            raise SystemExit(f"invalid --build-metadata value {entry!r}; expected KEY=VALUE")
        stripped_value = value.strip()
        metadata[key.strip()] = "<path>" if is_absolute_path(stripped_value) else stripped_value
    return metadata


def build_command(args: argparse.Namespace, completion: Path, model: Path, threads: int) -> list[str]:
    command = [
        str(completion),
        "--model",
        str(model),
        "--prompt",
        args.prompt,
        "--predict",
        str(args.predict),
        "--ctx-size",
        str(args.context_size),
        "--threads",
        str(threads),
        "--threads-batch",
        str(args.threads_batch or threads),
        "--batch-size",
        str(args.batch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--gpu-layers",
        "0",
        "--device",
        "none",
        "--flash-attn",
        "off",
        "--cache-type-k",
        args.cache_type_k,
        "--cache-type-v",
        args.cache_type_v,
        "--cpu-strict",
        args.cpu_strict,
        "--prio",
        str(args.priority),
        "--poll",
        str(args.poll),
        "--seed",
        str(args.seed),
        "--temp",
        str(args.temperature),
        "--perf",
        "--simple-io",
        "--no-display-prompt",
        "--log-colors",
        "off",
        "--no-log-timestamps",
    ]
    if args.cpu_mask:
        command.extend(("--cpu-mask", args.cpu_mask))
    if args.conversation_mode == "chat":
        command.extend(("--conversation", "--single-turn"))
    else:
        command.append("--no-conversation")
    command.append("--mmap" if args.mmap == "on" else "--no-mmap")
    command.append("--repack" if args.repack == "on" else "--no-repack")
    command.append("--warmup" if args.warmup == "on" else "--no-warmup")
    if args.ignore_eos:
        command.append("--ignore-eos")
    command.extend(args.extra_arg)
    return command


def summarize_runs(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    for threads in sorted({int(run["threads"]) for run in runs}):
        thread_runs = [run for run in runs if int(run["threads"]) == threads]
        latencies = [float(run["metrics"]["eval_ms_per_token"]) for run in thread_runs]  # type: ignore[index]
        throughputs = [float(run["metrics"]["eval_tokens_per_second"]) for run in thread_runs]  # type: ignore[index]
        eval_counts = {int(run["metrics"]["eval_count"]) for run in thread_runs}  # type: ignore[index]
        prompt_counts = {int(run["metrics"]["prompt_count"]) for run in thread_runs}  # type: ignore[index]
        if len(eval_counts) != 1 or len(prompt_counts) != 1:
            raise ValueError(f"T{threads} repetitions produced inconsistent prompt/eval token counts")
        summaries.append(
            {
                "threads": threads,
                "runs": len(thread_runs),
                "prompt_tokens": next(iter(prompt_counts)),
                "eval_tokens": next(iter(eval_counts)),
                "median_ms_per_token": statistics.median(latencies),
                "min_ms_per_token": min(latencies),
                "max_ms_per_token": max(latencies),
                "median_tokens_per_second": statistics.median(throughputs),
                "min_tokens_per_second": min(throughputs),
                "max_tokens_per_second": max(throughputs),
            }
        )
    return summaries


def write_markdown(path: Path, document: dict[str, object]) -> None:
    configuration = document["configuration"]
    binary = document["binary"]
    host = document["host"]
    lines = [
        "# llama.cpp CPU Actual-Completion Control",
        "",
        "## Reproducibility",
        "",
        f"- Host: `{host['cpu_model']}` "  # type: ignore[index]
        f"(`{host['logical_cpus']}` logical CPUs, `{host['platform']}`)",  # type: ignore[index]
        f"- Binary SHA-256: `{binary['sha256']}`",  # type: ignore[index]
        f"- Prompt SHA-256: `{document['prompt']['sha256']}`",  # type: ignore[index]
        f"- Prompt UTF-8 bytes: `{document['prompt']['utf8_bytes']}`",  # type: ignore[index]
        f"- Predicted tokens: `{configuration['predict']}`",  # type: ignore[index]
        f"- Context: `{configuration['context_size']}`",  # type: ignore[index]
        f"- Conversation mode: `{configuration['conversation_mode']}`",  # type: ignore[index]
        f"- KV cache: `{configuration['cache_type_k']}/{configuration['cache_type_v']}`",  # type: ignore[index]
        f"- mmap/repack/warmup: "
        f"`{configuration['mmap']}/{configuration['repack']}/{configuration['warmup']}`",  # type: ignore[index]
        f"- CPU strict/poll/priority: "
        f"`{configuration['cpu_strict']}/{configuration['poll']}/{configuration['priority']}`",  # type: ignore[index]
        "",
        "Version:",
        "",
        "```text",
        str(binary["version"]),  # type: ignore[index]
        "```",
        "",
    ]
    cmake = binary.get("cmake", {})  # type: ignore[union-attr]
    if cmake:
        lines.extend(("CMake configuration:", "", "```text"))
        lines.extend(f"{key}={value}" for key, value in sorted(cmake.items()))
        lines.extend(("```", ""))
    lines.extend(
        (
            "## Results",
            "",
            "| threads | runs | eval tokens | median ms/token | median tokens/s | min-max tokens/s |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for summary in document["summary"]:  # type: ignore[union-attr]
        lines.append(
            f"| {summary['threads']} | {summary['runs']} | {summary['eval_tokens']} | "
            f"{summary['median_ms_per_token']:.6g} | "
            f"{summary['median_tokens_per_second']:.6g} | {summary['min_tokens_per_second']:.6g}-"
            f"{summary['max_tokens_per_second']:.6g} |"
        )
    lines.extend(("", "Redacted command templates:", ""))
    for command in document["commands"]:  # type: ignore[union-attr]
        lines.extend(("```text", " ".join(command), "```", ""))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="GGUF model path; redacted from outputs")
    parser.add_argument("--llama-completion", type=Path, help="Path to llama-completion executable")
    parser.add_argument("--output-json", required=True, type=Path, help="Redacted structured result")
    parser.add_argument("--output-md", type=Path, help="Optional redacted human-readable result")
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--keep-prompt", action="store_true", help="Store prompt text instead of only its hash")
    parser.add_argument("--threads", nargs="+", default=[2, 4, 8, 16, 32], type=positive_int)
    parser.add_argument("--threads-batch", type=positive_int, help="Defaults to the generation thread count")
    parser.add_argument("--repetitions", default=3, type=positive_int)
    parser.add_argument("--predict", default=32, type=positive_int)
    parser.add_argument("--context-size", default=256, type=positive_int)
    parser.add_argument(
        "--conversation-mode",
        choices=["raw", "chat"],
        default="raw",
        help="Use the model chat template for one turn or decode the prompt as raw text",
    )
    parser.add_argument("--batch-size", default=2048, type=positive_int)
    parser.add_argument("--ubatch-size", default=512, type=positive_int)
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    parser.add_argument("--cpu-mask", default="")
    parser.add_argument("--cpu-strict", choices=["0", "1"], default="0")
    parser.add_argument("--poll", default=50, type=poll_level)
    parser.add_argument("--priority", default=0, type=priority)
    parser.add_argument("--mmap", choices=["on", "off"], default="on")
    parser.add_argument("--repack", choices=["on", "off"], default="on")
    parser.add_argument("--warmup", choices=["on", "off"], default="on")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--temperature", default=0.0, type=finite_float)
    parser.add_argument("--respect-eos", dest="ignore_eos", action="store_false")
    parser.set_defaults(ignore_eos=True)
    parser.add_argument("--build-metadata", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--extra-arg", action="append", default=[])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if len(set(args.threads)) != len(args.threads):
        raise SystemExit("--threads entries must be unique")
    completion = resolve_completion(args.llama_completion)
    model = args.model.resolve()
    if not model.is_file():
        raise SystemExit(f"model file not found: {args.model}")

    output_json = args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)

    binary = version_metadata(completion)
    binary["cmake"].update(parse_metadata(args.build_metadata))  # type: ignore[union-attr]
    replacements = {str(model): "<model>", str(completion): "<llama-completion>", args.prompt: "<prompt>"}
    replacements.update({argument: "<path>" for argument in args.extra_arg if is_absolute_path(argument)})
    runs: list[dict[str, object]] = []
    commands_by_threads: dict[int, list[str]] = {}
    for threads in args.threads:
        command = build_command(args, completion, model, threads)
        commands_by_threads[threads] = command
    commands = [
        redact_command(commands_by_threads[threads], completion, model, args.prompt) for threads in args.threads
    ]
    for repetition in range(1, args.repetitions + 1):
        ordered_threads = args.threads if repetition % 2 == 1 else list(reversed(args.threads))
        for threads in ordered_threads:
            command = commands_by_threads[threads]
            print(
                f"[llama.cpp control] T{threads} repetition {repetition}/{args.repetitions} starting",
                file=sys.stderr,
                flush=True,
            )
            started = time.perf_counter()
            process = subprocess.run(
                command,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
            )
            wall_seconds = time.perf_counter() - started
            stem = output_json.with_suffix("")
            stdout_path = Path(f"{stem}.t{threads}.r{repetition}.stdout.txt")
            stderr_path = Path(f"{stem}.t{threads}.r{repetition}.stderr.txt")
            stdout_path.write_text(redact_text(process.stdout, replacements), encoding="utf-8")
            stderr_path.write_text(redact_text(process.stderr, replacements), encoding="utf-8")
            if process.returncode != 0:
                raise SystemExit(
                    f"llama-completion failed for T{threads} repetition {repetition} with return code "
                    f"{process.returncode}; see {stderr_path}"
                )
            combined = f"{process.stdout}\n{process.stderr}"
            try:
                metrics = parse_perf_output(combined)
            except ValueError as exc:
                raise SystemExit(f"failed to parse T{threads} repetition {repetition}: {exc}") from exc
            system_info = next((line.strip() for line in combined.splitlines() if "system_info:" in line), "")
            print(
                f"[llama.cpp control] T{threads} repetition {repetition}/{args.repetitions} "
                f"finished in {wall_seconds:.2f}s: {metrics['eval_ms_per_token']:.3f} ms/token, "
                f"{metrics['eval_tokens_per_second']:.3f} t/s",
                file=sys.stderr,
                flush=True,
            )
            runs.append(
                {
                    "threads": threads,
                    "repetition": repetition,
                    "wall_seconds": wall_seconds,
                    "metrics": metrics,
                    "system_info": redact_text(system_info, replacements),
                    "stdout": stdout_path.name,
                    "stderr": stderr_path.name,
                }
            )

    document: dict[str, object] = {
        "schema_version": 1,
        "tool": "llama-completion",
        "host": host_metadata(),
        "binary": binary,
        "model": {"filename": "<model>", "size_bytes": model.stat().st_size},
        "prompt": prompt_metadata(args.prompt, args.keep_prompt),
        "configuration": {
            "threads": args.threads,
            "run_order": "alternating-forward-reverse",
            "threads_batch": args.threads_batch or "same-as-generation",
            "repetitions": args.repetitions,
            "predict": args.predict,
            "context_size": args.context_size,
            "conversation_mode": args.conversation_mode,
            "batch_size": args.batch_size,
            "ubatch_size": args.ubatch_size,
            "cache_type_k": args.cache_type_k,
            "cache_type_v": args.cache_type_v,
            "cpu_mask": args.cpu_mask,
            "cpu_strict": args.cpu_strict,
            "poll": args.poll,
            "priority": args.priority,
            "mmap": args.mmap,
            "repack": args.repack,
            "warmup": args.warmup,
            "seed": args.seed,
            "temperature": args.temperature,
            "ignore_eos": args.ignore_eos,
            "gpu_layers": 0,
            "device": "none",
            "flash_attention": "off",
            "extra_args": redact_command(args.extra_arg, completion, model, args.prompt),
        },
        "commands": commands,
        "runs": runs,
        "summary": summarize_runs(runs),
    }
    for summary in document["summary"]:  # type: ignore[union-attr]
        print(
            f"[llama.cpp control] T{summary['threads']} median: "
            f"{summary['median_ms_per_token']:.3f} ms/token, "
            f"{summary['median_tokens_per_second']:.3f} t/s",
            file=sys.stderr,
        )
    output_json.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.output_md is not None:
        write_markdown(args.output_md, document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
