#!/usr/bin/env python3
"""Run paired alternating LiteNN/llama.cpp GGUF decode controls."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import locale
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path, PurePosixPath, PureWindowsPath

from gguf_decode_compare import litenn_row, litenn_steady_generation_row
from run_llama_cpp_completion_control import (
    host_metadata,
    parse_perf_output,
    priority,
    prompt_metadata,
    sha256_file,
    version_metadata,
)


METRIC_RE = re.compile(r"(?P<name>[a-zA-Z0-9_]+)=(?P<value>[^\s]+)")


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


def poll_level(raw: str) -> int:
    value = int(raw)
    if value < 0 or value > 100:
        raise argparse.ArgumentTypeError("poll level must be in [0, 100]")
    return value


def is_absolute_path(value: str) -> bool:
    return Path(value).is_absolute() or PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def redact_text(text: str, replacements: dict[str, str]) -> str:
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
    result = text
    for source in sorted(candidates, key=len, reverse=True):
        result = result.replace(source, candidates[source])
    return result


def redact_command(command: list[str], replacements: dict[str, str]) -> list[str]:
    result = []
    for argument in command:
        redacted = redact_text(argument, replacements)
        result.append("<path>" if is_absolute_path(redacted) else redacted)
    return result


def decode_process_output(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        encoding = locale.getpreferredencoding(False) or "utf-8"
        return raw.decode(encoding, errors="replace")


def windows_frequency_sample() -> dict[str, object] | None:
    if sys.platform != "win32":
        return None

    class ProcessorPowerInformation(ctypes.Structure):
        _fields_ = [
            ("number", ctypes.c_ulong),
            ("max_mhz", ctypes.c_ulong),
            ("current_mhz", ctypes.c_ulong),
            ("mhz_limit", ctypes.c_ulong),
            ("max_idle_state", ctypes.c_ulong),
            ("current_idle_state", ctypes.c_ulong),
        ]

    processor_count = os.cpu_count() or 1
    entries = (ProcessorPowerInformation * processor_count)()
    try:
        power = ctypes.WinDLL("PowrProf.dll")
        function = power.CallNtPowerInformation
        function.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong]
        function.restype = ctypes.c_ulong
        status = function(11, None, 0, ctypes.byref(entries), ctypes.sizeof(entries))
    except (AttributeError, OSError):
        return None
    if status != 0:
        return None
    return {
        "source": "windows-processor-power-information",
        "current_mhz": [int(entry.current_mhz) for entry in entries],
        "limit_mhz": [int(entry.mhz_limit) for entry in entries],
        "max_mhz": [int(entry.max_mhz) for entry in entries],
    }


class WindowsPDHFrequencyMonitor:
    PDH_FORMAT_DOUBLE = 0x00000200
    PDH_MORE_DATA = 0x800007D2

    class FormattedValue(ctypes.Structure):
        _fields_ = [("status", wintypes.DWORD), ("double_value", ctypes.c_double)]

    class FormattedItem(ctypes.Structure):
        pass

    FormattedItem._fields_ = [("name", wintypes.LPWSTR), ("value", FormattedValue)]

    def __init__(self) -> None:
        self.query = ctypes.c_void_p()
        self.frequency_counter = ctypes.c_void_p()
        self.utility_counter = ctypes.c_void_p()
        self.pdh = None
        if sys.platform != "win32":
            return
        try:
            pdh = ctypes.WinDLL("pdh.dll")
            pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
            pdh.PdhOpenQueryW.restype = wintypes.LONG
            pdh.PdhAddEnglishCounterW.argtypes = [
                ctypes.c_void_p,
                wintypes.LPCWSTR,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            pdh.PdhAddEnglishCounterW.restype = wintypes.LONG
            pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
            pdh.PdhCollectQueryData.restype = wintypes.LONG
            pdh.PdhGetFormattedCounterArrayW.argtypes = [
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
                ctypes.POINTER(wintypes.DWORD),
                ctypes.c_void_p,
            ]
            pdh.PdhGetFormattedCounterArrayW.restype = wintypes.LONG
            pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
            pdh.PdhCloseQuery.restype = wintypes.LONG
            if self._status(pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query))) != 0:
                return
            if self._status(
                pdh.PdhAddEnglishCounterW(
                    self.query,
                    r"\Processor Information(*)\Actual Frequency",
                    0,
                    ctypes.byref(self.frequency_counter),
                )
            ) != 0 or self._status(
                pdh.PdhAddEnglishCounterW(
                    self.query,
                    r"\Processor Information(*)\% Processor Utility",
                    0,
                    ctypes.byref(self.utility_counter),
                )
            ) != 0:
                pdh.PdhCloseQuery(self.query)
                self.query = ctypes.c_void_p()
                return
            self.pdh = pdh
            pdh.PdhCollectQueryData(self.query)
        except (AttributeError, OSError):
            self.close()

    @staticmethod
    def _status(value: int) -> int:
        return value & 0xFFFFFFFF

    @property
    def available(self) -> bool:
        return self.pdh is not None and bool(self.query.value)

    def _formatted_values(self, counter: ctypes.c_void_p) -> dict[str, float]:
        assert self.pdh is not None
        size = wintypes.DWORD()
        count = wintypes.DWORD()
        status = self._status(
            self.pdh.PdhGetFormattedCounterArrayW(
                counter,
                self.PDH_FORMAT_DOUBLE,
                ctypes.byref(size),
                ctypes.byref(count),
                None,
            )
        )
        if status != self.PDH_MORE_DATA or size.value == 0:
            return {}
        buffer = ctypes.create_string_buffer(size.value)
        status = self._status(
            self.pdh.PdhGetFormattedCounterArrayW(
                counter,
                self.PDH_FORMAT_DOUBLE,
                ctypes.byref(size),
                ctypes.byref(count),
                buffer,
            )
        )
        if status != 0:
            return {}
        items = ctypes.cast(buffer, ctypes.POINTER(self.FormattedItem))
        return {
            str(items[index].name): float(items[index].value.double_value)
            for index in range(count.value)
            if items[index].name and items[index].value.status == 0
        }

    def sample(self) -> dict[str, object] | None:
        if not self.available or self.pdh is None:
            return None
        if self._status(self.pdh.PdhCollectQueryData(self.query)) != 0:
            return None
        frequencies = self._formatted_values(self.frequency_counter)
        utilities = self._formatted_values(self.utility_counter)
        names = sorted(
            name for name in frequencies.keys() & utilities.keys() if "_Total" not in name
        )
        if not names:
            return None
        valid_names = [name for name in names if frequencies[name] > 0.0]
        if len(valid_names) < len(names) // 2:
            return None
        current = [frequencies[name] for name in valid_names]
        utility = [max(0.0, utilities[name]) for name in valid_names]
        active = [
            frequencies[name]
            for name in valid_names
            if utilities[name] >= 10.0
        ]
        total_utility = sum(utility)
        weighted = (
            sum(frequencies[name] * max(0.0, utilities[name]) for name in valid_names) / total_utility
            if total_utility > 0.0
            else None
        )
        return {
            "source": "windows-pdh-processor-information",
            "current_mhz": current,
            "active_mhz": active,
            "utility_percent": utility,
            "weighted_actual_mhz": weighted,
            "monotonic_ns": time.monotonic_ns(),
        }

    def close(self) -> None:
        if self.pdh is not None and self.query.value:
            self.pdh.PdhCloseQuery(self.query)
        self.query = ctypes.c_void_p()
        self.pdh = None


class StatelessFrequencyMonitor:
    def sample(self) -> dict[str, object] | None:
        return frequency_sample()

    def close(self) -> None:
        pass


def create_frequency_monitor() -> WindowsPDHFrequencyMonitor | StatelessFrequencyMonitor:
    monitor = WindowsPDHFrequencyMonitor()
    if monitor.available:
        return monitor
    return StatelessFrequencyMonitor()


def linux_frequency_sample() -> dict[str, object] | None:
    if not sys.platform.startswith("linux"):
        return None
    current = []
    limits = []
    maximum = []
    for cpu_dir in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
        cpufreq = cpu_dir / "cpufreq"
        try:
            current.append(int((cpufreq / "scaling_cur_freq").read_text().strip()) // 1000)
            limits.append(int((cpufreq / "scaling_max_freq").read_text().strip()) // 1000)
            maximum.append(int((cpufreq / "cpuinfo_max_freq").read_text().strip()) // 1000)
        except (OSError, ValueError):
            continue
    if not current:
        return None
    return {
        "source": "linux-cpufreq",
        "current_mhz": current,
        "limit_mhz": limits,
        "max_mhz": maximum,
    }


def frequency_sample() -> dict[str, object] | None:
    sample = windows_frequency_sample() or linux_frequency_sample()
    if sample is not None:
        sample["monotonic_ns"] = time.monotonic_ns()
    return sample


def power_policy() -> dict[str, object]:
    if sys.platform == "win32":
        try:
            process = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, check=False)
            output = decode_process_output(process.stdout or process.stderr).strip()
            return {"source": "powercfg", "value": output or "unknown", "returncode": process.returncode}
        except OSError:
            return {"source": "powercfg", "value": "unavailable"}
    if sys.platform.startswith("linux"):
        governors = set()
        for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor"):
            try:
                governors.add(path.read_text().strip())
            except OSError:
                continue
        return {"source": "linux-cpufreq", "value": sorted(governors) or ["unknown"]}
    return {"source": "platform", "value": platform.platform()}


def summarize_frequency(samples: list[dict[str, object]]) -> dict[str, object]:
    current = [
        float(value) for sample in samples for value in sample.get("current_mhz", [])  # type: ignore[union-attr]
    ]
    limits = [float(value) for sample in samples for value in sample.get("limit_mhz", [])]  # type: ignore[union-attr]
    maximum = [float(value) for sample in samples for value in sample.get("max_mhz", [])]  # type: ignore[union-attr]
    active = [float(value) for sample in samples for value in sample.get("active_mhz", [])]  # type: ignore[union-attr]
    utility = [
        float(value) for sample in samples for value in sample.get("utility_percent", [])  # type: ignore[union-attr]
    ]
    weighted = [
        float(value)
        for sample in samples
        if (value := sample.get("weighted_actual_mhz")) is not None
    ]
    if not current:
        return {"available": False, "sample_count": len(samples)}
    return {
        "available": True,
        "source": samples[0].get("source", "unknown"),
        "sample_count": len(samples),
        "logical_cpu_observations": len(current),
        "current_mhz_min": min(current),
        "current_mhz_median": statistics.median(current),
        "current_mhz_max": max(current),
        "limit_mhz_min": min(limits) if limits else None,
        "limit_mhz_max": max(limits) if limits else None,
        "hardware_max_mhz": max(maximum) if maximum else None,
        "active_mhz_min": min(active) if active else None,
        "active_mhz_median": statistics.median(active) if active else None,
        "active_mhz_max": max(active) if active else None,
        "weighted_actual_mhz_median": statistics.median(weighted) if weighted else None,
        "utility_percent_median": statistics.median(utility) if utility else None,
        "utility_percent_max": max(utility) if utility else None,
    }


def run_monitored(
    command: list[str],
    artifact_prefix: Path,
    replacements: dict[str, str],
    monitor_interval_seconds: float,
) -> tuple[dict[str, object], str, str]:
    artifact_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_stdout = artifact_prefix.with_suffix(".raw.stdout.txt")
    raw_stderr = artifact_prefix.with_suffix(".raw.stderr.txt")
    started = time.perf_counter()
    policy_before = power_policy()
    samples: list[dict[str, object]] = []
    frequency_monitor = create_frequency_monitor()
    initial_sample = frequency_monitor.sample()
    if initial_sample is not None:
        samples.append(initial_sample)
    try:
        with raw_stdout.open("wb") as stdout_stream, raw_stderr.open("wb") as stderr_stream:
            process = subprocess.Popen(command, stdout=stdout_stream, stderr=stderr_stream)
            while True:
                try:
                    process.wait(timeout=monitor_interval_seconds)
                    break
                except subprocess.TimeoutExpired:
                    sample = frequency_monitor.sample()
                    if sample is not None:
                        samples.append(sample)
        final_sample = frequency_monitor.sample()
    finally:
        frequency_monitor.close()
    if final_sample is not None:
        samples.append(final_sample)
    wall_seconds = time.perf_counter() - started
    policy_after = power_policy()
    stdout_text = raw_stdout.read_bytes().decode("utf-8", errors="replace")
    stderr_text = raw_stderr.read_bytes().decode("utf-8", errors="replace")
    stdout_path = artifact_prefix.with_suffix(".stdout.txt")
    stderr_path = artifact_prefix.with_suffix(".stderr.txt")
    stdout_path.write_text(redact_text(stdout_text, replacements), encoding="utf-8")
    stderr_path.write_text(redact_text(stderr_text, replacements), encoding="utf-8")
    raw_stdout.unlink()
    raw_stderr.unlink()
    return (
        {
            "returncode": process.returncode,
            "wall_seconds": wall_seconds,
            "frequency": summarize_frequency(samples),
            "power_policy_before": policy_before,
            "power_policy_after": policy_after,
            "stdout": stdout_path.name,
            "stderr": stderr_path.name,
            "command": redact_command(command, replacements),
        },
        stdout_text,
        stderr_text,
    )


def series_statistics(values: list[float]) -> dict[str, float | int]:
    mean = statistics.mean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "standard_deviation": standard_deviation,
        "coefficient_of_variation_percent": standard_deviation * 100.0 / mean if mean > 0.0 else 0.0,
        "spread_percent": (max(values) / min(values) - 1.0) * 100.0 if min(values) > 0.0 else 0.0,
    }


def text_identity(text: str) -> dict[str, object]:
    encoded = text.encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "utf8_bytes": len(encoded)}


def normalize_completion_text(stdout_text: str) -> str:
    # llama-completion writes model LF bytes through the Windows text-mode stdout.
    # Undo that platform translation before recovering the generated token stream.
    return stdout_text.strip().replace("\r\n", "\n").replace("\r", "\n")


def binary_identity(path: Path) -> dict[str, object]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def token_ids_identity(token_ids: list[int]) -> dict[str, object]:
    encoded = ",".join(str(token_id) for token_id in token_ids).encode("ascii")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "count": len(token_ids)}


def load_tokenizer_token_ids(path: Path) -> list[int]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "litenn.llamacpp_tokens.v1":
        raise RuntimeError(f"unsupported tokenizer output schema: {document.get('schema')!r}")
    values = document.get("tokenIds")
    if not isinstance(values, list) or not values or any(not isinstance(value, int) or value < 0 for value in values):
        raise RuntimeError("tokenizer output contains invalid token ids")
    return values


def parse_forced_replay_metrics(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise RuntimeError(f"LiteNN decode output is missing its metrics row: {path}")
    values = {match.group("name"): match.group("value") for match in METRIC_RE.finditer(lines[2])}
    if "forced_replay" not in values or "forced_token_mismatch_count" not in values:
        raise RuntimeError(f"LiteNN decode output is missing fixed-replay metrics: {path}")
    first_mismatch = values.get("first_forced_token_mismatch_index")
    return {
        "enabled": values["forced_replay"] == "true",
        "natural_mismatch_count": int(values["forced_token_mismatch_count"]),
        "first_natural_mismatch_index": int(first_mismatch) if first_mismatch is not None else None,
    }


def load_litenn_generated_token_ids(path: Path, prompt_token_count: int) -> list[int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError(f"LiteNN decode output contains no token ids: {path}")
    values = json.loads(lines[0])
    if not isinstance(values, list) or any(not isinstance(value, int) or value < 0 for value in values):
        raise RuntimeError(f"LiteNN decode output contains invalid token ids: {path}")
    if prompt_token_count < 0 or len(values) < prompt_token_count:
        raise RuntimeError(f"LiteNN decode output is shorter than its prompt: {path}")
    return values[prompt_token_count:]


def build_llama_command(args: argparse.Namespace, model: Path, completion: Path) -> list[str]:
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
        str(args.llama_threads),
        "--threads-batch",
        str(args.llama_threads),
        "--batch-size",
        "2048",
        "--ubatch-size",
        "512",
        "--gpu-layers",
        "0",
        "--device",
        "none",
        "--flash-attn",
        "off",
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
        "--cpu-strict",
        args.llama_cpu_strict,
        "--prio",
        str(args.llama_priority),
        "--poll",
        str(args.llama_poll),
        "--seed",
        "42",
        "--temp",
        "0.0",
        "--perf",
        "--simple-io",
        "--no-display-prompt",
        "--log-colors",
        "off",
        "--no-log-timestamps",
        "--conversation",
        "--single-turn",
        "--mmap",
        "--repack",
        "--warmup",
        "--ignore-eos",
    ]
    if args.llama_cpu_mask:
        command.extend(("--cpu-mask", args.llama_cpu_mask))
    return command


def build_litenn_command(
    args: argparse.Namespace,
    model: Path,
    litenn: Path,
    tokenizer: Path,
    workdir: Path,
    cache_dir: Path,
    forced_generated_token_ids: list[int] | None = None,
) -> list[str]:
    command = [
        args.python,
        str(repo_root() / "example" / "gguf" / "qwen_smoke.py"),
        "--model",
        str(model),
        "--litenn",
        str(litenn),
        "--llamacpp-tokenizer-tool",
        str(tokenizer),
        "--prompt",
        args.prompt,
        "--stateful",
        "--max-tokens",
        str(args.predict),
        "--workdir",
        str(workdir),
        "--aot-cache-dir",
        str(cache_dir),
        "--require-aot-cache-hit",
        "--stream-stats",
        "--ignore-eos",
        "--llvm-opt-level",
        str(args.llvm_opt_level),
        "--cpu-aot-threads",
        str(args.litenn_threads),
        "--cpu-aot-worker-wait",
        args.litenn_worker_wait,
        "--cpu-aot-ggml-prepacked-weight-policy",
        "all",
        "--cpu-aot-ggml-prepacked-weight-layout",
        "field-interleaved-v4",
    ]
    if args.litenn_affinity != "default":
        command.extend(("--cpu-aot-affinity", args.litenn_affinity))
    if forced_generated_token_ids is not None:
        command.extend(
            ("--forced-generated-token-ids", ",".join(str(token_id) for token_id in forced_generated_token_ids))
        )
    return command


def capture_fixed_reference_trajectory(
    args: argparse.Namespace,
    model: Path,
    completion: Path,
    tokenizer: Path,
    output_dir: Path,
    replacements: dict[str, str],
) -> tuple[list[int], dict[str, object]]:
    capture_dir = output_dir / "fixed_replay_reference"
    capture_dir.mkdir(parents=True, exist_ok=True)
    llama_process, stdout_text, stderr_text = run_monitored(
        build_llama_command(args, model, completion),
        capture_dir / "llama_cpp",
        replacements,
        args.monitor_interval,
    )
    if llama_process["returncode"] != 0:
        raise RuntimeError("llama.cpp fixed-trajectory capture failed")
    generated_text = normalize_completion_text(stdout_text)
    if not generated_text:
        raise RuntimeError("llama.cpp fixed-trajectory capture produced no text")
    perf = parse_perf_output(f"{stdout_text}\n{stderr_text}")
    generated_text_input = capture_dir / "generated_text.bin"
    generated_text_input.write_bytes(generated_text.encode("utf-8"))
    token_output = capture_dir / "generated_tokens.json"
    tokenizer_process, _, _ = run_monitored(
        [str(tokenizer), "tokenize-file", str(model), str(generated_text_input), str(token_output)],
        capture_dir / "tokenize",
        replacements,
        args.monitor_interval,
    )
    if tokenizer_process["returncode"] != 0:
        raise RuntimeError("fixed-trajectory tokenization failed")
    token_ids = load_tokenizer_token_ids(token_output)
    if len(token_ids) != args.predict:
        raise RuntimeError(
            f"fixed-trajectory token count mismatch: expected {args.predict}, tokenized {len(token_ids)}"
        )
    return token_ids, {
        "source": "unmeasured_llama_cpp_completion_then_tokenizer_round_trip",
        "llama_cpp_process": llama_process,
        "tokenizer_process": tokenizer_process,
        "text": text_identity(generated_text),
        "tokens": token_ids_identity(token_ids),
        "prompt_tokens": perf["prompt_count"],
        "eval_tokens": perf["eval_count"],
        "ms_per_token": perf["eval_ms_per_token"],
        "tokens_per_second": perf["eval_tokens_per_second"],
    }


def write_markdown(path: Path, document: dict[str, object]) -> None:
    summary = document["summary"]
    gate = document["gate"]
    host = document["host"]
    lines = [
        "# Paired LiteNN / llama.cpp GGUF Decode Control",
        "",
        f"- Host: `{host['cpu_model']}` ({host['logical_cpus']} logical CPUs)",  # type: ignore[index]
        f"- Prompt SHA-256: `{document['prompt']['sha256']}`",  # type: ignore[index]
        f"- Prompt/eval tokens: `{document['configuration']['prompt_tokens']}/"  # type: ignore[index]
        f"{document['configuration']['eval_tokens']}`",  # type: ignore[index]
        f"- Variance threshold: `{document['configuration']['variance_threshold_percent']}%`",  # type: ignore[index]
        f"- Trajectory mode: `{'fixed reference replay' if document['configuration']['fixed_token_replay'] else 'natural greedy'}`",  # type: ignore[index]
        "",
        "| Pair | Order | llama ms/token | llama t/s | LiteNN ms/token | LiteNN t/s | LiteNN delta | "
        "llama MHz | LiteNN MHz | Trajectory parity |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for pair in document["pairs"]:  # type: ignore[union-attr]
        llama = pair["llama_cpp"]
        litenn = pair["litenn"]
        llama_frequency = llama["process"]["frequency"]
        litenn_frequency = litenn["process"]["frequency"]

        def frequency_value(value: dict[str, object]) -> str:
            raw = value.get("weighted_actual_mhz_median") or value.get("active_mhz_median")
            if raw is None:
                raw = value.get("current_mhz_median")
            return f"{float(raw):.0f}" if raw is not None else "n/a"

        lines.append(
            f"| {pair['repetition']} | {' -> '.join(pair['order'])} | "
            f"{llama['ms_per_token']:.3f} | {llama['tokens_per_second']:.3f} | "
            f"{litenn['ms_per_token']:.3f} | {litenn['tokens_per_second']:.3f} | "
            f"{pair['litenn_vs_llama_percent']:+.2f}% | {frequency_value(llama_frequency)} | "
            f"{frequency_value(litenn_frequency)} | {pair['trajectory_match']} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- llama.cpp median: `{summary['llama_cpp']['median']:.3f} t/s`, "  # type: ignore[index]
            f"CV `{summary['llama_cpp']['coefficient_of_variation_percent']:.2f}%`",  # type: ignore[index]
            f"- LiteNN median: `{summary['litenn']['median']:.3f} t/s`, "  # type: ignore[index]
            f"CV `{summary['litenn']['coefficient_of_variation_percent']:.2f}%`",  # type: ignore[index]
            f"- Median paired LiteNN delta: `{summary['paired_delta_percent']['median']:+.2f}%`",  # type: ignore[index]
            f"- Text parity: `{gate['text_parity']}`",  # type: ignore[index]
            f"- Text parity kind: `{gate['text_parity_kind']}`",  # type: ignore[index]
            f"- Trajectory parity: `{gate['trajectory_parity']}`",  # type: ignore[index]
            f"- Natural sampler parity: `{gate['natural_sampler_parity']}`",  # type: ignore[index]
            f"- No fallback: `{gate['no_fallback']}`",  # type: ignore[index]
            f"- Variance gate: `{gate['variance']}`",  # type: ignore[index]
            f"- Accepted: `{gate['accepted']}`",  # type: ignore[index]
        ]
    )
    if document["configuration"]["fixed_token_replay"]:  # type: ignore[index]
        mismatch_summary = summary["natural_mismatch_count"]
        lines.extend(
            [
                f"- LiteNN natural mismatch count median: `{mismatch_summary['median']:.0f}`",  # type: ignore[index]
                f"- First natural mismatch indices: `{summary['first_natural_mismatch_indices']}`",  # type: ignore[index]
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--litenn", required=True, type=Path)
    parser.add_argument("--llamacpp-tokenizer-tool", required=True, type=Path)
    parser.add_argument("--llama-completion", required=True, type=Path)
    parser.add_argument("--aot-cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--predict", default=16, type=positive_int)
    parser.add_argument("--context-size", default=256, type=positive_int)
    parser.add_argument("--repetitions", default=3, type=positive_int)
    parser.add_argument("--litenn-threads", default=8, type=positive_int)
    parser.add_argument("--litenn-affinity", choices=["default", "none", "compact", "spread"], default="default")
    parser.add_argument(
        "--litenn-worker-wait", choices=["adaptive", "low-power", "latency"], default="adaptive"
    )
    parser.add_argument("--llvm-opt-level", choices=[0, 1, 2, 3], default=0, type=int)
    parser.add_argument("--llama-threads", default=2, type=positive_int)
    parser.add_argument("--llama-cpu-mask", default="")
    parser.add_argument("--llama-cpu-strict", choices=["0", "1"], default="0")
    parser.add_argument("--llama-poll", default=50, type=poll_level)
    parser.add_argument("--llama-priority", default=0, type=priority)
    parser.add_argument("--monitor-interval", default=0.25, type=positive_float)
    parser.add_argument("--variance-threshold-percent", default=3.0, type=positive_float)
    parser.add_argument("--require-variance-gate", action="store_true")
    parser.add_argument(
        "--fixed-token-replay",
        action="store_true",
        help="Capture one llama.cpp trajectory, force LiteNN to replay it, and report natural sampler divergences",
    )
    parser.add_argument("--keep-prompt", action="store_true")
    return parser


def resolve_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise SystemExit(f"{label} not found: {path}")
    return resolved


def main() -> int:
    args = build_parser().parse_args()
    if args.repetitions < 3:
        raise SystemExit("paired control requires at least three repetitions")
    model = resolve_file(args.model, "model")
    litenn = resolve_file(args.litenn, "LiteNN GGUF tool")
    tokenizer = resolve_file(args.llamacpp_tokenizer_tool, "llama.cpp tokenizer tool")
    completion = resolve_file(args.llama_completion, "llama-completion")
    cache_dir = args.aot_cache_dir.resolve()
    if not cache_dir.is_dir():
        raise SystemExit(f"AOT cache directory not found: {args.aot_cache_dir}")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    replacements = {
        str(model): "<model>",
        str(litenn): "<litenn-gguf>",
        str(tokenizer): "<llamacpp-tokenizer>",
        str(completion): "<llama-completion>",
        str(repo_root()): "<repo>",
        str(Path.cwd()): "<repo>",
        str(args.aot_cache_dir.absolute()): "<aot-cache>",
        str(cache_dir): "<aot-cache>",
        args.prompt: "<prompt>",
    }
    document: dict[str, object] = {
        "schema_version": 2,
        "tool": "litenn-paired-gguf-decode-control",
        "status": "running",
        "host": host_metadata(),
        "power_policy": power_policy(),
        "llama_cpp_binary": version_metadata(completion),
        "litenn_binary": binary_identity(litenn),
        "llamacpp_tokenizer_binary": binary_identity(tokenizer),
        "model": {"filename": "<model>", "size_bytes": model.stat().st_size},
        "prompt": prompt_metadata(args.prompt, args.keep_prompt),
        "configuration": {
            "repetitions": args.repetitions,
            "predict": args.predict,
            "context_size": args.context_size,
            "prompt_tokens": None,
            "eval_tokens": None,
            "litenn_threads": args.litenn_threads,
            "litenn_affinity": args.litenn_affinity,
            "litenn_worker_wait": args.litenn_worker_wait,
            "llvm_opt_level": args.llvm_opt_level,
            "llama_threads": args.llama_threads,
            "llama_cpu_mask": args.llama_cpu_mask,
            "llama_cpu_strict": args.llama_cpu_strict,
            "llama_poll": args.llama_poll,
            "llama_priority": args.llama_priority,
            "monitor_interval_seconds": args.monitor_interval,
            "variance_threshold_percent": args.variance_threshold_percent,
            "fixed_token_replay": args.fixed_token_replay,
            "run_order": "odd=llama_cpp_then_litenn,even=litenn_then_llama_cpp",
        },
        "pairs": [],
    }
    output_json = output_dir / "paired_gguf_decode_control.json"
    output_md = output_dir / "paired_gguf_decode_control.md"

    def checkpoint() -> None:
        output_json.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    checkpoint()
    forced_generated_token_ids: list[int] | None = None
    if args.fixed_token_replay:
        print("[paired decode] capturing fixed reference trajectory", file=sys.stderr, flush=True)
        try:
            forced_generated_token_ids, fixed_reference = capture_fixed_reference_trajectory(
                args, model, completion, tokenizer, output_dir, replacements
            )
        except RuntimeError as error:
            document["status"] = "failed"
            document["failure"] = str(error)
            checkpoint()
            raise SystemExit(str(error)) from error
        document["fixed_replay_reference"] = fixed_reference
        print("[paired decode] fixed reference trajectory ready", file=sys.stderr, flush=True)
        checkpoint()
    pairs_document = document["pairs"]
    assert isinstance(pairs_document, list)
    for repetition in range(1, args.repetitions + 1):
        order = ["llama_cpp", "litenn"] if repetition % 2 == 1 else ["litenn", "llama_cpp"]
        pair: dict[str, object] = {"repetition": repetition, "order": order}
        pairs_document.append(pair)
        pair_dir = output_dir / f"pair_{repetition:02d}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        for runtime in order:
            print(
                f"[paired decode] pair {repetition}/{args.repetitions} {runtime} starting",
                file=sys.stderr,
                flush=True,
            )
            if runtime == "llama_cpp":
                command = build_llama_command(args, model, completion)
                process_record, stdout_text, stderr_text = run_monitored(
                    command, pair_dir / "llama_cpp", replacements, args.monitor_interval
                )
                if process_record["returncode"] != 0:
                    document["status"] = "failed"
                    document["failure"] = f"llama.cpp failed in pair {repetition}"
                    checkpoint()
                    raise SystemExit(document["failure"])
                metrics = parse_perf_output(f"{stdout_text}\n{stderr_text}")
                generated_text = normalize_completion_text(stdout_text)
                pair["llama_cpp"] = {
                    "process": process_record,
                    "prompt_tokens": metrics["prompt_count"],
                    "eval_tokens": metrics["eval_count"],
                    "ms_per_token": metrics["eval_ms_per_token"],
                    "tokens_per_second": metrics["eval_tokens_per_second"],
                    "text": text_identity(generated_text),
                }
            else:
                workdir = pair_dir / "litenn_workdir"
                command = build_litenn_command(
                    args, model, litenn, tokenizer, workdir, cache_dir, forced_generated_token_ids
                )
                process_record, _, _ = run_monitored(
                    command, pair_dir / "litenn", replacements, args.monitor_interval
                )
                if process_record["returncode"] != 0:
                    document["status"] = "failed"
                    document["failure"] = f"LiteNN failed in pair {repetition}"
                    checkpoint()
                    raise SystemExit(document["failure"])
                report_path = workdir / "qwen_smoke_report.json"
                base = litenn_row(report_path)
                steady = litenn_steady_generation_row(report_path, base)
                generated_text = (workdir / "generated_text.bin").read_text(encoding="utf-8").strip()
                token_output = workdir / "litenn_decode_tokens.txt"
                replay = parse_forced_replay_metrics(token_output)
                generated_token_ids = load_litenn_generated_token_ids(token_output, int(base["promptTokens"]))
                pair["litenn"] = {
                    "process": process_record,
                    "prompt_tokens": base["promptTokens"],
                    "eval_tokens": steady["tokens"],
                    "ms_per_token": steady["msPerToken"],
                    "tokens_per_second": steady["tokensPerSecond"],
                    "full_generation_ms_per_token": base["msPerToken"],
                    "full_generation_tokens_per_second": base["tokensPerSecond"],
                    "fallback_used": base["fallbackUsed"],
                    "fallback_count": base["fallbackCount"],
                    "text": text_identity(generated_text),
                    "tokens": token_ids_identity(generated_token_ids),
                    "fixed_replay": replay,
                }
            print(
                f"[paired decode] pair {repetition}/{args.repetitions} {runtime} finished",
                file=sys.stderr,
                flush=True,
            )
            checkpoint()

        llama = pair["llama_cpp"]
        litenn_result = pair["litenn"]
        pair["text_match"] = llama["text"] == litenn_result["text"]  # type: ignore[index]
        if args.fixed_token_replay:
            assert forced_generated_token_ids is not None
            fixed_reference = document["fixed_replay_reference"]
            pair["trajectory_match"] = (
                litenn_result["tokens"] == token_ids_identity(forced_generated_token_ids)  # type: ignore[index]
                and llama["text"] == fixed_reference["text"]  # type: ignore[index]
            )
        else:
            pair["trajectory_match"] = pair["text_match"]
        pair["litenn_vs_llama_percent"] = (
            float(litenn_result["tokens_per_second"]) / float(llama["tokens_per_second"]) - 1.0  # type: ignore[index]
        ) * 100.0
        if document["configuration"]["prompt_tokens"] is None:  # type: ignore[index]
            document["configuration"]["prompt_tokens"] = llama["prompt_tokens"]  # type: ignore[index]
            document["configuration"]["eval_tokens"] = llama["eval_tokens"]  # type: ignore[index]
        if int(llama["eval_tokens"]) != int(litenn_result["eval_tokens"]):  # type: ignore[index]
            document["status"] = "failed"
            document["failure"] = f"eval token window mismatch in pair {repetition}"
        elif int(llama["prompt_tokens"]) != int(litenn_result["prompt_tokens"]):  # type: ignore[index]
            document["status"] = "failed"
            document["failure"] = f"prompt token count mismatch in pair {repetition}"
        elif args.fixed_token_replay and not bool(litenn_result["fixed_replay"]["enabled"]):  # type: ignore[index]
            document["status"] = "failed"
            document["failure"] = f"LiteNN fixed replay was not enabled in pair {repetition}"
        elif not pair["trajectory_match"]:
            document["status"] = "failed"
            document["failure"] = f"generated trajectory mismatch in pair {repetition}"
        elif bool(litenn_result["fallback_used"]):  # type: ignore[index]
            document["status"] = "failed"
            document["failure"] = f"LiteNN fallback in pair {repetition}"
        checkpoint()
        if document["status"] == "failed":
            raise SystemExit(document["failure"])

    pairs = pairs_document
    llama_values = [float(pair["llama_cpp"]["tokens_per_second"]) for pair in pairs]  # type: ignore[index]
    litenn_values = [float(pair["litenn"]["tokens_per_second"]) for pair in pairs]  # type: ignore[index]
    deltas = [float(pair["litenn_vs_llama_percent"]) for pair in pairs]  # type: ignore[index]
    summary = {
        "llama_cpp": series_statistics(llama_values),
        "litenn": series_statistics(litenn_values),
        "paired_delta_percent": series_statistics(deltas),
    }
    if args.fixed_token_replay:
        mismatch_counts = [
            float(pair["litenn"]["fixed_replay"]["natural_mismatch_count"]) for pair in pairs  # type: ignore[index]
        ]
        summary["natural_mismatch_count"] = series_statistics(mismatch_counts)
        summary["first_natural_mismatch_indices"] = [
            pair["litenn"]["fixed_replay"]["first_natural_mismatch_index"] for pair in pairs  # type: ignore[index]
        ]
    variance_passed = (
        float(summary["llama_cpp"]["coefficient_of_variation_percent"]) <= args.variance_threshold_percent
        and float(summary["litenn"]["coefficient_of_variation_percent"]) <= args.variance_threshold_percent
    )
    gate = {
        "text_parity": all(bool(pair["text_match"]) for pair in pairs),  # type: ignore[index]
        "text_parity_kind": "fixed_reference_trajectory" if args.fixed_token_replay else "natural_greedy",
        "trajectory_parity": all(bool(pair["trajectory_match"]) for pair in pairs),  # type: ignore[index]
        "natural_sampler_parity": (
            all(
                int(pair["litenn"]["fixed_replay"]["natural_mismatch_count"]) == 0  # type: ignore[index]
                for pair in pairs
            )
            if args.fixed_token_replay
            else all(bool(pair["text_match"]) for pair in pairs)  # type: ignore[index]
        ),
        "no_fallback": all(not bool(pair["litenn"]["fallback_used"]) for pair in pairs),  # type: ignore[index]
        "variance": variance_passed,
    }
    gate["accepted"] = bool(gate["trajectory_parity"] and gate["no_fallback"] and gate["variance"])
    document["summary"] = summary
    document["gate"] = gate
    document["status"] = "complete"
    document["power_policy_after"] = power_policy()
    checkpoint()
    write_markdown(output_md, document)
    print(
        f"[paired decode] LiteNN median {summary['litenn']['median']:.3f} t/s, "
        f"llama.cpp median {summary['llama_cpp']['median']:.3f} t/s, "
        f"paired delta {summary['paired_delta_percent']['median']:+.2f}%, "
        f"variance_gate={variance_passed}",
        file=sys.stderr,
    )
    return 2 if args.require_variance_gate and not variance_passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
