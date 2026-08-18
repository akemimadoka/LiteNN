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

try:
    from .gguf_decode_compare import litenn_row, litenn_steady_generation_row
    from .process_memory import ProcessMemorySampler
    from .run_llama_cpp_completion_control import (
        host_metadata,
        parse_perf_output,
        priority,
        prompt_metadata,
        sha256_file,
        version_metadata,
    )
except ImportError:
    from gguf_decode_compare import litenn_row, litenn_steady_generation_row
    from process_memory import ProcessMemorySampler
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


def non_negative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return value


def poll_level(raw: str) -> int:
    value = int(raw)
    if value < 0 or value > 100:
        raise argparse.ArgumentTypeError("poll level must be in [0, 100]")
    return value


def cpu_set(raw: str) -> list[int]:
    result: set[int] = set()
    try:
        for part in raw.split(","):
            lower, separator, upper = part.strip().partition("-")
            if not lower:
                raise ValueError
            first = int(lower)
            last = int(upper) if separator else first
            if first < 0 or last < first:
                raise ValueError
            result.update(range(first, last + 1))
    except ValueError as error:
        raise argparse.ArgumentTypeError("CPU set must use non-negative comma-separated ids or ranges") from error
    if not result:
        raise argparse.ArgumentTypeError("CPU set must not be empty")
    return sorted(result)


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
        active_names = [name for name in valid_names if utilities[name] >= 10.0]
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
            "host_utility_percent_mean": sum(utility) / len(utility) if utility else None,
            "active_logical_cpus": active_names,
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
    load = os.getloadavg() if hasattr(os, "getloadavg") else None
    return {
        "source": "linux-cpufreq",
        "current_mhz": current,
        "limit_mhz": limits,
        "max_mhz": maximum,
        "host_load_1m": load[0] if load is not None else None,
        "host_load_5m": load[1] if load is not None else None,
        "host_load_15m": load[2] if load is not None else None,
    }


def frequency_sample() -> dict[str, object] | None:
    sample = windows_frequency_sample() or linux_frequency_sample()
    if sample is not None:
        sample["monotonic_ns"] = time.monotonic_ns()
    return sample


def host_activity_percent(sample: dict[str, object], logical_cpus: int | None = None) -> tuple[str, float] | None:
    utility = sample.get("host_utility_percent_mean")
    if isinstance(utility, (int, float)) and not isinstance(utility, bool) and math.isfinite(float(utility)):
        return "processor_utility_mean", float(utility)
    load = sample.get("host_load_1m")
    cpu_count = logical_cpus or os.cpu_count() or 1
    if isinstance(load, (int, float)) and not isinstance(load, bool) and math.isfinite(float(load)):
        return "load_1m_per_logical_cpu", float(load) * 100.0 / cpu_count
    return None


def wait_for_host_admission(
    maximum_activity_percent: float | None,
    consecutive_samples: int,
    warmup_samples: int,
    timeout_seconds: float,
    sample_interval_seconds: float,
) -> dict[str, object]:
    if maximum_activity_percent is None:
        return {"schema": "litenn.host_admission.v1", "enabled": False, "passed": True}

    started_ns = time.monotonic_ns()
    deadline = time.monotonic() + timeout_seconds
    samples: list[dict[str, object]] = []
    accepted_streak = 0
    activity_source: str | None = None
    monitor = create_frequency_monitor()
    try:
        attempt = 0
        while True:
            sample = monitor.sample()
            if sample is not None:
                sample = dict(sample)
                sample["admission_warmup"] = attempt < warmup_samples
                samples.append(sample)
                activity = host_activity_percent(sample)
                if attempt >= warmup_samples and activity is not None:
                    activity_source, activity_value = activity
                    accepted_streak = accepted_streak + 1 if activity_value <= maximum_activity_percent else 0
                    if accepted_streak >= consecutive_samples:
                        break
                elif attempt >= warmup_samples:
                    accepted_streak = 0
            attempt += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(sample_interval_seconds, remaining))
    finally:
        monitor.close()

    observations = [
        activity[1]
        for sample in samples
        if not bool(sample.get("admission_warmup"))
        and (activity := host_activity_percent(sample)) is not None
    ]
    passed = accepted_streak >= consecutive_samples
    return {
        "schema": "litenn.host_admission.v1",
        "enabled": True,
        "passed": passed,
        "activity_source": activity_source,
        "maximum_activity_percent": maximum_activity_percent,
        "required_consecutive_samples": consecutive_samples,
        "warmup_samples": warmup_samples,
        "accepted_streak": accepted_streak,
        "waited_ms": (time.monotonic_ns() - started_ns) / 1_000_000.0,
        "sample_count": len(samples),
        "activity_percent": {
            "minimum": min(observations) if observations else None,
            "median": statistics.median(observations) if observations else None,
            "maximum": max(observations) if observations else None,
        },
        "samples": samples,
    }


def cooldown_record(seconds: float, kind: str) -> dict[str, object]:
    started_ns = time.monotonic_ns()
    if seconds > 0.0:
        time.sleep(seconds)
    ended_ns = time.monotonic_ns()
    return {
        "schema": "litenn.benchmark_cooldown.v1",
        "kind": kind,
        "requested_seconds": seconds,
        "actual_ms": (ended_ns - started_ns) / 1_000_000.0,
    }


def apply_process_affinity(pid: int, cpu_ids: list[int] | None) -> dict[str, object]:
    if not cpu_ids:
        return {"schema": "litenn.process_affinity.v1", "requested_cpu_ids": [], "applied": False}
    if sys.platform == "win32":
        bit_count = ctypes.sizeof(ctypes.c_size_t) * 8
        if cpu_ids[-1] >= bit_count:
            raise RuntimeError(f"Windows process CPU id {cpu_ids[-1]} exceeds the {bit_count}-bit affinity mask")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x0200 | 0x1000, False, pid)
        if not handle:
            raise OSError(ctypes.get_last_error(), f"OpenProcess({pid}) for affinity failed")
        try:
            mask = sum(1 << cpu_id for cpu_id in cpu_ids)
            if not kernel32.SetProcessAffinityMask(handle, mask):
                raise OSError(ctypes.get_last_error(), f"SetProcessAffinityMask({pid}) failed")
        finally:
            kernel32.CloseHandle(handle)
    elif sys.platform.startswith("linux") and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(pid, set(cpu_ids))
    else:
        raise RuntimeError(f"process affinity control is unsupported on {sys.platform}")
    return {"schema": "litenn.process_affinity.v1", "requested_cpu_ids": cpu_ids, "applied": True}


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


def process_power_policy_stable(process: dict[str, object]) -> bool:
    before = process.get("power_policy_before")
    after = process.get("power_policy_after")
    return isinstance(before, dict) and isinstance(after, dict) and before == after


def paired_power_policy_stable(first: dict[str, object], second: dict[str, object]) -> bool:
    if not process_power_policy_stable(first) or not process_power_policy_stable(second):
        return False
    return first.get("power_policy_before") == second.get("power_policy_before")


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
    process_cpu_ids: list[int] | None = None,
) -> tuple[dict[str, object], str, str]:
    artifact_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_stdout = artifact_prefix.with_suffix(".raw.stdout.txt")
    raw_stderr = artifact_prefix.with_suffix(".raw.stderr.txt")
    started = time.perf_counter()
    policy_before = power_policy()
    samples: list[dict[str, object]] = []
    frequency_monitor = create_frequency_monitor()
    resource_document: dict[str, object] | None = None
    resource_sampler: ProcessMemorySampler | None = None
    final_sample: dict[str, object] | None = None
    affinity: dict[str, object] = {
        "schema": "litenn.process_affinity.v1",
        "requested_cpu_ids": process_cpu_ids or [],
        "applied": False,
    }
    try:
        with raw_stdout.open("wb") as stdout_stream, raw_stderr.open("wb") as stderr_stream:
            process = subprocess.Popen(command, stdout=stdout_stream, stderr=stderr_stream)
            try:
                affinity = apply_process_affinity(process.pid, process_cpu_ids)
            except BaseException:
                process.terminate()
                process.wait(timeout=10.0)
                raise
            resource_sampler = ProcessMemorySampler(
                process.pid, max(10, round(monitor_interval_seconds * 1000.0)), lambda: "paired_runtime"
            )
            resource_sampler.start()
            initial_sample = frequency_monitor.sample()
            if initial_sample is not None:
                samples.append(initial_sample)
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
        if resource_sampler is not None:
            resource_document = resource_sampler.stop()
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
            "affinity": affinity,
            "telemetry": {
                "schema": "litenn.paired_process_monitor.v1",
                "sample_interval_seconds": monitor_interval_seconds,
                "frequency_samples": samples,
                "runtime_process_resources": resource_document,
            },
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


def _samples_in_interval(
    samples: list[dict[str, object]], start_ns: int, end_ns: int
) -> list[dict[str, object]]:
    result = []
    for sample in samples:
        timestamp = sample.get("monotonic_ns")
        if isinstance(timestamp, int) and not isinstance(timestamp, bool) and start_ns <= timestamp <= end_ns:
            result.append(sample)
    return result


def _numeric_values(samples: list[dict[str, object]], name: str) -> list[float]:
    return [
        float(value)
        for sample in samples
        if isinstance((value := sample.get(name)), (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]


def summarize_window_telemetry(
    window: dict[str, object],
    frequency_samples: list[dict[str, object]],
    resource_samples: list[dict[str, object]],
) -> dict[str, object]:
    start_ns = int(window["decodeStartMonotonicNs"])
    end_ns = int(window["decodeEndMonotonicNs"])
    host = _samples_in_interval(frequency_samples, start_ns, end_ns)
    process = _samples_in_interval(resource_samples, start_ns, end_ns)
    weighted_frequency = _numeric_values(host, "weighted_actual_mhz")
    host_utility = _numeric_values(host, "host_utility_percent_mean")
    host_load = _numeric_values(host, "host_load_1m")
    active_cpu_union = sorted(
        {
            str(cpu)
            for sample in host
            for cpu in sample.get("active_logical_cpus", [])  # type: ignore[union-attr]
        }
    )
    allowed_sets = [
        {int(cpu) for cpu in value}
        for sample in process
        if isinstance((value := sample.get("allowed_cpu_ids")), list)
        and all(isinstance(cpu, int) and not isinstance(cpu, bool) for cpu in value)
    ]
    cpu_user = _numeric_values(process, "cpu_user_ms")
    cpu_system = _numeric_values(process, "cpu_system_ms")
    cpu_delta_ms = None
    cpu_utilization_percent = None
    if len(process) >= 2 and len(cpu_user) >= 2 and len(cpu_system) >= 2:
        cpu_delta_ms = max(0.0, cpu_user[-1] - cpu_user[0]) + max(0.0, cpu_system[-1] - cpu_system[0])
        observed_ms = (int(process[-1]["monotonic_ns"]) - int(process[0]["monotonic_ns"])) / 1_000_000.0
        if observed_ms > 0.0:
            cpu_utilization_percent = cpu_delta_ms * 100.0 / observed_ms

    def range_summary(name: str) -> dict[str, float | None]:
        values = _numeric_values(process, name)
        return {"minimum": min(values) if values else None, "maximum": max(values) if values else None}

    return {
        "schema": "litenn.decode_window_telemetry.v1",
        "available": bool(host) and len(process) >= 2,
        "hostSampleCount": len(host),
        "processSampleCount": len(process),
        "weightedActualMHz": {
            "minimum": min(weighted_frequency) if weighted_frequency else None,
            "median": statistics.median(weighted_frequency) if weighted_frequency else None,
            "maximum": max(weighted_frequency) if weighted_frequency else None,
        },
        "hostUtilityPercentMean": {
            "minimum": min(host_utility) if host_utility else None,
            "median": statistics.median(host_utility) if host_utility else None,
            "maximum": max(host_utility) if host_utility else None,
        },
        "hostLoad1m": {
            "minimum": min(host_load) if host_load else None,
            "median": statistics.median(host_load) if host_load else None,
            "maximum": max(host_load) if host_load else None,
        },
        "activeLogicalCPUUnion": active_cpu_union,
        "processCPUTimeDeltaMs": cpu_delta_ms,
        "processCPUUtilizationPercent": cpu_utilization_percent,
        "rssBytes": range_summary("rss_bytes"),
        "privateBytes": range_summary("private_bytes"),
        "allowedCPUUnion": sorted(set().union(*allowed_sets)) if allowed_sets else [],
        "allowedCPUIntersection": sorted(set.intersection(*allowed_sets)) if allowed_sets else [],
    }


def assess_window_host_stability(
    report: dict[str, object], maximum_activity_excursion_ratio: float, minimum_frequency_ratio: float
) -> dict[str, object]:
    measured = [window for window in report["windows"] if window["phase"] == "measured"]  # type: ignore[index]
    activity_values: list[float] = []
    activity_metric: str | None = None
    frequency_values: list[float] = []
    for window in measured:
        telemetry = window["telemetry"]
        utility = telemetry["hostUtilityPercentMean"]["median"]
        load = telemetry["hostLoad1m"]["median"]
        frequency = telemetry["weightedActualMHz"]["median"]
        if isinstance(utility, (int, float)) and not isinstance(utility, bool):
            activity_metric = "hostUtilityPercentMean"
            activity_values.append(float(utility))
        elif isinstance(load, (int, float)) and not isinstance(load, bool):
            activity_metric = "hostLoad1m"
            activity_values.append(float(load))
        if isinstance(frequency, (int, float)) and not isinstance(frequency, bool):
            frequency_values.append(float(frequency))

    activity_available = len(activity_values) == len(measured) and bool(activity_values)
    frequency_available = len(frequency_values) == len(measured) and bool(frequency_values)
    activity_baseline = statistics.median(activity_values) if activity_available else None
    maximum_activity = max(activity_values) if activity_available else None
    activity_excursion_ratio = (
        maximum_activity / activity_baseline
        if activity_baseline is not None and activity_baseline > 0.0 and maximum_activity is not None
        else 1.0 if activity_available else None
    )
    frequency_baseline = statistics.median(frequency_values) if frequency_available else None
    minimum_frequency = min(frequency_values) if frequency_available else None
    frequency_ratio = (
        minimum_frequency / frequency_baseline
        if frequency_baseline is not None and frequency_baseline > 0.0 and minimum_frequency is not None
        else None
    )
    activity_passed = (
        activity_excursion_ratio is None or activity_excursion_ratio <= maximum_activity_excursion_ratio
    )
    frequency_passed = frequency_ratio is None or frequency_ratio >= minimum_frequency_ratio
    return {
        "schema": "litenn.decode_window_host_stability.v1",
        "available": activity_available and frequency_available,
        "activity_metric": activity_metric,
        "activity_baseline": activity_baseline,
        "maximum_activity": maximum_activity,
        "activity_excursion_ratio": activity_excursion_ratio,
        "maximum_activity_excursion_ratio": maximum_activity_excursion_ratio,
        "frequency_baseline_mhz": frequency_baseline,
        "minimum_frequency_mhz": minimum_frequency,
        "minimum_frequency_ratio_observed": frequency_ratio,
        "minimum_frequency_ratio_required": minimum_frequency_ratio,
        "activity_passed": activity_passed,
        "frequency_passed": frequency_passed,
        "passed": activity_passed and frequency_passed,
    }


def assess_window_process_affinity(
    report: dict[str, object], requested_cpu_ids: list[int] | None
) -> dict[str, object]:
    requested = requested_cpu_ids or []
    if not requested:
        return {
            "schema": "litenn.decode_window_process_affinity.v1",
            "requested_cpu_ids": [],
            "available": True,
            "passed": True,
        }
    observed = [
        window["telemetry"]["allowedCPUIntersection"]  # type: ignore[index]
        for window in report["windows"]  # type: ignore[index]
        if window["phase"] == "measured"  # type: ignore[index]
    ]
    available = bool(observed) and all(bool(value) for value in observed)
    passed = available and all(value == requested for value in observed)
    return {
        "schema": "litenn.decode_window_process_affinity.v1",
        "requested_cpu_ids": requested,
        "observed_window_intersections": observed,
        "available": available,
        "passed": passed,
    }


def load_litenn_runtime_resource_document(report_path: Path) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    steps = report.get("steps")
    if not isinstance(steps, list):
        raise RuntimeError("LiteNN smoke report is missing process steps")
    decode_steps = [step for step in steps if isinstance(step, dict) and step.get("name") == "litenn_decode_token_ids"]
    if len(decode_steps) != 1 or not isinstance(decode_steps[0].get("memory"), str):
        raise RuntimeError("LiteNN smoke report is missing decode-process telemetry")
    recorded_path = Path(str(decode_steps[0]["memory"]))
    candidates = [recorded_path] if recorded_path.is_absolute() else [repo_root() / recorded_path, report_path.parent / recorded_path.name]
    memory_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if memory_path is None:
        raise RuntimeError("LiteNN decode-process telemetry file does not exist")
    document = json.loads(memory_path.read_text(encoding="utf-8"))
    if document.get("schema") != "litenn.process_memory.v2" or not isinstance(document.get("samples"), list):
        raise RuntimeError("LiteNN decode-process telemetry has an unsupported schema")
    return document


def load_in_process_decode_report(
    path: Path,
    *,
    producer: str,
    warmup_windows: int,
    measured_windows: int,
    prompt_tokens: int,
    decode_tokens: int,
    frequency_samples: list[dict[str, object]] | None = None,
    resource_samples: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "litenn.in_process_decode_windows.v2",
        "producer": producer,
        "warmupWindows": warmup_windows,
        "measuredWindows": measured_windows,
        "promptTokens": prompt_tokens,
        "decodeTokensPerWindow": decode_tokens,
    }
    for name, value in expected.items():
        if document.get(name) != value:
            raise RuntimeError(f"in-process decode report {name} mismatch: expected {value!r}, got {document.get(name)!r}")
    windows = document.get("windows")
    if not isinstance(windows, list) or len(windows) != warmup_windows + measured_windows:
        raise RuntimeError("in-process decode report has an invalid window count")
    measured_throughputs: list[float] = []
    measured_latencies: list[float] = []
    phase_counts = {"warmup": 0, "measured": 0}
    previous_window_end = 0
    for window in windows:
        if not isinstance(window, dict) or window.get("phase") not in phase_counts:
            raise RuntimeError("in-process decode report contains an invalid window")
        phase = str(window["phase"])
        expected_index = phase_counts[phase]
        if window.get("index") != expected_index or window.get("decodeTokens") != decode_tokens:
            raise RuntimeError("in-process decode report window order or token count is invalid")
        phase_counts[phase] += 1
        timestamps = []
        for name in (
            "windowStartMonotonicNs",
            "decodeStartMonotonicNs",
            "decodeEndMonotonicNs",
            "windowEndMonotonicNs",
        ):
            value = window.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise RuntimeError(f"in-process decode report contains invalid {name}")
            timestamps.append(value)
        if timestamps != sorted(timestamps) or timestamps[1] == timestamps[2]:
            raise RuntimeError("in-process decode report contains invalid window timestamp ordering")
        if previous_window_end > timestamps[0]:
            raise RuntimeError("in-process decode report contains overlapping windows")
        previous_window_end = timestamps[-1]
        for name in ("stateResetMs", "prefillMs", "decodeWallMs", "moduleRunMs", "msPerToken", "tokensPerSecond"):
            value = window.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise RuntimeError(f"in-process decode report contains invalid {name}")
            if name in ("decodeWallMs", "moduleRunMs", "msPerToken", "tokensPerSecond") and value <= 0.0:
                raise RuntimeError(f"in-process decode report contains non-positive {name}")
        timestamp_decode_ms = (timestamps[2] - timestamps[1]) / 1_000_000.0
        if not math.isclose(
            timestamp_decode_ms,
            float(window["decodeWallMs"]),
            rel_tol=1e-6,
            abs_tol=0.05,
        ):
            raise RuntimeError("in-process decode report timestamp duration does not match decodeWallMs")
        if phase == "measured":
            measured_throughputs.append(float(window["tokensPerSecond"]))
            measured_latencies.append(float(window["msPerToken"]))
    if phase_counts != {"warmup": warmup_windows, "measured": measured_windows}:
        raise RuntimeError("in-process decode report phase coverage is invalid")
    throughput = series_statistics(measured_throughputs)
    latency = series_statistics(measured_latencies)
    summary = document.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("in-process decode report is missing its summary")
    reported_median = summary.get("tokensPerSecondMedian")
    reported_cv = summary.get("tokensPerSecondCVPercent")
    if not isinstance(reported_median, (int, float)) or not math.isclose(
        float(reported_median), float(throughput["median"]), rel_tol=1e-9, abs_tol=1e-9
    ):
        raise RuntimeError("in-process decode report throughput median does not match its windows")
    if not isinstance(reported_cv, (int, float)) or not math.isclose(
        float(reported_cv), float(throughput["coefficient_of_variation_percent"]), rel_tol=1e-9, abs_tol=1e-9
    ):
        raise RuntimeError("in-process decode report throughput CV does not match its windows")
    measured_windows_raw = [window for window in windows if window["phase"] == "measured"]
    measured_values = [float(window["tokensPerSecond"]) for window in measured_windows_raw]
    adjacent_deltas = [
        (current / previous - 1.0) * 100.0
        for previous, current in zip(measured_values, measured_values[1:])
        if previous > 0.0
    ]
    monotonic_direction = "flat"
    if any(current != previous for previous, current in zip(measured_values, measured_values[1:])):
        if all(current >= previous for previous, current in zip(measured_values, measured_values[1:])):
            monotonic_direction = "increasing"
        elif all(current <= previous for previous, current in zip(measured_values, measured_values[1:])):
            monotonic_direction = "decreasing"
        else:
            monotonic_direction = "mixed"
    temporal_drift = {
        "first_to_last_percent": (measured_values[-1] / measured_values[0] - 1.0) * 100.0,
        "maximum_absolute_adjacent_percent": max((abs(value) for value in adjacent_deltas), default=0.0),
        "direction": monotonic_direction,
    }
    host_samples = frequency_samples or []
    process_samples = resource_samples or []
    for window in windows:
        window["telemetry"] = summarize_window_telemetry(window, host_samples, process_samples)
    measured_coverage = [bool(window["telemetry"]["available"]) for window in measured_windows_raw]
    document["telemetry"] = {
        "schema": "litenn.decode_window_telemetry.v1",
        "hostRawSampleCount": len(host_samples),
        "processRawSampleCount": len(process_samples),
        "measuredWindowsCovered": sum(measured_coverage),
        "allMeasuredWindowsCovered": all(measured_coverage),
    }
    document["validated_statistics"] = {
        "tokens_per_second": throughput,
        "ms_per_token": latency,
        "temporal_drift": temporal_drift,
    }
    return document


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
    prompt_token_ids: list[int] | None = None,
    benchmark_report_path: Path | None = None,
) -> list[str]:
    command = [
        args.python,
        str(repo_root() / "example" / "gguf" / "qwen_smoke.py"),
        "--model",
        str(model),
        "--litenn",
        str(litenn),
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
    if prompt_token_ids is None:
        command.extend(("--llamacpp-tokenizer-tool", str(tokenizer), "--prompt", args.prompt))
    else:
        command.extend(("--token-ids", ",".join(str(token_id) for token_id in prompt_token_ids)))
    if args.litenn_max_cache_length is not None:
        command.extend(("--max-cache-length", str(args.litenn_max_cache_length)))
    if args.litenn_affinity != "default":
        command.extend(("--cpu-aot-affinity", args.litenn_affinity))
    if forced_generated_token_ids is not None:
        command.extend(
            ("--forced-generated-token-ids", ",".join(str(token_id) for token_id in forced_generated_token_ids))
        )
    in_process_windows = int(getattr(args, "in_process_windows", 0))
    if in_process_windows:
        if benchmark_report_path is None:
            raise ValueError("in-process LiteNN command requires a benchmark report path")
        command.extend(
            (
                "--benchmark-warmup-windows",
                str(getattr(args, "in_process_warmup_windows", 1)),
                "--benchmark-windows",
                str(in_process_windows),
                "--benchmark-window-report",
                str(benchmark_report_path),
            )
        )
    return command


def build_llama_in_process_command(
    args: argparse.Namespace,
    model: Path,
    adapter: Path,
    prompt_token_ids: list[int],
    generated_token_ids: list[int],
    report_path: Path,
) -> list[str]:
    return [
        str(adapter),
        "benchmark-fixed-decode",
        str(model),
        ",".join(str(token_id) for token_id in prompt_token_ids),
        ",".join(str(token_id) for token_id in generated_token_ids),
        str(args.in_process_warmup_windows),
        str(args.in_process_windows),
        str(args.context_size),
        str(args.llama_threads),
        str(report_path),
    ]


def capture_chat_prompt_token_ids(
    args: argparse.Namespace,
    model: Path,
    adapter: Path,
    output_dir: Path,
    replacements: dict[str, str],
) -> tuple[list[int], dict[str, object]]:
    capture_dir = output_dir / "fixed_replay_prompt"
    capture_dir.mkdir(parents=True, exist_ok=True)
    token_output = capture_dir / "prompt_tokens.json"
    process, _, _ = run_monitored(
        [str(adapter), "tokenize-chat-template", str(model), args.prompt, str(token_output)],
        capture_dir / "tokenize_chat_template",
        replacements,
        args.monitor_interval,
    )
    if process["returncode"] != 0:
        raise RuntimeError("fixed-trajectory chat prompt tokenization failed")
    token_ids = load_tokenizer_token_ids(token_output)
    return token_ids, {"process": process, "tokens": token_ids_identity(token_ids)}


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
        f"- Host admission maximum activity: `"
        f"{document['configuration']['host_admission_max_activity_percent']}`",  # type: ignore[index]
        f"- Runtime/pair cooldown: `{document['configuration']['cooldown_between_runtimes_seconds']}/"
        f"{document['configuration']['cooldown_between_pairs_seconds']} s`",  # type: ignore[index]
        f"- Shared process CPU set: `{document['configuration']['process_cpu_set']}`",  # type: ignore[index]
        f"- Trajectory mode: `{'fixed reference replay' if document['configuration']['fixed_token_replay'] else 'natural greedy'}`",  # type: ignore[index]
        "",
        "| Pair | Order | llama ms/token | llama t/s | LiteNN ms/token | LiteNN t/s | LiteNN delta | "
        "llama MHz | LiteNN MHz | Policy stable | Trajectory parity |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
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
            f"{frequency_value(litenn_frequency)} | {pair['power_policy_stable']} | "
            f"{pair['trajectory_match']} |"
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
            f"- Power-policy stability: `{gate['power_policy_stability']}`",  # type: ignore[index]
            f"- Process-level variance gate: `{gate['process_variance']}`",  # type: ignore[index]
            f"- In-process window variance gate: `{gate['in_process_variance']}`",  # type: ignore[index]
            f"- Window telemetry coverage gate: `{gate['window_telemetry']}`",  # type: ignore[index]
            f"- Host-stability gate: `{gate['host_stability']}`",  # type: ignore[index]
            f"- Process-affinity gate: `{gate['process_affinity']}`",  # type: ignore[index]
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
    if document["configuration"]["in_process_windows"]:  # type: ignore[index]
        window_cv = summary["in_process_window_cv_percent"]
        temporal_drift = summary["in_process_temporal_drift_percent"]
        lines.extend(
            [
                f"- llama.cpp median within-process CV: `{window_cv['llama_cpp']['median']:.2f}%`",  # type: ignore[index]
                f"- LiteNN median within-process CV: `{window_cv['litenn']['median']:.2f}%`",  # type: ignore[index]
                f"- llama.cpp median first-to-last window drift: `{temporal_drift['llama_cpp']['median']:+.2f}%`",  # type: ignore[index]
                f"- LiteNN median first-to-last window drift: `{temporal_drift['litenn']['median']:+.2f}%`",  # type: ignore[index]
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
    parser.add_argument(
        "--litenn-max-cache-length",
        type=positive_int,
        help="Use this explicit LiteNN KV-cache capacity so paired runs can reuse a shape-stable AOT artifact",
    )
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
    parser.add_argument(
        "--process-cpu-set",
        type=cpu_set,
        help="Apply the same OS process-affinity CPU set (for example 0-7) to both timed runtimes",
    )
    parser.add_argument("--monitor-interval", default=0.25, type=positive_float)
    parser.add_argument(
        "--host-admission-max-activity-percent",
        type=positive_float,
        help="Wait before each runtime until consecutive host activity samples do not exceed this value",
    )
    parser.add_argument("--host-admission-consecutive-samples", default=3, type=positive_int)
    parser.add_argument("--host-admission-warmup-samples", default=2, type=non_negative_int)
    parser.add_argument("--host-admission-timeout-seconds", default=60.0, type=positive_float)
    parser.add_argument("--cooldown-between-runtimes-seconds", default=0.0, type=non_negative_float)
    parser.add_argument("--cooldown-between-pairs-seconds", default=0.0, type=non_negative_float)
    parser.add_argument("--window-host-activity-excursion-ratio", default=2.0, type=positive_float)
    parser.add_argument("--window-minimum-frequency-ratio", default=0.95, type=positive_float)
    parser.add_argument(
        "--require-host-stability",
        action="store_true",
        help="Require pre-runtime admission and complete window activity/frequency evidence",
    )
    parser.add_argument("--variance-threshold-percent", default=3.0, type=positive_float)
    parser.add_argument("--require-variance-gate", action="store_true")
    parser.add_argument(
        "--in-process-warmup-windows",
        default=1,
        type=non_negative_int,
        help="Discard this many fixed-trajectory windows after mapping each runtime",
    )
    parser.add_argument(
        "--in-process-windows",
        default=0,
        type=non_negative_int,
        help="Measure this many fixed-trajectory windows inside each runtime process",
    )
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
    if args.in_process_windows and not args.fixed_token_replay:
        raise SystemExit("--in-process-windows requires --fixed-token-replay")
    if args.in_process_windows and args.predict < 2:
        raise SystemExit("--in-process-windows requires --predict of at least two")
    if args.require_host_stability and not args.in_process_windows:
        raise SystemExit("--require-host-stability requires --in-process-windows")
    if args.require_host_stability and args.host_admission_max_activity_percent is None:
        raise SystemExit("--require-host-stability requires --host-admission-max-activity-percent")
    if args.window_minimum_frequency_ratio > 1.0:
        raise SystemExit("--window-minimum-frequency-ratio must not exceed 1")
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
            "litenn_max_cache_length": args.litenn_max_cache_length,
            "litenn_affinity": args.litenn_affinity,
            "litenn_worker_wait": args.litenn_worker_wait,
            "llvm_opt_level": args.llvm_opt_level,
            "llama_threads": args.llama_threads,
            "llama_cpu_mask": args.llama_cpu_mask,
            "llama_cpu_strict": args.llama_cpu_strict,
            "llama_poll": args.llama_poll,
            "llama_priority": args.llama_priority,
            "process_cpu_set": args.process_cpu_set,
            "monitor_interval_seconds": args.monitor_interval,
            "host_admission_max_activity_percent": args.host_admission_max_activity_percent,
            "host_admission_consecutive_samples": args.host_admission_consecutive_samples,
            "host_admission_warmup_samples": args.host_admission_warmup_samples,
            "host_admission_timeout_seconds": args.host_admission_timeout_seconds,
            "cooldown_between_runtimes_seconds": args.cooldown_between_runtimes_seconds,
            "cooldown_between_pairs_seconds": args.cooldown_between_pairs_seconds,
            "window_host_activity_excursion_ratio": args.window_host_activity_excursion_ratio,
            "window_minimum_frequency_ratio": args.window_minimum_frequency_ratio,
            "require_host_stability": args.require_host_stability,
            "variance_threshold_percent": args.variance_threshold_percent,
            "fixed_token_replay": args.fixed_token_replay,
            "in_process_warmup_windows": args.in_process_warmup_windows,
            "in_process_windows": args.in_process_windows,
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
    prompt_token_ids: list[int] | None = None
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
    if args.in_process_windows:
        print("[paired decode] capturing fixed chat prompt tokens", file=sys.stderr, flush=True)
        try:
            prompt_token_ids, prompt_capture = capture_chat_prompt_token_ids(
                args, model, tokenizer, output_dir, replacements
            )
        except RuntimeError as error:
            document["status"] = "failed"
            document["failure"] = str(error)
            checkpoint()
            raise SystemExit(str(error)) from error
        fixed_reference = document["fixed_replay_reference"]
        if len(prompt_token_ids) != int(fixed_reference["prompt_tokens"]):  # type: ignore[index]
            document["status"] = "failed"
            document["failure"] = "fixed chat prompt token count differs from llama.cpp completion"
            checkpoint()
            raise SystemExit(document["failure"])
        document["fixed_replay_prompt"] = prompt_capture
        print("[paired decode] fixed chat prompt tokens ready", file=sys.stderr, flush=True)
        checkpoint()
    pairs_document = document["pairs"]
    assert isinstance(pairs_document, list)
    for repetition in range(1, args.repetitions + 1):
        order = ["llama_cpp", "litenn"] if repetition % 2 == 1 else ["litenn", "llama_cpp"]
        pair: dict[str, object] = {
            "repetition": repetition,
            "order": order,
            "cooldowns": [],
            "host_admissions": {},
        }
        pairs_document.append(pair)
        pair_dir = output_dir / f"pair_{repetition:02d}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        if repetition > 1:
            pair["cooldowns"].append(  # type: ignore[union-attr]
                cooldown_record(args.cooldown_between_pairs_seconds, "between_pairs")
            )
        for runtime_index, runtime in enumerate(order):
            if runtime_index > 0:
                pair["cooldowns"].append(  # type: ignore[union-attr]
                    cooldown_record(args.cooldown_between_runtimes_seconds, "between_runtimes")
                )
            if args.host_admission_max_activity_percent is not None:
                print(
                    f"[paired decode] pair {repetition}/{args.repetitions} {runtime} waiting for host admission",
                    file=sys.stderr,
                    flush=True,
                )
            admission = wait_for_host_admission(
                args.host_admission_max_activity_percent,
                args.host_admission_consecutive_samples,
                args.host_admission_warmup_samples,
                args.host_admission_timeout_seconds,
                args.monitor_interval,
            )
            pair["host_admissions"][runtime] = admission  # type: ignore[index]
            checkpoint()
            if not bool(admission["passed"]):
                document["status"] = "failed"
                document["failure"] = (
                    f"host admission timed out before pair {repetition} {runtime}; "
                    "raw admission samples were retained"
                )
                checkpoint()
                raise SystemExit(document["failure"])
            print(
                f"[paired decode] pair {repetition}/{args.repetitions} {runtime} starting",
                file=sys.stderr,
                flush=True,
            )
            if runtime == "llama_cpp":
                in_process_report_path = pair_dir / "llama_cpp_in_process_windows.json"
                if args.in_process_windows:
                    assert prompt_token_ids is not None
                    assert forced_generated_token_ids is not None
                    command = build_llama_in_process_command(
                        args,
                        model,
                        tokenizer,
                        prompt_token_ids,
                        forced_generated_token_ids,
                        in_process_report_path,
                    )
                else:
                    command = build_llama_command(args, model, completion)
                process_record, stdout_text, stderr_text = run_monitored(
                    command,
                    pair_dir / "llama_cpp",
                    replacements,
                    args.monitor_interval,
                    args.process_cpu_set,
                )
                if process_record["returncode"] != 0:
                    document["status"] = "failed"
                    document["failure"] = f"llama.cpp failed in pair {repetition}"
                    checkpoint()
                    raise SystemExit(document["failure"])
                if args.in_process_windows:
                    assert prompt_token_ids is not None
                    fixed_reference = document["fixed_replay_reference"]
                    process_telemetry = process_record["telemetry"]
                    runtime_resources = process_telemetry["runtime_process_resources"]
                    window_report = load_in_process_decode_report(
                        in_process_report_path,
                        producer="llama.cpp",
                        warmup_windows=args.in_process_warmup_windows,
                        measured_windows=args.in_process_windows,
                        prompt_tokens=len(prompt_token_ids),
                        decode_tokens=args.predict - 1,
                        frequency_samples=process_telemetry["frequency_samples"],
                        resource_samples=runtime_resources["samples"],
                    )
                    window_statistics = window_report["validated_statistics"]
                    pair["llama_cpp"] = {
                        "process": process_record,
                        "prompt_tokens": len(prompt_token_ids),
                        "eval_tokens": args.predict - 1,
                        "ms_per_token": window_statistics["ms_per_token"]["median"],
                        "tokens_per_second": window_statistics["tokens_per_second"]["median"],
                        "text": fixed_reference["text"],
                        "in_process": window_report,
                    }
                else:
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
                in_process_report_path = pair_dir / "litenn_in_process_windows.json"
                command = build_litenn_command(
                    args,
                    model,
                    litenn,
                    tokenizer,
                    workdir,
                    cache_dir,
                    forced_generated_token_ids,
                    prompt_token_ids if args.in_process_windows else None,
                    in_process_report_path if args.in_process_windows else None,
                )
                process_record, _, _ = run_monitored(
                    command,
                    pair_dir / "litenn",
                    replacements,
                    args.monitor_interval,
                    args.process_cpu_set,
                )
                if process_record["returncode"] != 0:
                    document["status"] = "failed"
                    document["failure"] = f"LiteNN failed in pair {repetition}"
                    checkpoint()
                    raise SystemExit(document["failure"])
                report_path = workdir / "qwen_smoke_report.json"
                base = litenn_row(report_path)
                steady = litenn_steady_generation_row(report_path, base) if not args.in_process_windows else None
                token_output = workdir / "litenn_decode_tokens.txt"
                replay = parse_forced_replay_metrics(token_output)
                generated_token_ids = load_litenn_generated_token_ids(token_output, int(base["promptTokens"]))
                if args.in_process_windows:
                    assert prompt_token_ids is not None
                    fixed_reference = document["fixed_replay_reference"]
                    process_telemetry = process_record["telemetry"]
                    process_telemetry["launcher_process_resources"] = process_telemetry[
                        "runtime_process_resources"
                    ]
                    runtime_resources = load_litenn_runtime_resource_document(report_path)
                    process_telemetry["runtime_process_resources"] = runtime_resources
                    window_report = load_in_process_decode_report(
                        in_process_report_path,
                        producer="LiteNN",
                        warmup_windows=args.in_process_warmup_windows,
                        measured_windows=args.in_process_windows,
                        prompt_tokens=len(prompt_token_ids),
                        decode_tokens=args.predict - 1,
                        frequency_samples=process_telemetry["frequency_samples"],
                        resource_samples=runtime_resources["samples"],
                    )
                    window_statistics = window_report["validated_statistics"]
                    ms_per_token = window_statistics["ms_per_token"]["median"]
                    tokens_per_second = window_statistics["tokens_per_second"]["median"]
                    eval_tokens = args.predict - 1
                    generated_text_identity = fixed_reference["text"]
                else:
                    assert steady is not None
                    ms_per_token = steady["msPerToken"]
                    tokens_per_second = steady["tokensPerSecond"]
                    eval_tokens = steady["tokens"]
                    generated_text = (workdir / "generated_text.bin").read_text(encoding="utf-8").strip()
                    generated_text_identity = text_identity(generated_text)
                pair["litenn"] = {
                    "process": process_record,
                    "prompt_tokens": base["promptTokens"],
                    "eval_tokens": eval_tokens,
                    "ms_per_token": ms_per_token,
                    "tokens_per_second": tokens_per_second,
                    "full_generation_ms_per_token": base["msPerToken"],
                    "full_generation_tokens_per_second": base["tokensPerSecond"],
                    "fallback_used": base["fallbackUsed"],
                    "fallback_count": base["fallbackCount"],
                    "text": generated_text_identity,
                    "tokens": token_ids_identity(generated_token_ids),
                    "fixed_replay": replay,
                }
                if args.in_process_windows:
                    pair["litenn"]["in_process"] = window_report
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
        if args.in_process_windows:
            llama_window_cv = float(
                llama["in_process"]["validated_statistics"]["tokens_per_second"][  # type: ignore[index]
                    "coefficient_of_variation_percent"
                ]
            )
            litenn_window_cv = float(
                litenn_result["in_process"]["validated_statistics"]["tokens_per_second"][  # type: ignore[index]
                    "coefficient_of_variation_percent"
                ]
            )
            pair["in_process_variance"] = {
                "llama_cpp_cv_percent": llama_window_cv,
                "litenn_cv_percent": litenn_window_cv,
                "passed": (
                    llama_window_cv <= args.variance_threshold_percent
                    and litenn_window_cv <= args.variance_threshold_percent
                ),
            }
            llama_telemetry_covered = bool(  # type: ignore[index]
                llama["in_process"]["telemetry"]["allMeasuredWindowsCovered"]
            )
            litenn_telemetry_covered = bool(  # type: ignore[index]
                litenn_result["in_process"]["telemetry"]["allMeasuredWindowsCovered"]
            )
            pair["window_telemetry"] = {
                "llama_cpp_covered": llama_telemetry_covered,
                "litenn_covered": litenn_telemetry_covered,
                "passed": llama_telemetry_covered and litenn_telemetry_covered,
            }
            llama_host_stability = assess_window_host_stability(
                llama["in_process"],  # type: ignore[index]
                args.window_host_activity_excursion_ratio,
                args.window_minimum_frequency_ratio,
            )
            litenn_host_stability = assess_window_host_stability(
                litenn_result["in_process"],  # type: ignore[index]
                args.window_host_activity_excursion_ratio,
                args.window_minimum_frequency_ratio,
            )
            pair["host_stability"] = {
                "llama_cpp": llama_host_stability,
                "litenn": litenn_host_stability,
                "available": bool(llama_host_stability["available"] and litenn_host_stability["available"]),
                "passed": bool(llama_host_stability["passed"] and litenn_host_stability["passed"]),
            }
            llama_affinity = assess_window_process_affinity(
                llama["in_process"], args.process_cpu_set  # type: ignore[index]
            )
            litenn_affinity = assess_window_process_affinity(
                litenn_result["in_process"], args.process_cpu_set  # type: ignore[index]
            )
            pair["process_affinity"] = {
                "llama_cpp": llama_affinity,
                "litenn": litenn_affinity,
                "passed": bool(llama_affinity["passed"] and litenn_affinity["passed"]),
            }
        pair["power_policy_stable"] = paired_power_policy_stable(  # type: ignore[index]
            llama["process"], litenn_result["process"]  # type: ignore[index]
        )
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
        elif not pair["power_policy_stable"]:
            document["status"] = "failed"
            document["failure"] = f"power policy changed within or between runtimes in pair {repetition}"
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
    process_variance_passed = (
        float(summary["llama_cpp"]["coefficient_of_variation_percent"]) <= args.variance_threshold_percent
        and float(summary["litenn"]["coefficient_of_variation_percent"]) <= args.variance_threshold_percent
    )
    if args.in_process_windows:
        summary["in_process_window_cv_percent"] = {
            "llama_cpp": series_statistics(
                [float(pair["in_process_variance"]["llama_cpp_cv_percent"]) for pair in pairs]  # type: ignore[index]
            ),
            "litenn": series_statistics(
                [float(pair["in_process_variance"]["litenn_cv_percent"]) for pair in pairs]  # type: ignore[index]
            ),
        }
        summary["in_process_temporal_drift_percent"] = {
            "llama_cpp": series_statistics(
                [
                    float(pair["llama_cpp"]["in_process"]["validated_statistics"]["temporal_drift"]["first_to_last_percent"])  # type: ignore[index]
                    for pair in pairs
                ]
            ),
            "litenn": series_statistics(
                [
                    float(pair["litenn"]["in_process"]["validated_statistics"]["temporal_drift"]["first_to_last_percent"])  # type: ignore[index]
                    for pair in pairs
                ]
            ),
        }
    in_process_variance_passed = not args.in_process_windows or all(
        bool(pair["in_process_variance"]["passed"]) for pair in pairs  # type: ignore[index]
    )
    window_telemetry_passed = not args.in_process_windows or all(
        bool(pair["window_telemetry"]["passed"]) for pair in pairs  # type: ignore[index]
    )
    host_stability_passed = not args.in_process_windows or all(
        bool(pair["host_stability"]["passed"])  # type: ignore[index]
        and (
            not args.require_host_stability
            or (
                bool(pair["host_stability"]["available"])  # type: ignore[index]
                and all(
                    bool(admission["enabled"] and admission["passed"])
                    for admission in pair["host_admissions"].values()  # type: ignore[union-attr]
                )
            )
        )
        for pair in pairs
    )
    process_affinity_passed = not args.process_cpu_set or all(
        all(
            bool(pair[runtime]["process"]["affinity"]["applied"])  # type: ignore[index]
            for runtime in ("llama_cpp", "litenn")
        )
        and (not args.in_process_windows or bool(pair["process_affinity"]["passed"]))  # type: ignore[index]
        for pair in pairs
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
        "power_policy_stability": all(bool(pair["power_policy_stable"]) for pair in pairs),
        "process_variance": process_variance_passed,
        "in_process_variance": in_process_variance_passed,
        "window_telemetry": window_telemetry_passed,
        "host_stability": host_stability_passed,
        "process_affinity": process_affinity_passed,
        "variance": process_variance_passed and in_process_variance_passed,
    }
    gate["accepted"] = bool(
        gate["trajectory_parity"]
        and gate["no_fallback"]
        and gate["power_policy_stability"]
        and gate["window_telemetry"]
        and gate["host_stability"]
        and gate["process_affinity"]
        and gate["variance"]
    )
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
        f"variance_gate={gate['variance']}",
        file=sys.stderr,
    )
    if not gate["power_policy_stability"]:
        return 2
    return 2 if args.require_variance_gate and not gate["accepted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
