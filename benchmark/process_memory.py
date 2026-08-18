"""Low-overhead cross-platform process memory sampling without optional packages."""

from __future__ import annotations

import ctypes
import os
import platform
import threading
import time
from collections.abc import Callable
from ctypes import wintypes
from pathlib import Path


def _kilobytes(value: str) -> int:
    fields = value.split()
    if not fields:
        return 0
    multiplier = 1024 if len(fields) == 1 or fields[1].lower() == "kb" else 1
    return int(fields[0]) * multiplier


def _parse_cpu_list(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        lower, separator, upper = part.strip().partition("-")
        if not lower:
            continue
        start = int(lower)
        end = int(upper) if separator else start
        if end < start:
            raise ValueError(f"invalid CPU range: {part}")
        result.extend(range(start, end + 1))
    return result


def parse_linux_status(text: str) -> dict[str, object]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        name, separator, value = line.partition(":")
        if separator:
            values[name] = value.strip()

    def read(name: str) -> int | None:
        return _kilobytes(values[name]) if name in values else None

    rss_file = read("RssFile")
    rss_shmem = read("RssShmem")
    mapped_resident = None if rss_file is None and rss_shmem is None else (rss_file or 0) + (rss_shmem or 0)
    allowed_cpu_ids = _parse_cpu_list(values["Cpus_allowed_list"]) if "Cpus_allowed_list" in values else None
    return {
        "rss_bytes": read("VmRSS"),
        "peak_rss_bytes": read("VmHWM"),
        "private_bytes": None,
        "peak_private_bytes": None,
        "virtual_bytes": read("VmSize"),
        "anonymous_resident_bytes": read("RssAnon"),
        "mapped_resident_bytes": mapped_resident,
        "thread_count": int(values["Threads"]) if "Threads" in values else None,
        "allowed_cpu_ids": allowed_cpu_ids,
        "allowed_cpu_count": len(allowed_cpu_ids) if allowed_cpu_ids is not None else None,
    }


class _WindowsMemoryReader:
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    def __init__(self, pid: int) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        self._close_handle = kernel32.CloseHandle
        self._close_handle.argtypes = [wintypes.HANDLE]
        self._close_handle.restype = wintypes.BOOL
        self._handle = kernel32.OpenProcess(
            self.PROCESS_QUERY_INFORMATION | self.PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not self._handle:
            raise OSError(ctypes.get_last_error(), f"OpenProcess({pid}) failed")
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self._get_process_memory_info = psapi.GetProcessMemoryInfo
        self._get_process_memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(self.Counters), wintypes.DWORD]
        self._get_process_memory_info.restype = wintypes.BOOL
        self._get_process_times = kernel32.GetProcessTimes
        self._get_process_times.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        ]
        self._get_process_times.restype = wintypes.BOOL
        self._get_process_affinity_mask = kernel32.GetProcessAffinityMask
        self._get_process_affinity_mask.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._get_process_affinity_mask.restype = wintypes.BOOL

    @staticmethod
    def _filetime_milliseconds(value: wintypes.FILETIME) -> float:
        ticks = (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)
        return ticks / 10_000.0

    def read(self) -> dict[str, object]:
        counters = self.Counters()
        counters.cb = ctypes.sizeof(counters)
        if not self._get_process_memory_info(self._handle, ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not self._get_process_times(
            self._handle, ctypes.byref(creation), ctypes.byref(exit_time), ctypes.byref(kernel), ctypes.byref(user)
        ):
            raise OSError(ctypes.get_last_error(), "GetProcessTimes failed")
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        allowed_cpu_ids = None
        if self._get_process_affinity_mask(
            self._handle, ctypes.byref(process_mask), ctypes.byref(system_mask)
        ):
            allowed_cpu_ids = [index for index in range(ctypes.sizeof(ctypes.c_size_t) * 8) if process_mask.value >> index & 1]
        return {
            "rss_bytes": int(counters.WorkingSetSize),
            "peak_rss_bytes": int(counters.PeakWorkingSetSize),
            "private_bytes": int(counters.PrivateUsage),
            "peak_private_bytes": int(counters.PeakPagefileUsage),
            "virtual_bytes": None,
            "anonymous_resident_bytes": None,
            "mapped_resident_bytes": None,
            "cpu_user_ms": self._filetime_milliseconds(user),
            "cpu_system_ms": self._filetime_milliseconds(kernel),
            "thread_count": None,
            "allowed_cpu_ids": allowed_cpu_ids,
            "allowed_cpu_count": len(allowed_cpu_ids) if allowed_cpu_ids is not None else None,
        }

    def close(self) -> None:
        if self._handle:
            self._close_handle(self._handle)
            self._handle = None


class _LinuxMemoryReader:
    def __init__(self, pid: int) -> None:
        self._pid = pid
        self._status = Path(f"/proc/{pid}/status")
        self._stat = Path(f"/proc/{pid}/stat")
        self._clock_ticks = os.sysconf("SC_CLK_TCK")

    def read(self) -> dict[str, object]:
        result = parse_linux_status(self._status.read_text(encoding="ascii", errors="replace"))
        stat_tail = self._stat.read_text(encoding="ascii", errors="replace").rsplit(")", 1)[1].split()
        result["cpu_user_ms"] = int(stat_tail[11]) * 1000.0 / self._clock_ticks
        result["cpu_system_ms"] = int(stat_tail[12]) * 1000.0 / self._clock_ticks
        if hasattr(os, "sched_getaffinity"):
            allowed_cpu_ids = sorted(os.sched_getaffinity(self._pid))
            result["allowed_cpu_ids"] = allowed_cpu_ids
            result["allowed_cpu_count"] = len(allowed_cpu_ids)
        return result

    def close(self) -> None:
        pass


class _MacOSMemoryReader:
    PROC_PIDTASKINFO = 4

    class TaskInfo(ctypes.Structure):
        _fields_ = [
            ("virtual_size", ctypes.c_uint64),
            ("resident_size", ctypes.c_uint64),
            ("total_user", ctypes.c_uint64),
            ("total_system", ctypes.c_uint64),
            ("threads_user", ctypes.c_uint64),
            ("threads_system", ctypes.c_uint64),
            ("policy", ctypes.c_int32),
            ("faults", ctypes.c_int32),
            ("pageins", ctypes.c_int32),
            ("cow_faults", ctypes.c_int32),
            ("messages_sent", ctypes.c_int32),
            ("messages_received", ctypes.c_int32),
            ("syscalls_mach", ctypes.c_int32),
            ("syscalls_unix", ctypes.c_int32),
            ("csw", ctypes.c_int32),
            ("threadnum", ctypes.c_int32),
            ("numrunning", ctypes.c_int32),
            ("priority", ctypes.c_int32),
        ]

    def __init__(self, pid: int) -> None:
        self._pid = pid
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self._proc_pidinfo = libproc.proc_pidinfo
        self._proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        self._proc_pidinfo.restype = ctypes.c_int

    def read(self) -> dict[str, object]:
        info = self.TaskInfo()
        size = self._proc_pidinfo(
            self._pid, self.PROC_PIDTASKINFO, 0, ctypes.byref(info), ctypes.sizeof(info)
        )
        if size != ctypes.sizeof(info):
            raise OSError(ctypes.get_errno(), f"proc_pidinfo({self._pid}) failed")
        return {
            "rss_bytes": int(info.resident_size),
            "peak_rss_bytes": None,
            "private_bytes": None,
            "peak_private_bytes": None,
            "virtual_bytes": int(info.virtual_size),
            "anonymous_resident_bytes": None,
            "mapped_resident_bytes": None,
            "cpu_user_ms": int(info.total_user) / 1_000_000.0,
            "cpu_system_ms": int(info.total_system) / 1_000_000.0,
            "thread_count": int(info.threadnum),
            "allowed_cpu_ids": None,
            "allowed_cpu_count": None,
        }

    def close(self) -> None:
        pass


def _memory_reader(pid: int):
    system = platform.system()
    if system == "Windows":
        return _WindowsMemoryReader(pid)
    if system == "Linux":
        return _LinuxMemoryReader(pid)
    if system == "Darwin":
        return _MacOSMemoryReader(pid)
    raise RuntimeError(f"process memory sampling is unsupported on {system}")


def summarize_samples(samples: list[dict[str, object]]) -> dict[str, object]:
    metrics = (
        "rss_bytes",
        "peak_rss_bytes",
        "private_bytes",
        "peak_private_bytes",
        "virtual_bytes",
        "anonymous_resident_bytes",
        "mapped_resident_bytes",
    )
    peaks: dict[str, object] = {}
    for metric in metrics:
        candidates = [sample for sample in samples if isinstance(sample.get(metric), int)]
        if not candidates:
            peaks[metric] = None
            continue
        peak = max(candidates, key=lambda sample: int(sample[metric]))
        peaks[metric] = {
            "bytes": int(peak[metric]),
            "elapsed_ms": float(peak["elapsed_ms"]),
            "stage": str(peak["stage"]),
        }
    return {"sample_count": len(samples), "peaks": peaks}


class ProcessMemorySampler:
    def __init__(self, pid: int, interval_ms: int, stage: Callable[[], str]) -> None:
        if interval_ms <= 0:
            raise ValueError("memory sample interval must be positive")
        self._pid = pid
        self._interval_seconds = interval_ms / 1000.0
        self._stage = stage
        self._origin_ns = time.perf_counter_ns()
        self._stop = threading.Event()
        self._reader = _memory_reader(pid)
        self._thread = threading.Thread(target=self._run, name=f"memory-sampler-{pid}", daemon=True)
        self.samples: list[dict[str, object]] = []
        self.error: str | None = None

    def start(self) -> None:
        self._thread.start()

    def _sample(self) -> None:
        timestamp_ns = time.perf_counter_ns()
        values = self._reader.read()
        self.samples.append(
            {
                "monotonic_ns": timestamp_ns,
                "elapsed_ms": (timestamp_ns - self._origin_ns) / 1_000_000.0,
                "stage": self._stage(),
                **values,
            }
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except (OSError, ProcessLookupError) as error:
                self.error = str(error)
                break
            self._stop.wait(self._interval_seconds)

    def stop(self) -> dict[str, object]:
        self._stop.set()
        self._thread.join(timeout=max(5.0, self._interval_seconds * 2.0))
        self._reader.close()
        summary = summarize_samples(self.samples)
        return {
            "schema": "litenn.process_memory.v2",
            "pid": self._pid,
            "platform": platform.system().lower(),
            "interval_ms": self._interval_seconds * 1000.0,
            "error": self.error,
            "samples": self.samples,
            **summary,
        }


def sample_current_process() -> dict[str, object]:
    reader = _memory_reader(os.getpid())
    try:
        return reader.read()
    finally:
        reader.close()
