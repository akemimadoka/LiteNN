#!/usr/bin/env python3
"""Create a LiteNN performance profile bundle.

The bundle wraps existing profiling commands, records a Chrome Trace-compatible
waterfall timeline, captures stdout/stderr, and writes a small manifest/summary.

Examples:
  python311 benchmark/profile_bundle.py \
    --out-dir benchmark/results/profile_bundle_smoke \
    --litenn-profile build-release/benchmark/litenn_profile.exe

  python311 benchmark/profile_bundle.py \
    --out-dir build/qwen_profile_bundle \
    --sensitive-path F:/Models/private.gguf \
    --command python311 example/gguf/qwen_smoke.py --model F:/Models/private.gguf --token-ids 1,2,3 --steps 1
"""

from __future__ import annotations

import argparse
import html
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DEFAULT_TRACE_PID = 1


@dataclass
class StepResult:
    name: str
    command: list[str]
    redacted_command: list[str]
    returncode: int
    start_ns: int
    end_ns: int
    stdout: Path
    stderr: Path
    sampler_outputs: list[Path] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000_000.0


@dataclass(frozen=True)
class CollapsedStack:
    frames: tuple[str, ...]
    samples: int


@dataclass(frozen=True)
class GGUFDecodeStep:
    step: int
    phase: str
    step_ms: float
    generated_tokens: int
    tokens_per_second: float
    input_prep_ms: float | None = None
    module_run_ms: float | None = None
    helper_profile_enabled: bool = False
    helper_total_ms: float | None = None
    module_non_helper_ms: float | None = None
    helper_profile_emit_ms: float | None = None
    logits_output_ms: float | None = None
    sampling_ms: float | None = None
    state_update_ms: float | None = None
    host_overhead_ms: float | None = None


@dataclass(frozen=True)
class GGUFHelperEvent:
    step: int
    helper: str
    detail: str
    operator: str
    role: str
    calls: int
    total_ms: float
    avg_ms: float


@dataclass(frozen=True)
class GGUFNodeEvent:
    step: int
    subgraph: int
    node: int
    op: str
    schema: int
    calls: int
    inclusive_ms: float
    self_ms: float
    helper_ms: float


@dataclass(frozen=True)
class GGUFNodeProfileSummary:
    step: int
    self_ms: float
    helper_ms: float
    calls: int
    nodes: int
    emitted_nodes: int


@dataclass(frozen=True)
class GGUFDecodeAnalysis:
    steps: list[GGUFDecodeStep]
    helpers: list[GGUFHelperEvent]
    nodes: list[GGUFNodeEvent]
    node_summaries: list[GGUFNodeProfileSummary]


@dataclass(frozen=True)
class LogEvidence:
    name: str
    stdout: Path
    stderr: Path


def now_ns() -> int:
    return time.perf_counter_ns()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover_litenn_profile(explicit: Path | None) -> Path | None:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"litenn_profile executable does not exist: {explicit}")
        return explicit

    for name in ("litenn_profile.exe", "litenn_profile"):
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)

    root = repo_root()
    for candidate in (
        root / "build-release" / "benchmark" / "litenn_profile.exe",
        root / "build-release" / "benchmark" / "litenn_profile",
        root / "build" / "benchmark" / "litenn_profile.exe",
        root / "build" / "benchmark" / "litenn_profile",
    ):
        if candidate.exists():
            return candidate
    return None


def normalize_sensitive_paths(paths: Iterable[Path]) -> list[tuple[str, str]]:
    replacements: list[tuple[str, str]] = []
    for path in paths:
        text = str(path)
        if not text:
            continue
        label = "<path:redacted>"
        variants = { text }
        try:
            resolved = str(path.resolve())
        except OSError:
            resolved = text
        variants.add(resolved)
        variants.add(text.replace("\\", "/"))
        variants.add(resolved.replace("\\", "/"))
        variants.add(text.replace("\\", "\\\\"))
        variants.add(resolved.replace("\\", "\\\\"))
        replacements.extend((variant, label) for variant in sorted(variants, key=len, reverse=True) if variant)
    return replacements


def redact_text(text: str, replacements: list[tuple[str, str]]) -> str:
    redacted = text
    for source, target in replacements:
        if source:
            redacted = redacted.replace(source, target)
    return redacted


def redact_command(command: list[str], replacements: list[tuple[str, str]]) -> list[str]:
    return [redact_text(arg, replacements) for arg in command]


def redact_json_value(value: object, replacements: list[tuple[str, str]]) -> object:
    if isinstance(value, str):
        return redact_text(value, replacements)
    if isinstance(value, list):
        return [redact_json_value(item, replacements) for item in value]
    if isinstance(value, dict):
        return { str(key): redact_json_value(item, replacements) for key, item in value.items() }
    return value


def write_redacted_text(path: Path, text: str, replacements: list[tuple[str, str]]) -> None:
    path.write_text(redact_text(text, replacements), encoding="utf-8")


def read_collapsed_stacks(paths: Iterable[Path], replacements: list[tuple[str, str]]) -> list[CollapsedStack]:
    merged: dict[tuple[str, ...], int] = {}
    for path in paths:
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = redact_text(raw_line.strip(), replacements)
            if not line or line.startswith("#"):
                continue
            try:
                stack_text, count_text = line.rsplit(maxsplit=1)
            except ValueError as exc:
                raise SystemExit(f"Invalid collapsed stack line in {path}: {raw_line!r}") from exc
            try:
                samples = int(count_text)
            except ValueError as exc:
                raise SystemExit(f"Invalid collapsed stack sample count in {path}: {raw_line!r}") from exc
            frames = tuple(frame for frame in stack_text.split(";") if frame)
            if not frames or samples <= 0:
                continue
            merged[frames] = merged.get(frames, 0) + samples
    return [
        CollapsedStack(frames=frames, samples=samples)
        for frames, samples in sorted(merged.items(), key=lambda item: (-item[1], item[0]))
    ]


def write_merged_collapsed_stacks(out_dir: Path, stacks: list[CollapsedStack]) -> Path:
    path = out_dir / "collapsed_stacks.txt"
    lines = [f"{';'.join(stack.frames)} {stack.samples}" for stack in stacks]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def write_speedscope(out_dir: Path, stacks: list[CollapsedStack]) -> Path:
    frame_ids: dict[str, int] = {}
    frames: list[dict[str, str]] = []

    def frame_id(name: str) -> int:
        if name not in frame_ids:
            frame_ids[name] = len(frames)
            frames.append({ "name": name })
        return frame_ids[name]

    samples = [[frame_id(frame) for frame in stack.frames] for stack in stacks]
    weights = [stack.samples for stack in stacks]
    total = sum(weights)
    data = {
        "$schema": "https://www.speedscope.app/file-format-schema.json",
        "shared": { "frames": frames },
        "profiles": [
            {
                "type": "sampled",
                "name": "LiteNN collapsed stacks",
                "unit": "samples",
                "startValue": 0,
                "endValue": total,
                "samples": samples,
                "weights": weights,
            }
        ],
        "activeProfileIndex": 0,
    }
    path = out_dir / "speedscope.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def flame_color(name: str) -> str:
    value = 0
    for ch in name:
        value = (value * 131 + ord(ch)) & 0xFFFFFFFF
    r = 180 + (value & 0x3F)
    g = 80 + ((value >> 8) & 0x5F)
    b = 40 + ((value >> 16) & 0x3F)
    return f"rgb({r},{g},{b})"


def build_flame_tree(stacks: list[CollapsedStack]) -> dict[str, object]:
    root: dict[str, object] = { "name": "root", "samples": 0, "children": {} }
    for stack in stacks:
        root["samples"] = int(root["samples"]) + stack.samples
        node = root
        for frame in stack.frames:
            children = node["children"]
            assert isinstance(children, dict)
            child = children.setdefault(frame, { "name": frame, "samples": 0, "children": {} })
            child["samples"] = int(child["samples"]) + stack.samples
            node = child
    return root


def flame_depth(node: dict[str, object]) -> int:
    children = node["children"]
    assert isinstance(children, dict)
    if not children:
        return 0
    return 1 + max(flame_depth(child) for child in children.values())


def write_flamegraph(out_dir: Path, stacks: list[CollapsedStack]) -> tuple[Path, Path]:
    root = build_flame_tree(stacks)
    total = max(int(root["samples"]), 1)
    width = 1200
    frame_height = 18
    depth = flame_depth(root)
    height = max(80, (depth + 2) * frame_height + 40)
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;font-size:12px}.frame:hover{stroke:#111;stroke-width:1}</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="8" y="18">LiteNN Flame Graph, samples={total}</text>',
    ]

    def render(node: dict[str, object], x: float, y: float, scale: float) -> None:
        children = node["children"]
        assert isinstance(children, dict)
        cursor = x
        for child in sorted(children.values(), key=lambda entry: (-int(entry["samples"]), str(entry["name"]))):
            samples = int(child["samples"])
            child_width = samples * scale
            if child_width < 0.5:
                continue
            name = str(child["name"])
            escaped = html.escape(name)
            svg_lines.append(
                f'<g><title>{escaped} ({samples} samples)</title>'
                f'<rect class="frame" x="{cursor:.3f}" y="{y:.3f}" width="{child_width:.3f}" '
                f'height="{frame_height - 1}" fill="{flame_color(name)}"/>'
                f'<text x="{cursor + 3:.3f}" y="{y + 13:.3f}" fill="#111">{escaped[:80]}</text></g>'
            )
            render(child, cursor, y + frame_height, scale)
            cursor += child_width

    render(root, 0.0, 32.0, width / total)
    svg_lines.append("</svg>")
    svg_path = out_dir / "flamegraph.svg"
    svg_path.write_text("\n".join(svg_lines), encoding="utf-8")

    html_path = out_dir / "flamegraph.html"
    html_path.write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>LiteNN Flame Graph</title>\n"
        + svg_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return svg_path, html_path


def maybe_wrap_sampler(command: list[str], sampler: str, step_dir: Path) -> tuple[list[str], list[Path], str]:
    if sampler == "none":
        return command, [], "none"

    if sampler == "linux-perf":
        perf = shutil.which("perf")
        if perf is None:
            raise SystemExit("--sampler linux-perf requested, but 'perf' was not found on PATH")
        perf_data = step_dir / "perf.data"
        return [perf, "record", "-F", "99", "-g", "-o", str(perf_data), "--", *command], [perf_data], "linux-perf"

    if sampler == "windows-xperf":
        xperf = shutil.which("xperf")
        if xperf is None:
            raise SystemExit("--sampler windows-xperf requested, but 'xperf' was not found on PATH")
        etl_path = step_dir / "xperf.etl"
        return command, [etl_path], "windows-xperf"

    raise SystemExit(f"Unsupported sampler: {sampler}")


def run_xperf(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def start_platform_sampler(sampler_name: str, sampler_outputs: list[Path]) -> list[str]:
    if sampler_name != "windows-xperf":
        return []
    xperf = shutil.which("xperf")
    if xperf is None:
        raise SystemExit("--sampler windows-xperf requested, but 'xperf' was not found on PATH")
    completed = run_xperf(
        [
            xperf,
            "-on",
            "PROC_THREAD+LOADER+PROFILE",
            "-stackwalk",
            "Profile",
            "-BufferSize",
            "1024",
            "-MinBuffers",
            "64",
            "-MaxBuffers",
            "256",
        ]
    )
    lines = [
        "windows-xperf start:",
        "command=" + " ".join(completed.args if isinstance(completed.args, list) else [str(completed.args)]),
        f"returncode={completed.returncode}",
        completed.stdout,
        completed.stderr,
    ]
    if completed.returncode != 0:
        raise SystemExit("xperf failed to start ETW sampling:\n" + "\n".join(lines))
    if sampler_outputs:
        lines.append(f"etl={sampler_outputs[0]}")
    return lines


def stop_platform_sampler(sampler_name: str, sampler_outputs: list[Path]) -> list[str]:
    if sampler_name != "windows-xperf":
        return []
    xperf = shutil.which("xperf")
    if xperf is None:
        return ["windows-xperf stop skipped: xperf disappeared from PATH"]
    if not sampler_outputs:
        return ["windows-xperf stop skipped: no ETL output path"]
    completed = run_xperf([xperf, "-d", str(sampler_outputs[0])])
    lines = [
        "windows-xperf stop:",
        "command=" + " ".join(completed.args if isinstance(completed.args, list) else [str(completed.args)]),
        f"returncode={completed.returncode}",
        completed.stdout,
        completed.stderr,
    ]
    if completed.returncode != 0:
        lines.append("warning: xperf failed to stop cleanly; ETW session may need manual cleanup")
    return lines


def run_step(
    name: str,
    command: list[str],
    out_dir: Path,
    replacements: list[tuple[str, str]],
    sampler: str,
    env: dict[str, str] | None = None,
) -> StepResult:
    step_dir = out_dir / name
    step_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = step_dir / "stdout.txt"
    stderr_path = step_dir / "stderr.txt"
    wrapped_command, sampler_outputs, sampler_name = maybe_wrap_sampler(command, sampler, step_dir)

    sampler_log_lines = [f"sampler={sampler_name}"]
    start = now_ns()
    sampler_log_lines.extend(start_platform_sampler(sampler_name, sampler_outputs))
    try:
        completed = subprocess.run(
            wrapped_command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            env=env,
            check=False,
        )
    finally:
        sampler_log_lines.extend(stop_platform_sampler(sampler_name, sampler_outputs))
        end = now_ns()

    write_redacted_text(stdout_path, completed.stdout, replacements)
    write_redacted_text(stderr_path, completed.stderr, replacements)

    sampler_log = step_dir / "sampler.txt"
    write_redacted_text(sampler_log, "\n".join(sampler_log_lines) + "\n", replacements)

    return StepResult(
        name=name,
        command=command,
        redacted_command=redact_command(command, replacements),
        returncode=completed.returncode,
        start_ns=start,
        end_ns=end,
        stdout=stdout_path,
        stderr=stderr_path,
        sampler_outputs=[path for path in sampler_outputs if path.exists()],
    )


def load_chrome_trace_events(path: Path, replacements: list[tuple[str, str]], pid_offset: int) -> list[dict[str, object]]:
    trace = load_json(path)
    if not isinstance(trace, dict) or not isinstance(trace.get("traceEvents"), list):
        raise SystemExit(f"unsupported Chrome Trace JSON: {path}")
    events: list[dict[str, object]] = []
    for raw_event in trace["traceEvents"]:
        if not isinstance(raw_event, dict):
            continue
        event = redact_json_value(raw_event, replacements)
        if not isinstance(event, dict):
            continue
        pid = event.get("pid")
        event["pid"] = (int(pid) if isinstance(pid, int) else DEFAULT_TRACE_PID) + pid_offset
        args = event.setdefault("args", {})
        if isinstance(args, dict):
            args.setdefault("imported_trace", redact_text(str(path), replacements))
        events.append(event)
    return events


def chrome_trace_events(
    results: list[StepResult],
    metadata: dict[str, object],
    imported_events: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if results:
        origin = min(result.start_ns for result in results)
    else:
        origin = now_ns()

    events: list[dict[str, object]] = []
    for index, result in enumerate(results):
        events.append(
            {
                "name": result.name,
                "cat": "litenn.profile_bundle",
                "ph": "X",
                "pid": DEFAULT_TRACE_PID,
                "tid": index + 1,
                "ts": (result.start_ns - origin) / 1000.0,
                "dur": (result.end_ns - result.start_ns) / 1000.0,
                "args": {
                    "command": result.redacted_command,
                    "returncode": result.returncode,
                    "stdout": str(result.stdout),
                    "stderr": str(result.stderr),
                },
            }
        )

    if imported_events:
        events.extend(imported_events)

    events.append(
        {
            "name": "bundle_metadata",
            "cat": "litenn.profile_bundle",
            "ph": "M",
            "pid": DEFAULT_TRACE_PID,
            "tid": 0,
            "args": metadata,
        }
    )
    return { "traceEvents": events, "metadata": metadata }


def parse_key_values(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)=(\"[^\"]*\"|\S+)", line):
        value = match.group(2)
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1]
        values[match.group(1)] = value
    return values


def optional_float(values: dict[str, str], key: str) -> float | None:
    raw = values.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def resolve_report_path(raw: object, base: Path) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return base / path


def load_qwen_smoke_evidence(report_path: Path) -> tuple[list[LogEvidence], dict[str, object]]:
    report = load_json(report_path)
    if not isinstance(report, dict) or report.get("schema") not in (
        "litenn.gguf_qwen_smoke.v1",
        "litenn.gguf_qwen_smoke.v2",
    ):
        raise SystemExit(f"unsupported qwen smoke report: {report_path}")

    base = report_path.parent
    evidence: list[LogEvidence] = []
    for raw_step in report.get("steps", []):
        if not isinstance(raw_step, dict):
            continue
        stdout = raw_step.get("stdout")
        stderr = raw_step.get("stderr")
        if stdout is None or stderr is None:
            continue
        evidence.append(
            LogEvidence(
                name=str(raw_step.get("name", "qwen_smoke_step")),
                stdout=resolve_report_path(stdout, base),
                stderr=resolve_report_path(stderr, base),
            )
        )

    links: dict[str, object] = {
        "qwen_smoke_report": str(report_path),
        "qwen_smoke_step_count": len(evidence),
    }
    for key in ("trace", "waterfall", "token_output", "text_output"):
        value = report.get(key)
        if value:
            links[f"qwen_smoke_{key}"] = str(resolve_report_path(value, base))
    return evidence, links


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def detail_has_output_columns(detail: str, columns: int) -> bool:
    return (
        re.search(rf"\bout=\d+x{columns}\b", detail) is not None
        or re.search(rf"\bout_columns={columns}\b", detail) is not None
        or re.search(rf"\bn={columns}\b", detail) is not None
    )


def detail_has_input_columns(detail: str, columns: int) -> bool:
    return (
        re.search(rf"\blhs=\d+x{columns}\b", detail) is not None
        or re.search(rf"\bin={columns}\b", detail) is not None
        or re.search(rf"\bk={columns}\b", detail) is not None
    )


def detail_value(detail: str, key: str) -> str | None:
    match = re.search(rf"\b{re.escape(key)}=(?P<value>[^\s]+)", detail)
    return match.group("value") if match is not None else None


def classify_gguf_helper(helper: str, detail: str) -> tuple[str, str]:
    if "ggml_block_matmul" in helper or "ggml_block_grouped_matmul" in helper:
        if detail_has_output_columns(detail, 152064):
            return "projection", "logits"
        if detail_has_output_columns(detail, 27648):
            return "projection", "ffn_gate_up_grouped"
        if detail_has_output_columns(detail, 13824):
            return "projection", "ffn_gate_or_up"
        if detail_has_input_columns(detail, 13824):
            return "projection", "ffn_down"
        if detail_has_output_columns(detail, 7168):
            return "projection", "qkv_grouped"
        if detail_has_output_columns(detail, 1024):
            return "projection", "kv"
        if detail_has_output_columns(detail, 5120):
            return "projection", "hidden_or_output"
        return "projection", "quantized_matmul"
    if "active_prefix_attention" in helper or "paged_attention" in helper:
        return "attention", "paged" if "paged" in helper else "active_prefix"
    if "rope" in helper:
        return "position_encoding", "rope"
    if "scatter_update" in helper or "paged_kv_append" in helper:
        return "kv_update", "append"
    if "get_rows" in helper or "embedding" in helper:
        return "embedding", "token_lookup"
    if "rms" in helper or "norm" in helper:
        return "normalization", "norm"
    return "other", "unknown"


def node_kind_for_operator(operator: str, role: str) -> str:
    if operator == "projection":
        return "QuantizedMatMulNode"
    if operator == "attention":
        return "GroupedPagedAttentionNode" if role == "paged" else "GroupedActivePrefixAttentionNode"
    if operator == "position_encoding":
        return "RoPENode"
    if operator == "kv_update":
        return "PagedKVAppendNode" if role == "append" else "ScatterNode"
    if operator == "embedding":
        return "GetRowsNode"
    if operator == "normalization":
        return "RMSNormNode"
    return "unknown"


def parse_gguf_decode_logs(results: Iterable[LogEvidence]) -> GGUFDecodeAnalysis:
    steps_by_id: dict[int, GGUFDecodeStep] = {}
    helpers: list[GGUFHelperEvent] = []
    nodes: list[GGUFNodeEvent] = []
    node_summaries: list[GGUFNodeProfileSummary] = []
    helper_pattern = re.compile(
        r"decode step (?P<step>\d+) helper (?P<helper>\S+)(?: detail=\"(?P<detail>[^\"]*)\")? "
        r"calls=(?P<calls>\d+) total_ms=(?P<total>[0-9.+\-eE]+) avg_ms=(?P<avg>[0-9.+\-eE]+)"
    )
    step_ok_pattern = re.compile(r"decode step (?P<step>\d+) ok (?P<total>[0-9.+\-eE]+) ms")
    node_pattern = re.compile(
        r"decode step (?P<step>\d+) node subgraph=(?P<subgraph>\d+) node=(?P<node>\d+) "
        r"op=(?P<op>\S+) schema=(?P<schema>\d+) calls=(?P<calls>\d+) "
        r"inclusive_ms=(?P<inclusive>[0-9.+\-eE]+) self_ms=(?P<self>[0-9.+\-eE]+) "
        r"helper_ms=(?P<helper>[0-9.+\-eE]+)"
    )
    node_summary_pattern = re.compile(
        r"decode step (?P<step>\d+) node_profile self_ms=(?P<self>[0-9.+\-eE]+) "
        r"helper_ms=(?P<helper>[0-9.+\-eE]+) calls=(?P<calls>\d+) nodes=(?P<nodes>\d+) "
        r"emitted_nodes=(?P<emitted>\d+)"
    )
    for result in results:
        for path in (result.stdout, result.stderr):
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if "stream stats " in line:
                    values = parse_key_values(line)
                    try:
                        step_id = int(values["step"])
                        steps_by_id[step_id] = GGUFDecodeStep(
                            step=step_id,
                            phase=values.get("phase", "unknown"),
                            step_ms=float(values["step_ms"]),
                            generated_tokens=int(values.get("generated_tokens", "0")),
                            tokens_per_second=float(values.get("generated_tokens_per_second", "0")),
                            input_prep_ms=optional_float(values, "input_prep_ms"),
                            module_run_ms=optional_float(values, "module_run_ms"),
                            helper_profile_enabled=values.get(
                                "helper_profile_enabled", "true" if "helper_total_ms" in values else "false"
                            ) == "true",
                            helper_total_ms=optional_float(values, "helper_total_ms"),
                            module_non_helper_ms=optional_float(values, "module_non_helper_ms"),
                            helper_profile_emit_ms=optional_float(values, "helper_profile_emit_ms"),
                            logits_output_ms=optional_float(values, "logits_output_ms"),
                            sampling_ms=optional_float(values, "sampling_ms"),
                            state_update_ms=optional_float(values, "state_update_ms"),
                            host_overhead_ms=optional_float(values, "host_overhead_ms"),
                        )
                    except (KeyError, ValueError):
                        continue
                if "decode step " in line and " ok " in line:
                    match = step_ok_pattern.search(line)
                    if match is not None:
                        try:
                            step_id = int(match.group("step"))
                            steps_by_id.setdefault(
                                step_id,
                                GGUFDecodeStep(
                                    step=step_id,
                                    phase="unknown",
                                    step_ms=float(match.group("total")),
                                    generated_tokens=0,
                                    tokens_per_second=0.0,
                                ),
                            )
                        except ValueError:
                            continue
                if "decode step " in line and " helper " in line:
                    match = helper_pattern.search(line)
                    if match is None:
                        continue
                    helper = match.group("helper")
                    detail = match.group("detail") or ""
                    operator, role = classify_gguf_helper(helper, detail)
                    helpers.append(
                        GGUFHelperEvent(
                            step=int(match.group("step")),
                            helper=helper,
                            detail=detail,
                            operator=operator,
                            role=role,
                            calls=int(match.group("calls")),
                            total_ms=float(match.group("total")),
                            avg_ms=float(match.group("avg")),
                        )
                    )
                if "decode step " in line and " node subgraph=" in line:
                    match = node_pattern.search(line)
                    if match is None:
                        continue
                    nodes.append(
                        GGUFNodeEvent(
                            step=int(match.group("step")),
                            subgraph=int(match.group("subgraph")),
                            node=int(match.group("node")),
                            op=match.group("op"),
                            schema=int(match.group("schema")),
                            calls=int(match.group("calls")),
                            inclusive_ms=float(match.group("inclusive")),
                            self_ms=float(match.group("self")),
                            helper_ms=float(match.group("helper")),
                        )
                    )
                if "decode step " in line and " node_profile " in line:
                    match = node_summary_pattern.search(line)
                    if match is None:
                        continue
                    node_summaries.append(
                        GGUFNodeProfileSummary(
                            step=int(match.group("step")),
                            self_ms=float(match.group("self")),
                            helper_ms=float(match.group("helper")),
                            calls=int(match.group("calls")),
                            nodes=int(match.group("nodes")),
                            emitted_nodes=int(match.group("emitted")),
                        )
                    )
    return GGUFDecodeAnalysis(
        steps=[steps_by_id[key] for key in sorted(steps_by_id)],
        helpers=helpers,
        nodes=nodes,
        node_summaries=node_summaries,
    )


def write_gguf_decode_analysis(out_dir: Path, analysis: GGUFDecodeAnalysis) -> dict[str, object] | None:
    if not analysis.steps and not analysis.helpers and not analysis.nodes and not analysis.node_summaries:
        return None

    step_ms_by_id = { step.step: step.step_ms for step in analysis.steps }
    total_step_ms = sum(step.step_ms for step in analysis.steps)
    helper_totals_by_step: dict[int, dict[tuple[str, str], dict[str, object]]] = {}
    operator_totals_by_step: dict[int, dict[tuple[str, str], dict[str, object]]] = {}
    node_summaries_by_step = {summary.step: summary for summary in analysis.node_summaries}
    for helper in analysis.helpers:
        step_totals = helper_totals_by_step.setdefault(helper.step, {})
        key = (helper.helper, helper.detail)
        total = step_totals.setdefault(
            key, { "helper": helper.helper, "detail": helper.detail, "calls": 0, "total_ms": 0.0 }
        )
        total["calls"] = int(total["calls"]) + helper.calls
        total["total_ms"] = float(total["total_ms"]) + helper.total_ms
        operator_step_totals = operator_totals_by_step.setdefault(helper.step, {})
        operator_key = (helper.operator, helper.role)
        operator_total = operator_step_totals.setdefault(
            operator_key, { "operator": helper.operator, "role": helper.role, "calls": 0, "total_ms": 0.0 }
        )
        operator_total["calls"] = int(operator_total["calls"]) + helper.calls
        operator_total["total_ms"] = float(operator_total["total_ms"]) + helper.total_ms

    def helper_percent(total_ms: float, denominator_ms: float) -> float | None:
        return total_ms * 100.0 / denominator_ms if denominator_ms > 0.0 else None

    runtime_bucket_specs = [
        ("input_prep", "input_prep_ms", "stream_stats_input_prep"),
        ("module_run", "module_run_ms", "stream_stats_module_run"),
        ("helper_total", "helper_total_ms", "stream_stats_helper_total"),
        ("module_non_helper", "module_non_helper_ms", "stream_stats_module_run_minus_helper_total"),
        ("helper_profile_emit", "helper_profile_emit_ms", "stream_stats_helper_profile_emit"),
        ("logits_output", "logits_output_ms", "stream_stats_logits_output"),
        ("sampling", "sampling_ms", "stream_stats_sampling"),
        ("state_update", "state_update_ms", "stream_stats_state_update"),
        ("host_overhead", "host_overhead_ms", "stream_stats_unattributed_host_overhead"),
    ]

    def helper_total_ms_for_step(step: GGUFDecodeStep) -> float:
        if not step.helper_profile_enabled:
            return 0.0
        if step.helper_total_ms is not None:
            return step.helper_total_ms
        return sum(float(helper["total_ms"]) for helper in helper_totals_by_step.get(step.step, {}).values())

    def runtime_accounted_ms(step: GGUFDecodeStep) -> float:
        non_overlapping_fields = (
            "input_prep_ms",
            "module_run_ms",
            "helper_profile_emit_ms",
            "logits_output_ms",
            "sampling_ms",
            "state_update_ms",
            "host_overhead_ms",
        )
        return sum(
            value
            for field_name in non_overlapping_fields
            for value in (getattr(step, field_name),)
            if value is not None
        )

    step_dicts = [
        {
            "step": step.step,
            "phase": step.phase,
            "step_ms": step.step_ms,
            "generated_tokens": step.generated_tokens,
            "generated_tokens_per_second": step.tokens_per_second,
            "input_prep_ms": step.input_prep_ms,
            "module_run_ms": step.module_run_ms,
            "helper_profile_enabled": step.helper_profile_enabled,
            "stream_helper_total_ms": step.helper_total_ms,
            "module_non_helper_ms": step.module_non_helper_ms,
            "helper_profile_emit_ms": step.helper_profile_emit_ms,
            "logits_output_ms": step.logits_output_ms,
            "sampling_ms": step.sampling_ms,
            "state_update_ms": step.state_update_ms,
            "host_overhead_ms": step.host_overhead_ms,
            "node_profile_summary": (
                {
                    "self_ms": node_summaries_by_step[step.step].self_ms,
                    "helper_ms": node_summaries_by_step[step.step].helper_ms,
                    "calls": node_summaries_by_step[step.step].calls,
                    "nodes": node_summaries_by_step[step.step].nodes,
                    "emitted_nodes": node_summaries_by_step[step.step].emitted_nodes,
                }
                if step.step in node_summaries_by_step
                else None
            ),
            "runtime_accounted_ms": runtime_accounted_ms(step),
            "helper_total_ms": helper_total_ms_for_step(step),
            "helper_percent_of_step": (
                helper_percent(helper_total_ms_for_step(step), step.step_ms) if step.helper_profile_enabled else None
            ),
            "residual_ms": (
                max(0.0, step.step_ms - helper_total_ms_for_step(step)) if step.helper_profile_enabled else None
            ),
            "residual_percent_of_step": (
                helper_percent(max(0.0, step.step_ms - helper_total_ms_for_step(step)), step.step_ms)
                if step.helper_profile_enabled
                else None
            ),
            "top_helper": next(
                (
                    {
                        "helper": helper["helper"],
                        "detail": helper["detail"],
                        "total_ms": helper["total_ms"],
                        "percent_of_step": helper_percent(float(helper["total_ms"]), step.step_ms),
                    }
                    for helper in sorted(
                        helper_totals_by_step.get(step.step, {}).values(),
                        key=lambda item: (-float(item["total_ms"]), str(item["helper"])),
                    )[:1]
                ),
                None,
            ),
            "top_operator": next(
                (
                    {
                        "operator": operator["operator"],
                        "role": operator["role"],
                        "total_ms": operator["total_ms"],
                        "percent_of_step": helper_percent(float(operator["total_ms"]), step.step_ms),
                    }
                    for operator in sorted(
                        operator_totals_by_step.get(step.step, {}).values(),
                        key=lambda item: (-float(item["total_ms"]), str(item["operator"]), str(item["role"])),
                    )[:1]
                ),
                None,
            ),
        }
        for step in analysis.steps
    ]
    residual_buckets: list[dict[str, object]] = []
    for phase in ("all", "prompt_replay", "generation"):
        selected_steps = (
            step_dicts
            if phase == "all"
            else [step for step in step_dicts if str(step.get("phase", "")) == phase]
        )
        if not selected_steps:
            continue
        measured_steps = [step for step in selected_steps if step["residual_ms"] is not None]
        if not measured_steps:
            continue
        total_phase_ms = sum(float(step["step_ms"]) for step in measured_steps)
        residual_phase_ms = sum(float(step["residual_ms"]) for step in measured_steps)
        residual_buckets.append(
            {
                "bucket": "non_helper_residual",
                "phase": phase,
                "steps": len(measured_steps),
                "total_ms": residual_phase_ms,
                "avg_ms_per_step": residual_phase_ms / len(measured_steps),
                "percent_of_phase_steps": helper_percent(residual_phase_ms, total_phase_ms),
                "percent_of_all_steps": helper_percent(residual_phase_ms, total_step_ms),
                "attribution": "step_ms_minus_helper_total",
            }
        )
    runtime_buckets: list[dict[str, object]] = []
    for phase in ("all", "prompt_replay", "generation"):
        selected_steps = (
            step_dicts
            if phase == "all"
            else [step for step in step_dicts if str(step.get("phase", "")) == phase]
        )
        if not selected_steps:
            continue
        total_phase_ms = sum(float(step["step_ms"]) for step in selected_steps)
        for bucket_name, field_name, attribution in runtime_bucket_specs:
            selected_values = [step.get(field_name) for step in selected_steps if step.get(field_name) is not None]
            if not selected_values:
                continue
            total_bucket_ms = sum(float(value) for value in selected_values)
            runtime_buckets.append(
                {
                    "bucket": bucket_name,
                    "phase": phase,
                    "steps": len(selected_steps),
                    "steps_with_values": len(selected_values),
                    "total_ms": total_bucket_ms,
                    "avg_ms_per_step": total_bucket_ms / len(selected_steps),
                    "avg_ms_per_measured_step": total_bucket_ms / len(selected_values),
                    "percent_of_phase_steps": helper_percent(total_bucket_ms, total_phase_ms),
                    "percent_of_all_steps": helper_percent(total_bucket_ms, total_step_ms),
                    "attribution": attribution,
                }
            )
    top_residual_steps = sorted(
        (
            {
                "step": int(step["step"]),
                "phase": step["phase"],
                "step_ms": step["step_ms"],
                "helper_total_ms": step["helper_total_ms"],
                "residual_ms": step["residual_ms"],
                "residual_percent_of_step": step["residual_percent_of_step"],
                "top_operator": step["top_operator"],
            }
            for step in step_dicts
            if step["residual_ms"] is not None and float(step["residual_ms"]) > 0.0
        ),
        key=lambda item: (-float(item["residual_ms"]), int(item["step"])),
    )
    helper_dicts = [
        {
            "step": helper.step,
            "helper": helper.helper,
            "detail": helper.detail,
            "operator": helper.operator,
            "role": helper.role,
            "calls": helper.calls,
            "total_ms": helper.total_ms,
            "avg_ms": helper.avg_ms,
        }
        for helper in analysis.helpers
    ]
    node_timing_dicts = [
        {
            "step": node.step,
            "node_kind": node.op,
            "node_name": f"subgraph{node.subgraph}/node{node.node}",
            "subgraph": node.subgraph,
            "node": node.node,
            "schema": node.schema,
            "operator": "generated",
            "role": "plan_node",
            "helper": "",
            "detail": "",
            "format": None,
            "activation": None,
            "lhs_shape": None,
            "out_shape": None,
            "query_shape": None,
            "keys_shape": None,
            "kv_shape": None,
            "requested_threads": None,
            "resolved_threads": None,
            "calls": node.calls,
            "total_ms": node.self_ms,
            "inclusive_ms": node.inclusive_ms,
            "self_ms": node.self_ms,
            "helper_ms": node.helper_ms,
            "avg_ms": node.self_ms / node.calls if node.calls else 0.0,
            "attribution": "native-plan-marker",
        }
        for node in analysis.nodes
    ] + [
        {
            "step": helper.step,
            "node_kind": node_kind_for_operator(helper.operator, helper.role),
            "node_name": f"{helper.operator}/{helper.role}",
            "operator": helper.operator,
            "role": helper.role,
            "helper": helper.helper,
            "detail": helper.detail,
            "format": detail_value(helper.detail, "format"),
            "activation": detail_value(helper.detail, "activation"),
            "lhs_shape": detail_value(helper.detail, "lhs"),
            "out_shape": detail_value(helper.detail, "out"),
            "query_shape": detail_value(helper.detail, "query") or detail_value(helper.detail, "queries"),
            "keys_shape": detail_value(helper.detail, "keys"),
            "kv_shape": detail_value(helper.detail, "kv"),
            "requested_threads": detail_value(helper.detail, "requested_threads"),
            "resolved_threads": detail_value(helper.detail, "resolved_threads"),
            "calls": helper.calls,
            "total_ms": helper.total_ms,
            "inclusive_ms": helper.total_ms,
            "self_ms": None,
            "helper_ms": helper.total_ms,
            "avg_ms": helper.avg_ms,
            "attribution": "helper-derived",
        }
        for helper in analysis.helpers
    ]
    helper_totals: dict[tuple[str, str], dict[str, object]] = {}
    for helper in analysis.helpers:
        key = (helper.helper, helper.detail)
        total = helper_totals.setdefault(
            key, { "helper": helper.helper, "detail": helper.detail, "calls": 0, "total_ms": 0.0 }
        )
        total["calls"] = int(total["calls"]) + helper.calls
        total["total_ms"] = float(total["total_ms"]) + helper.total_ms
    ranked_helpers = sorted(helper_totals.values(), key=lambda item: (-float(item["total_ms"]), str(item["helper"])))
    for helper in ranked_helpers:
        helper["percent_of_steps"] = helper_percent(float(helper["total_ms"]), total_step_ms)
    operator_totals: dict[tuple[str, str], dict[str, object]] = {}
    for helper in analysis.helpers:
        key = (helper.operator, helper.role)
        total = operator_totals.setdefault(
            key, { "operator": helper.operator, "role": helper.role, "calls": 0, "total_ms": 0.0 }
        )
        total["calls"] = int(total["calls"]) + helper.calls
        total["total_ms"] = float(total["total_ms"]) + helper.total_ms
    ranked_operators = sorted(
        operator_totals.values(), key=lambda item: (-float(item["total_ms"]), str(item["operator"]), str(item["role"]))
    )
    for operator in ranked_operators:
        operator["percent_of_steps"] = helper_percent(float(operator["total_ms"]), total_step_ms)
    native_node_totals: dict[str, dict[str, object]] = {}
    for node in analysis.nodes:
        total = native_node_totals.setdefault(
            node.op,
            {
                "node_kind": node.op,
                "calls": 0,
                "inclusive_ms": 0.0,
                "self_ms": 0.0,
                "helper_ms": 0.0,
            },
        )
        total["calls"] = int(total["calls"]) + node.calls
        total["inclusive_ms"] = float(total["inclusive_ms"]) + node.inclusive_ms
        total["self_ms"] = float(total["self_ms"]) + node.self_ms
        total["helper_ms"] = float(total["helper_ms"]) + node.helper_ms
    ranked_native_nodes = sorted(
        native_node_totals.values(), key=lambda item: (-float(item["self_ms"]), str(item["node_kind"]))
    )
    total_native_self_ms = sum(float(node["self_ms"]) for node in ranked_native_nodes)
    for node in ranked_native_nodes:
        node["percent_of_native_self"] = helper_percent(float(node["self_ms"]), total_native_self_ms)

    summary = {
        "step_count": len(analysis.steps),
        "helper_event_count": len(analysis.helpers),
        "native_node_event_count": len(analysis.nodes),
        "native_node_profile_summaries": [
            {
                "step": node.step,
                "self_ms": node.self_ms,
                "helper_ms": node.helper_ms,
                "calls": node.calls,
                "nodes": node.nodes,
                "emitted_nodes": node.emitted_nodes,
            }
            for node in analysis.node_summaries
        ],
        "total_step_ms": total_step_ms,
        "total_helper_ms": sum(float(helper["total_ms"]) for helper in ranked_helpers),
        "total_residual_ms": sum(
            float(step["residual_ms"]) for step in step_dicts if step["residual_ms"] is not None
        ),
        "residual_percent_of_steps": helper_percent(
            sum(float(step["residual_ms"]) for step in step_dicts if step["residual_ms"] is not None), total_step_ms
        ) if any(step["residual_ms"] is not None for step in step_dicts) else None,
        "generation_step_count": sum(1 for step in analysis.steps if step.phase == "generation"),
        "prompt_replay_step_count": sum(1 for step in analysis.steps if step.phase == "prompt_replay"),
        "helpers": ranked_helpers,
        "operators": ranked_operators,
        "native_node_operators": ranked_native_nodes,
        "residual_buckets": residual_buckets,
        "runtime_buckets": runtime_buckets,
        "top_residual_steps": top_residual_steps[:20],
        "steps": step_dicts,
        "helper_events": helper_dicts,
        "node_timings": node_timing_dicts,
    }

    json_path = out_dir / "gguf_decode_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    trace_events: list[dict[str, object]] = []
    step_starts: dict[int, float] = {}
    cursor_us = 0.0
    for step in analysis.steps:
        step_starts[step.step] = cursor_us
        trace_events.append(
            {
                "name": f"decode_step_{step.step}_{step.phase}",
                "cat": "litenn.gguf.decode",
                "ph": "X",
                "pid": DEFAULT_TRACE_PID + 1,
                "tid": 1,
                "ts": cursor_us,
                "dur": step.step_ms * 1000.0,
                "args": {
                    "step": step.step,
                    "phase": step.phase,
                    "generated_tokens": step.generated_tokens,
                    "generated_tokens_per_second": step.tokens_per_second,
                },
            }
        )
        cursor_us += step.step_ms * 1000.0
    for step in analysis.steps:
        runtime_cursor_us = step_starts.get(step.step, 0.0)
        for bucket_name, field_name, attribution in runtime_bucket_specs:
            value = getattr(step, field_name)
            if value is None or value <= 0.0:
                continue
            trace_events.append(
                {
                    "name": f"decode_loop_{bucket_name}",
                    "cat": "litenn.gguf.runtime_bucket",
                    "ph": "X",
                    "pid": DEFAULT_TRACE_PID + 1,
                    "tid": 4,
                    "ts": runtime_cursor_us,
                    "dur": value * 1000.0,
                    "args": {
                        "step": step.step,
                        "phase": step.phase,
                        "bucket": bucket_name,
                        "attribution": attribution,
                    },
                }
            )
            runtime_cursor_us += value * 1000.0
    for helper in analysis.helpers:
        trace_events.append(
            {
                "name": helper.helper,
                "cat": "litenn.gguf.helper",
                "ph": "X",
                "pid": DEFAULT_TRACE_PID + 1,
                "tid": 2,
                "ts": step_starts.get(helper.step, 0.0),
                "dur": helper.total_ms * 1000.0,
                "args": {
                    "step": helper.step,
                    "detail": helper.detail,
                    "operator": helper.operator,
                    "role": helper.role,
                    "calls": helper.calls,
                    "avg_ms": helper.avg_ms,
                    "percent_of_step": helper_percent(helper.total_ms, step_ms_by_id.get(helper.step, 0.0)),
                },
            }
        )
    for node in analysis.nodes:
        trace_events.append(
            {
                "name": f"{node.op}:sg{node.subgraph}:n{node.node}",
                "cat": "litenn.gguf.node",
                "ph": "X",
                "pid": DEFAULT_TRACE_PID + 1,
                "tid": 3,
                "ts": step_starts.get(node.step, 0.0),
                "dur": node.self_ms * 1000.0,
                "args": {
                    "step": node.step,
                    "subgraph": node.subgraph,
                    "node": node.node,
                    "schema": node.schema,
                    "calls": node.calls,
                    "inclusive_ms": node.inclusive_ms,
                    "self_ms": node.self_ms,
                    "helper_ms": node.helper_ms,
                    "attribution": "native-plan-marker",
                },
            }
        )
    for step in step_dicts:
        if step["residual_ms"] is None:
            continue
        residual_ms = float(step["residual_ms"])
        if residual_ms <= 0.0:
            continue
        trace_events.append(
            {
                "name": "decode_step_non_helper_residual",
                "cat": "litenn.gguf.residual",
                "ph": "X",
                "pid": DEFAULT_TRACE_PID + 1,
                "tid": 3,
                "ts": step_starts.get(int(step["step"]), 0.0) + float(step["helper_total_ms"]) * 1000.0,
                "dur": residual_ms * 1000.0,
                "args": {
                    "step": step["step"],
                    "phase": step["phase"],
                    "residual_percent_of_step": step["residual_percent_of_step"],
                    "attribution": "step_ms_minus_helper_total",
                },
            }
        )
    trace_path = out_dir / "gguf_decode_trace.json"
    trace_path.write_text(json.dumps({ "traceEvents": trace_events }, indent=2), encoding="utf-8")

    def format_optional_ms(value: object) -> str:
        return "n/a" if value is None else f"{float(value):.3f}"

    md_lines = [
        "# GGUF Decode Summary",
        "",
        f"- steps: {len(analysis.steps)}",
        f"- helper events: {len(analysis.helpers)}",
        f"- native node events: {len(analysis.nodes)}",
        f"- total step ms: {summary['total_step_ms']:.3f}",
        f"- total helper ms: {summary['total_helper_ms']:.3f}",
        f"- residual/non-helper ms: {summary['total_residual_ms']:.3f}",
        "",
        "## Top Helpers",
        "",
        "| Helper | Detail | Calls | Total ms | % of steps |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for helper in ranked_helpers[:20]:
        percent = helper["percent_of_steps"]
        percent_text = "n/a" if percent is None else f"{float(percent):.2f}%"
        md_lines.append(
            f"| `{helper['helper']}` | `{helper['detail']}` | {helper['calls']} | "
            f"{float(helper['total_ms']):.3f} | {percent_text} |"
        )
    if not ranked_helpers:
        md_lines.append("| `none` |  | 0 | 0.000 | n/a |")
    md_lines.extend(
        [
            "",
            "## Top Operators",
            "",
            "| Operator | Role | Calls | Total ms | % of steps |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for operator in ranked_operators[:20]:
        percent = operator["percent_of_steps"]
        percent_text = "n/a" if percent is None else f"{float(percent):.2f}%"
        md_lines.append(
            f"| `{operator['operator']}` | `{operator['role']}` | {operator['calls']} | "
            f"{float(operator['total_ms']):.3f} | {percent_text} |"
        )
    if not ranked_operators:
        md_lines.append("| `none` | `none` | 0 | 0.000 | n/a |")
    md_lines.extend(
        [
            "",
            "## Native Node Profile Totals",
            "",
            "| Step | Self ms | Helper ms | Calls | Nodes | Emitted nodes |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for node in analysis.node_summaries:
        md_lines.append(
            f"| {node.step} | {node.self_ms:.3f} | {node.helper_ms:.3f} | {node.calls} | "
            f"{node.nodes} | {node.emitted_nodes} |"
        )
    if not analysis.node_summaries:
        md_lines.append("| 0 | 0.000 | 0.000 | 0 | 0 | 0 |")
    md_lines.extend(
        [
            "",
            "## Native Node Kinds",
            "",
            "| Node kind | Calls | Inclusive ms | Self ms | Helper ms | % of native self |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for node in ranked_native_nodes[:20]:
        percent = node["percent_of_native_self"]
        percent_text = "n/a" if percent is None else f"{float(percent):.2f}%"
        md_lines.append(
            f"| `{node['node_kind']}` | {node['calls']} | {float(node['inclusive_ms']):.3f} | "
            f"{float(node['self_ms']):.3f} | {float(node['helper_ms']):.3f} | {percent_text} |"
        )
    if not ranked_native_nodes:
        md_lines.append("| `none` | 0 | 0.000 | 0.000 | 0.000 | n/a |")
    md_lines.extend(
        [
            "",
            "## Residual Buckets",
            "",
            "| Bucket | Phase | Steps | Total ms | Avg ms/step | % of phase | % of all steps | Attribution |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for bucket in residual_buckets:
        percent_phase = bucket["percent_of_phase_steps"]
        percent_all = bucket["percent_of_all_steps"]
        md_lines.append(
            f"| `{bucket['bucket']}` | `{bucket['phase']}` | {bucket['steps']} | "
            f"{float(bucket['total_ms']):.3f} | {float(bucket['avg_ms_per_step']):.3f} | "
            f"{'n/a' if percent_phase is None else f'{float(percent_phase):.2f}%'} | "
            f"{'n/a' if percent_all is None else f'{float(percent_all):.2f}%'} | "
            f"`{bucket['attribution']}` |"
        )
    if not residual_buckets:
        md_lines.append("| `none` | `none` | 0 | 0.000 | 0.000 | n/a | n/a | `none` |")
    md_lines.extend(
        [
            "",
            "## Runtime Buckets",
            "",
            "| Bucket | Phase | Steps | Measured steps | Total ms | Avg ms/step | Avg ms/measured step | % of phase | % of all steps | Attribution |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for bucket in runtime_buckets:
        percent_phase = bucket["percent_of_phase_steps"]
        percent_all = bucket["percent_of_all_steps"]
        md_lines.append(
            f"| `{bucket['bucket']}` | `{bucket['phase']}` | {bucket['steps']} | {bucket['steps_with_values']} | "
            f"{float(bucket['total_ms']):.3f} | {float(bucket['avg_ms_per_step']):.3f} | "
            f"{float(bucket['avg_ms_per_measured_step']):.3f} | "
            f"{'n/a' if percent_phase is None else f'{float(percent_phase):.2f}%'} | "
            f"{'n/a' if percent_all is None else f'{float(percent_all):.2f}%'} | "
            f"`{bucket['attribution']}` |"
        )
    if not runtime_buckets:
        md_lines.append("| `none` | `none` | 0 | 0 | 0.000 | 0.000 | 0.000 | n/a | n/a | `none` |")
    md_lines.extend(
        [
            "",
            "## Top Residual Steps",
            "",
            "| Step | Phase | Step ms | Helper ms | Residual ms | Residual % | Top operator |",
            "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for step in top_residual_steps[:20]:
        residual_percent = step["residual_percent_of_step"]
        top_operator = step["top_operator"]
        top_operator_text = (
            "`none`"
            if not isinstance(top_operator, dict)
            else f"`{top_operator['operator']}/{top_operator['role']}` {float(top_operator['total_ms']):.3f} ms"
        )
        md_lines.append(
            f"| {step['step']} | `{step['phase']}` | {float(step['step_ms']):.3f} | "
            f"{float(step['helper_total_ms']):.3f} | {float(step['residual_ms']):.3f} | "
            f"{'n/a' if residual_percent is None else f'{float(residual_percent):.2f}%'} | {top_operator_text} |"
        )
    if not top_residual_steps:
        md_lines.append("| 0 | `none` | 0.000 | 0.000 | 0.000 | n/a | `none` |")
    md_lines.extend(
        [
            "",
            "## Node Timings",
            "",
            "| Step | Node kind | Node | Helper | Format | Activation | LHS | Out | Calls | Inclusive ms | Self ms | Helper ms | Attribution |",
            "| ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for timing in sorted(node_timing_dicts, key=lambda item: (int(item["step"]), -float(item["total_ms"])))[:80]:
        md_lines.append(
            f"| {timing['step']} | `{timing['node_kind']}` | `{timing['node_name']}` | `{timing['helper']}` | "
            f"`{timing.get('format') or 'n/a'}` | `{timing.get('activation') or 'n/a'}` | "
            f"`{timing.get('lhs_shape') or 'n/a'}` | "
            f"`{timing.get('out_shape') or 'n/a'}` | {timing['calls']} | "
            f"{format_optional_ms(timing.get('inclusive_ms'))} | {format_optional_ms(timing.get('self_ms'))} | "
            f"{format_optional_ms(timing.get('helper_ms'))} | "
            f"`{timing['attribution']}` |"
        )
    if not node_timing_dicts:
        md_lines.append(
            "| 0 | `none` | `none` | `none` | `n/a` | `n/a` | `n/a` | `n/a` | 0 | 0.000 | 0.000 | 0.000 | `none` |"
        )
    md_lines.extend(
        [
            "",
            "## Steps",
            "",
            "| Step | Phase | Step ms | Module ms | Module non-helper ms | Host overhead ms | Helper ms | Helper % | Residual ms | Residual % | Top helper | Top operator | Tokens/s |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |",
        ]
    )
    for step in analysis.steps:
        step_summary = next(item for item in step_dicts if item["step"] == step.step)
        helper_percent_text = (
            "n/a"
            if step_summary["helper_percent_of_step"] is None
            else f"{float(step_summary['helper_percent_of_step']):.2f}%"
        )
        residual_percent_text = (
            "n/a"
            if step_summary["residual_percent_of_step"] is None
            else f"{float(step_summary['residual_percent_of_step']):.2f}%"
        )
        top_helper = step_summary["top_helper"]
        top_helper_text = (
            "`none`"
            if not isinstance(top_helper, dict)
            else f"`{top_helper['helper']}` {float(top_helper['total_ms']):.3f} ms"
        )
        top_operator = step_summary["top_operator"]
        top_operator_text = (
            "`none`"
            if not isinstance(top_operator, dict)
            else f"`{top_operator['operator']}/{top_operator['role']}` {float(top_operator['total_ms']):.3f} ms"
        )
        md_lines.append(
            f"| {step.step} | `{step.phase}` | {step.step_ms:.3f} | "
            f"{format_optional_ms(step_summary['module_run_ms'])} | "
            f"{format_optional_ms(step_summary['module_non_helper_ms'])} | "
            f"{format_optional_ms(step_summary['host_overhead_ms'])} | "
            f"{format_optional_ms(step_summary['helper_total_ms'] if step_summary['helper_profile_enabled'] else None)} | "
            f"{helper_percent_text} | {format_optional_ms(step_summary['residual_ms'])} | {residual_percent_text} | "
            f"{top_helper_text} | {top_operator_text} | {step.tokens_per_second:.3f} |"
        )
    md_path = out_dir / "gguf_decode_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "gguf_decode_summary": str(json_path),
        "gguf_decode_trace": str(trace_path),
        "gguf_decode_markdown": str(md_path),
        "gguf_decode_step_count": len(analysis.steps),
        "gguf_decode_helper_event_count": len(analysis.helpers),
    }


def write_manifest(
    out_dir: Path,
    results: list[StepResult],
    metadata: dict[str, object],
    stack_outputs: dict[str, object] | None = None,
    analysis_outputs: dict[str, object] | None = None,
    imported_outputs: dict[str, object] | None = None,
) -> dict[str, object]:
    manifest = {
        "format": "litenn-profile-bundle-v1",
        "metadata": metadata,
        "steps": [
            {
                "name": result.name,
                "command": result.redacted_command,
                "returncode": result.returncode,
                "duration_ms": result.duration_ms,
                "stdout": str(result.stdout),
                "stderr": str(result.stderr),
                "sampler_outputs": [str(path) for path in result.sampler_outputs],
            }
            for result in results
        ],
    }
    if stack_outputs is not None:
        manifest["stack_outputs"] = stack_outputs
    if analysis_outputs is not None:
        manifest["analysis_outputs"] = analysis_outputs
    if imported_outputs is not None:
        manifest["imported_outputs"] = imported_outputs
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def summarize(out_dir: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# LiteNN Profile Bundle",
        "",
        "## Metadata",
        "",
        "| Key | Value |",
        "| --- | --- |",
    ]
    metadata = manifest["metadata"]
    assert isinstance(metadata, dict)
    for key in sorted(metadata):
        lines.append(f"| `{key}` | `{metadata[key]}` |")

    lines.extend(["", "## Steps", "", "| Step | Duration ms | Return code | Outputs |", "| --- | ---: | ---: | --- |"])
    for raw_step in manifest["steps"]:
        step = dict(raw_step)
        outputs = f"`{step['stdout']}`, `{step['stderr']}`"
        lines.append(f"| `{step['name']}` | {float(step['duration_ms']):.3f} | {step['returncode']} | {outputs} |")
    if not manifest["steps"]:
        lines.append("| `none` | 0.000 | 0 | - |")

    stack_outputs = manifest.get("stack_outputs")
    if isinstance(stack_outputs, dict):
        lines.extend(["", "## Stack Outputs", "", "| Artifact | Path |", "| --- | --- |"])
        for key in ("collapsed_stacks", "speedscope", "flamegraph_svg", "flamegraph_html"):
            if key in stack_outputs:
                lines.append(f"| `{key}` | `{stack_outputs[key]}` |")
        if "total_samples" in stack_outputs:
            lines.append(f"| `total_samples` | `{stack_outputs['total_samples']}` |")

    analysis_outputs = manifest.get("analysis_outputs")
    if isinstance(analysis_outputs, dict):
        lines.extend(["", "## Analysis Outputs", "", "| Artifact | Path |", "| --- | --- |"])
        for key in ("gguf_decode_summary", "gguf_decode_trace", "gguf_decode_markdown"):
            if key in analysis_outputs:
                lines.append(f"| `{key}` | `{analysis_outputs[key]}` |")
        for key in ("gguf_decode_step_count", "gguf_decode_helper_event_count"):
            if key in analysis_outputs:
                lines.append(f"| `{key}` | `{analysis_outputs[key]}` |")

    imported_outputs = manifest.get("imported_outputs")
    if isinstance(imported_outputs, dict):
        lines.extend(["", "## Imported Outputs", "", "| Artifact | Path |", "| --- | --- |"])
        for key in sorted(imported_outputs):
            lines.append(f"| `{key}` | `{imported_outputs[key]}` |")

    lines.extend(
        [
            "",
            "## Next Diagnostics",
            "",
            "- Open `trace.json` in `chrome://tracing` or Perfetto to inspect the current command-level waterfall.",
            "- Use `--stream-stats --profile-helpers --profile-nodes` when generated-code node attribution is required; omit both profile flags for representative throughput measurements.",
            "- Use `--sampler linux-perf` on Linux to capture raw `perf.data`; convert it to collapsed stacks and pass `--collapsed-stacks` to generate Speedscope/flame graph output.",
            "- Pass private model files through `--sensitive-path` so manifest, summary, trace, stdout, and stderr redact them.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark/results/profile_bundle"))
    parser.add_argument("--litenn-profile", type=Path, help="Path to litenn_profile")
    parser.add_argument(
        "--litenn-profile-arg",
        action="append",
        default=[],
        help="Additional argument passed to litenn_profile; repeat for multiple args",
    )
    parser.add_argument("--skip-litenn-profile", action="store_true", help="Do not run litenn_profile automatically")
    parser.add_argument("--command", nargs=argparse.REMAINDER, help="Additional command to profile after '--command'")
    parser.add_argument(
        "--sampler",
        choices=("none", "linux-perf", "windows-xperf"),
        default="none",
        help="Optional platform sampler wrapper for profiled commands",
    )
    parser.add_argument(
        "--sensitive-path",
        action="append",
        default=[],
        type=Path,
        help="Path to redact from manifest, trace, summary, stdout, and stderr; repeatable",
    )
    parser.add_argument(
        "--collapsed-stacks",
        action="append",
        default=[],
        type=Path,
        help="Collapsed stack input in 'frame;frame count' format; repeatable",
    )
    parser.add_argument(
        "--qwen-smoke-report",
        action="append",
        default=[],
        type=Path,
        help="Existing example/gguf/qwen_smoke.py report to import for GGUF decode analysis; repeatable",
    )
    parser.add_argument(
        "--trace-json",
        action="append",
        default=[],
        type=Path,
        help="Existing Chrome Trace / Perfetto JSON to merge into the bundle trace; repeatable",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    replacements = normalize_sensitive_paths(args.sensitive_path)
    commands: list[tuple[str, list[str], dict[str, str] | None]] = []

    if not args.skip_litenn_profile:
        litenn_profile = discover_litenn_profile(args.litenn_profile)
        if litenn_profile is None:
            raise SystemExit("litenn_profile was not found; pass --litenn-profile or --skip-litenn-profile")
        profile_out = out_dir / "litenn_profile_output"
        commands.append(
            (
                "litenn_profile",
                [str(litenn_profile), "--out-dir", str(profile_out), *args.litenn_profile_arg],
                None,
            )
        )

    if args.command:
        command = args.command
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            raise SystemExit("--command was provided without a command")
        commands.append(("command", command, None))

    if not commands and not args.collapsed_stacks and not args.qwen_smoke_report:
        raise SystemExit("No profile command was selected")

    metadata = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cwd": str(repo_root()),
        "sampler": args.sampler,
        "pid": os.getpid(),
    }

    results: list[StepResult] = []
    for name, command, env in commands:
        result = run_step(name, command, out_dir, replacements, args.sampler, env)
        results.append(result)
        if result.returncode != 0:
            break

    imported_evidence: list[LogEvidence] = []
    imported_trace_events: list[dict[str, object]] = []
    imported_outputs: dict[str, object] | None = None
    if args.qwen_smoke_report or args.trace_json:
        imported_outputs = {}
        imported_trace_count = 0
        for report_index, report_path in enumerate(args.qwen_smoke_report):
            evidence, links = load_qwen_smoke_evidence(report_path)
            imported_evidence.extend(evidence)
            prefix = f"qwen_smoke_{report_index}"
            for key, value in links.items():
                imported_outputs[f"{prefix}_{key}"] = (
                    redact_text(value, replacements) if isinstance(value, str) else value
                )
            trace_value = links.get("qwen_smoke_trace")
            if isinstance(trace_value, str):
                trace_events = load_chrome_trace_events(Path(trace_value), replacements, 100 * (report_index + 1))
                imported_trace_events.extend(trace_events)
                imported_trace_count += len(trace_events)
        for trace_index, trace_path in enumerate(args.trace_json):
            trace_events = load_chrome_trace_events(trace_path, replacements, 1000 + 100 * trace_index)
            imported_trace_events.extend(trace_events)
            imported_trace_count += len(trace_events)
            imported_outputs[f"trace_json_{trace_index}"] = redact_text(str(trace_path), replacements)
        if imported_trace_count:
            imported_outputs["merged_trace_event_count"] = imported_trace_count

    stack_outputs: dict[str, object] | None = None
    if args.collapsed_stacks:
        stacks = read_collapsed_stacks(args.collapsed_stacks, replacements)
        collapsed_path = write_merged_collapsed_stacks(out_dir, stacks)
        speedscope_path = write_speedscope(out_dir, stacks)
        flamegraph_svg, flamegraph_html = write_flamegraph(out_dir, stacks)
        stack_outputs = {
            "collapsed_stacks": str(collapsed_path),
            "speedscope": str(speedscope_path),
            "flamegraph_svg": str(flamegraph_svg),
            "flamegraph_html": str(flamegraph_html),
            "total_samples": sum(stack.samples for stack in stacks),
        }

    log_evidence: list[LogEvidence] = [
        LogEvidence(name=result.name, stdout=result.stdout, stderr=result.stderr) for result in results
    ]
    log_evidence.extend(imported_evidence)
    analysis_outputs = write_gguf_decode_analysis(out_dir, parse_gguf_decode_logs(log_evidence))
    manifest = write_manifest(out_dir, results, metadata, stack_outputs, analysis_outputs, imported_outputs)
    trace = chrome_trace_events(results, metadata, imported_trace_events)
    (out_dir / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    summarize(out_dir, manifest)

    failures = [result for result in results if result.returncode != 0]
    if failures:
        failed = failures[0]
        print(f"profile bundle written to {out_dir}")
        print(f"step failed: {failed.name} returncode={failed.returncode}", file=sys.stderr)
        return failed.returncode if failed.returncode != 0 else 1

    print(f"profile bundle written to {out_dir}")
    print(f"trace: {out_dir / 'trace.json'}")
    print(f"summary: {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
