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


@dataclass(frozen=True)
class GGUFHelperEvent:
    step: int
    helper: str
    detail: str
    calls: int
    total_ms: float
    avg_ms: float


@dataclass(frozen=True)
class GGUFDecodeAnalysis:
    steps: list[GGUFDecodeStep]
    helpers: list[GGUFHelperEvent]


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
        raise SystemExit(
            "--sampler windows-xperf needs a start/stop ETW session and is tracked as the next implementation slice"
        )

    raise SystemExit(f"Unsupported sampler: {sampler}")


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

    start = now_ns()
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
    end = now_ns()

    write_redacted_text(stdout_path, completed.stdout, replacements)
    write_redacted_text(stderr_path, completed.stderr, replacements)

    sampler_log = step_dir / "sampler.txt"
    sampler_log.write_text(f"sampler={sampler_name}\n", encoding="utf-8")

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


def chrome_trace_events(results: list[StepResult], metadata: dict[str, object]) -> dict[str, object]:
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


def resolve_report_path(raw: object, base: Path) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
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


def parse_gguf_decode_logs(results: Iterable[LogEvidence]) -> GGUFDecodeAnalysis:
    steps: list[GGUFDecodeStep] = []
    helpers: list[GGUFHelperEvent] = []
    helper_pattern = re.compile(
        r"decode step (?P<step>\d+) helper (?P<helper>\S+)(?: detail=\"(?P<detail>[^\"]*)\")? "
        r"calls=(?P<calls>\d+) total_ms=(?P<total>[0-9.+\-eE]+) avg_ms=(?P<avg>[0-9.+\-eE]+)"
    )
    for result in results:
        for path in (result.stdout, result.stderr):
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if "stream stats " in line:
                    values = parse_key_values(line)
                    try:
                        steps.append(
                            GGUFDecodeStep(
                                step=int(values["step"]),
                                phase=values.get("phase", "unknown"),
                                step_ms=float(values["step_ms"]),
                                generated_tokens=int(values.get("generated_tokens", "0")),
                                tokens_per_second=float(values.get("generated_tokens_per_second", "0")),
                            )
                        )
                    except (KeyError, ValueError):
                        continue
                if "decode step " in line and " helper " in line:
                    match = helper_pattern.search(line)
                    if match is None:
                        continue
                    helpers.append(
                        GGUFHelperEvent(
                            step=int(match.group("step")),
                            helper=match.group("helper"),
                            detail=match.group("detail") or "",
                            calls=int(match.group("calls")),
                            total_ms=float(match.group("total")),
                            avg_ms=float(match.group("avg")),
                        )
                    )
    return GGUFDecodeAnalysis(steps=steps, helpers=helpers)


def write_gguf_decode_analysis(out_dir: Path, analysis: GGUFDecodeAnalysis) -> dict[str, object] | None:
    if not analysis.steps and not analysis.helpers:
        return None

    step_ms_by_id = { step.step: step.step_ms for step in analysis.steps }
    total_step_ms = sum(step.step_ms for step in analysis.steps)
    helper_totals_by_step: dict[int, dict[tuple[str, str], dict[str, object]]] = {}
    for helper in analysis.helpers:
        step_totals = helper_totals_by_step.setdefault(helper.step, {})
        key = (helper.helper, helper.detail)
        total = step_totals.setdefault(
            key, { "helper": helper.helper, "detail": helper.detail, "calls": 0, "total_ms": 0.0 }
        )
        total["calls"] = int(total["calls"]) + helper.calls
        total["total_ms"] = float(total["total_ms"]) + helper.total_ms

    def helper_percent(total_ms: float, denominator_ms: float) -> float | None:
        return total_ms * 100.0 / denominator_ms if denominator_ms > 0.0 else None

    step_dicts = [
        {
            "step": step.step,
            "phase": step.phase,
            "step_ms": step.step_ms,
            "generated_tokens": step.generated_tokens,
            "generated_tokens_per_second": step.tokens_per_second,
            "helper_total_ms": sum(
                float(helper["total_ms"]) for helper in helper_totals_by_step.get(step.step, {}).values()
            ),
            "helper_percent_of_step": helper_percent(
                sum(float(helper["total_ms"]) for helper in helper_totals_by_step.get(step.step, {}).values()),
                step.step_ms,
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
        }
        for step in analysis.steps
    ]
    helper_dicts = [
        {
            "step": helper.step,
            "helper": helper.helper,
            "detail": helper.detail,
            "calls": helper.calls,
            "total_ms": helper.total_ms,
            "avg_ms": helper.avg_ms,
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

    summary = {
        "step_count": len(analysis.steps),
        "helper_event_count": len(analysis.helpers),
        "total_step_ms": total_step_ms,
        "total_helper_ms": sum(float(helper["total_ms"]) for helper in ranked_helpers),
        "generation_step_count": sum(1 for step in analysis.steps if step.phase == "generation"),
        "prompt_replay_step_count": sum(1 for step in analysis.steps if step.phase == "prompt_replay"),
        "helpers": ranked_helpers,
        "steps": step_dicts,
        "helper_events": helper_dicts,
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
                    "calls": helper.calls,
                    "avg_ms": helper.avg_ms,
                    "percent_of_step": helper_percent(helper.total_ms, step_ms_by_id.get(helper.step, 0.0)),
                },
            }
        )
    trace_path = out_dir / "gguf_decode_trace.json"
    trace_path.write_text(json.dumps({ "traceEvents": trace_events }, indent=2), encoding="utf-8")

    md_lines = [
        "# GGUF Decode Summary",
        "",
        f"- steps: {len(analysis.steps)}",
        f"- helper events: {len(analysis.helpers)}",
        f"- total step ms: {summary['total_step_ms']:.3f}",
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
            "## Steps",
            "",
            "| Step | Phase | Step ms | Helper ms | Helper % | Top helper | Tokens/s |",
            "| ---: | --- | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for step in analysis.steps:
        step_summary = next(item for item in step_dicts if item["step"] == step.step)
        helper_percent_text = (
            "n/a"
            if step_summary["helper_percent_of_step"] is None
            else f"{float(step_summary['helper_percent_of_step']):.2f}%"
        )
        top_helper = step_summary["top_helper"]
        top_helper_text = (
            "`none`"
            if not isinstance(top_helper, dict)
            else f"`{top_helper['helper']}` {float(top_helper['total_ms']):.3f} ms"
        )
        md_lines.append(
            f"| {step.step} | `{step.phase}` | {step.step_ms:.3f} | "
            f"{float(step_summary['helper_total_ms']):.3f} | {helper_percent_text} | "
            f"{top_helper_text} | {step.tokens_per_second:.3f} |"
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
            "- When profiling GGUF decode with `--stream-stats` and `LITENN_COMPILE_DIAGNOSTICS=1`, open `gguf_decode_trace.json` and `gguf_decode_summary.md` for token-step and helper attribution.",
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
    imported_outputs: dict[str, object] | None = None
    if args.qwen_smoke_report:
        imported_outputs = {}
        for report_index, report_path in enumerate(args.qwen_smoke_report):
            evidence, links = load_qwen_smoke_evidence(report_path)
            imported_evidence.extend(evidence)
            prefix = f"qwen_smoke_{report_index}"
            for key, value in links.items():
                imported_outputs[f"{prefix}_{key}"] = (
                    redact_text(value, replacements) if isinstance(value, str) else value
                )

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
    trace = chrome_trace_events(results, metadata)
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
