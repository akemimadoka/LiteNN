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
import json
import os
import platform
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


def write_manifest(out_dir: Path, results: list[StepResult], metadata: dict[str, object]) -> dict[str, object]:
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

    lines.extend(
        [
            "",
            "## Next Diagnostics",
            "",
            "- Open `trace.json` in `chrome://tracing` or Perfetto to inspect the current command-level waterfall.",
            "- Use `--sampler linux-perf` on Linux to capture raw `perf.data`; Speedscope/flame graph conversion is the next slice.",
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

    if not commands:
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

    manifest = write_manifest(out_dir, results, metadata)
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
