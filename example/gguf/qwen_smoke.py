#!/usr/bin/env python3
"""Run a practical LiteNN GGUF/Qwen smoke sequence.

Examples:
  python311 example/gguf/qwen_smoke.py \
    --model path/to/qwen.gguf \
    --litenn build-release/tools/gguf/litenn_gguf_convert.exe \
    --token-ids 1,2,3,4 \
    --max-tokens 8 \
    --output build/gguf_qwen_smoke/generated_token_ids.txt \
    --workdir build/gguf_qwen_smoke

  python311 example/gguf/qwen_smoke.py \
    --model path/to/qwen.gguf \
    --token-ids 1,2,3,4 \
    --capture-llamacpp \
    --llama-debug third_party/llama.cpp/build/bin/llama-debug.exe \
    --llama-cli third_party/llama.cpp/build/bin/llama-cli.exe \
    --prompt "hello" \
    --compare-logits \
    --compare-text
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


TRACE_PID = 1
TIMED_LINE_RE = re.compile(r"^\[LiteNN (?P<category>compile|gguf)\] (?P<label>.+): ok (?P<ms>[0-9]+(?:\.[0-9]+)?) ms$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def discover_litenn(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"LiteNN GGUF tool does not exist: {explicit}")
        return explicit
    root = repo_root()
    for name in ("litenn_gguf_convert.exe", "litenn_gguf_convert"):
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)
    for candidate in (
        root / "build-release" / "tools" / "gguf" / "litenn_gguf_convert.exe",
        root / "build-release" / "tools" / "gguf" / "litenn_gguf_convert",
        root / "build" / "tools" / "gguf" / "litenn_gguf_convert.exe",
        root / "build" / "tools" / "gguf" / "litenn_gguf_convert",
    ):
        if candidate.exists():
            return candidate
    raise SystemExit("litenn_gguf_convert executable was not found; pass --litenn")


def now_ns() -> int:
    return time.perf_counter_ns()


def run_step(name: str, command: list[str], workdir: Path, env: dict[str, str] | None = None) -> dict[str, object]:
    stdout = workdir / f"{name}.stdout.txt"
    stderr = workdir / f"{name}.stderr.txt"

    def pump(stream, output_file, mirror) -> None:
        try:
            for line in stream:
                output_file.write(line)
                output_file.flush()
                encoding = mirror.encoding or "utf-8"
                mirror.write(line.encode(encoding, errors="replace").decode(encoding, errors="replace"))
                mirror.flush()
        finally:
            stream.close()

    start_ns = now_ns()
    with stdout.open("w", encoding="utf-8", errors="replace") as stdout_file, stderr.open(
        "w", encoding="utf-8", errors="replace"
    ) as stderr_file:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_thread = threading.Thread(target=pump, args=(process.stdout, stdout_file, sys.stdout), daemon=True)
        stderr_thread = threading.Thread(target=pump, args=(process.stderr, stderr_file, sys.stderr), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        try:
            returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
    end_ns = now_ns()
    return {
        "name": name,
        "command": command,
        "returncode": returncode,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_ms": (end_ns - start_ns) / 1_000_000.0,
        "stdout": str(stdout),
        "stderr": str(stderr),
    }


def parse_litenn_timed_events(step: dict[str, object]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    cursor_ms = 0.0
    for key in ("stderr", "stdout"):
        path = Path(str(step[key]))
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = TIMED_LINE_RE.match(line)
            if not match:
                continue
            duration_ms = float(match.group("ms"))
            events.append(
                {
                    "name": match.group("label"),
                    "category": f"litenn.{match.group('category')}",
                    "step": step["name"],
                    "start_ms": cursor_ms,
                    "duration_ms": duration_ms,
                    "source": key,
                }
            )
            cursor_ms += duration_ms
    return events


def write_profile_artifacts(workdir: Path, steps: list[dict[str, object]]) -> tuple[Path, Path]:
    if steps:
        origin = min(int(step.get("start_ns", 0)) for step in steps if "start_ns" in step)
    else:
        origin = now_ns()

    trace_events: list[dict[str, object]] = []
    waterfall_lines = [
        "# Qwen Smoke Waterfall",
        "",
        "| Step | Event | Duration ms | Source |",
        "| --- | --- | ---: | --- |",
    ]
    for step_index, step in enumerate(steps):
        start_ns = int(step.get("start_ns", origin))
        end_ns = int(step.get("end_ns", start_ns))
        trace_events.append(
            {
                "name": str(step["name"]),
                "cat": "litenn.qwen_smoke.step",
                "ph": "X",
                "pid": TRACE_PID,
                "tid": step_index + 1,
                "ts": (start_ns - origin) / 1000.0,
                "dur": (end_ns - start_ns) / 1000.0,
                "args": {
                    "returncode": step["returncode"],
                    "stdout": step["stdout"],
                    "stderr": step["stderr"],
                },
            }
        )
        waterfall_lines.append(
            f"| `{step['name']}` | `<whole step>` | {float(step.get('duration_ms', 0.0)):.3f} | command |"
        )
        for event in parse_litenn_timed_events(step):
            trace_events.append(
                {
                    "name": event["name"],
                    "cat": event["category"],
                    "ph": "X",
                    "pid": TRACE_PID,
                    "tid": step_index + 1,
                    "ts": (start_ns - origin) / 1000.0 + float(event["start_ms"]) * 1000.0,
                    "dur": float(event["duration_ms"]) * 1000.0,
                    "args": {
                        "step": event["step"],
                        "source": event["source"],
                    },
                }
            )
            waterfall_lines.append(
                f"| `{event['step']}` | `{event['name']}` | {float(event['duration_ms']):.3f} | {event['source']} |"
            )

    trace_path = workdir / "qwen_smoke_trace.json"
    trace_path.write_text(json.dumps({ "traceEvents": trace_events }, indent=2) + "\n", encoding="utf-8")
    waterfall_path = workdir / "qwen_smoke_waterfall.md"
    waterfall_path.write_text("\n".join(waterfall_lines) + "\n", encoding="utf-8")
    return trace_path, waterfall_path


def require_ok(step: dict[str, object], steps: list[dict[str, object]] | None = None, workdir: Path | None = None) -> None:
    if int(step["returncode"]) != 0:
        if steps is not None and workdir is not None:
            write_profile_artifacts(workdir, steps)
        stdout_path = Path(str(step["stdout"]))
        stderr_path = Path(str(step["stderr"]))
        stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
        detail = stderr_text.strip() or stdout_text.strip()
        if len(detail) > 2000:
            detail = detail[:2000] + "\n... truncated ..."
        message = (
            f"step failed: {step['name']} returncode={step['returncode']}\n"
            f"stdout: {stdout_path}\n"
            f"stderr: {stderr_path}"
        )
        if detail:
            message += f"\n\n{detail}"
        else:
            message += "\n\nNo stdout/stderr was captured; the process may have crashed or been terminated."
        raise SystemExit(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Input GGUF model")
    parser.add_argument("--litenn", type=Path, help="Path to litenn_gguf_convert")
    parser.add_argument("--workdir", type=Path, default=Path("build/gguf_qwen_smoke"))
    parser.add_argument("--token-ids", help="Externally tokenized prompt ids, comma-separated")
    parser.add_argument("--prompt", help="Text prompt for llama.cpp capture")
    parser.add_argument(
        "--llamacpp-tokenizer-tool",
        type=Path,
        help="Optional API-level llama.cpp tokenizer adapter for direct prompt execution",
    )
    parser.add_argument("--apply-chat-template", action="store_true", help="Format --prompt as one user turn")
    parser.add_argument("--steps", dest="steps", type=int, default=8)
    parser.add_argument("--max-tokens", dest="steps", type=int, help="Alias for --steps")
    parser.add_argument(
        "--until-eos",
        action="store_true",
        help="Keep decoding until EOS is generated, using --steps/--max-tokens as the safety cap",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Disable the default EOS stop condition and always run up to --steps/--max-tokens",
    )
    parser.add_argument("--output", type=Path, help="Generated token-id output path for LiteNN token-id decode")
    parser.add_argument("--text-output", type=Path, help="Detokenized generated text output")
    parser.add_argument("--profile", default="qwen2-like-causal-lm")
    parser.add_argument(
        "--backend-policy",
        choices=("cpu-aot",),
        default="cpu-aot",
        help="Execution policy for this smoke driver; CUDA policies are tracked separately",
    )
    parser.add_argument("--sample", choices=("greedy", "random"), default="greedy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--llvm-opt-level",
        type=int,
        default=0,
        choices=(0, 1, 2, 3),
        help="CPU AOT LLVM optimization level for LiteNN decode smoke; default keeps first-run latency lower",
    )
    parser.add_argument(
        "--no-compile-diagnostics",
        action="store_true",
        help="Suppress LiteNN decode compile/run progress diagnostics",
    )
    parser.add_argument(
        "--aot-cache-dir",
        type=Path,
        help="Optional LiteNN GGUF decode AOT artifact cache directory; disabled by default until large-object cache cost is reduced",
    )
    parser.add_argument(
        "--require-aot-cache-hit",
        action="store_true",
        help="Fail instead of compiling when the separated AOT cache is missing or invalid",
    )
    parser.add_argument(
        "--no-aot-cache-write",
        action="store_true",
        help="Compile and run on a cache miss without writing separated AOT cache files",
    )
    parser.add_argument(
        "--stateful",
        action="store_true",
        help="Run LiteNN direct token-id decode through the runtime-schedule stateful/logits-only AOT path",
    )
    parser.add_argument("--stream-tokens", action="store_true", help="Mirror generated token events to stdout")
    parser.add_argument("--stream-stats", action="store_true", help="Mirror per-step live decode statistics to stdout")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Build/load the decode artifact for the requested token capacity, then stop before token execution",
    )
    parser.add_argument("--capture-llamacpp", action="store_true")
    parser.add_argument("--compare-logits", action="store_true")
    parser.add_argument(
        "--decode-logits-reference",
        type=Path,
        help="Exact-token llama.cpp decode-logits manifest to compare after replay",
    )
    parser.add_argument(
        "--llamacpp-decode-golden-tool",
        type=Path,
        help="API-level helper used to capture and compare exact-token llama.cpp decode logits",
    )
    parser.add_argument("--compare-text", action="store_true")
    parser.add_argument("--llama-debug", type=Path)
    parser.add_argument("--llama-cli", type=Path)
    parser.add_argument("--allow-analysis-failure", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.until_eos and args.ignore_eos:
        raise SystemExit("--until-eos and --ignore-eos cannot be used together")
    if args.llamacpp_decode_golden_tool and args.steps < 2:
        raise SystemExit("--llamacpp-decode-golden-tool requires at least two generated steps")
    if args.capture_llamacpp and not args.prompt:
        raise SystemExit("--capture-llamacpp requires --prompt")
    if args.llamacpp_tokenizer_tool and not args.prompt:
        raise SystemExit("--llamacpp-tokenizer-tool requires --prompt")
    if args.llamacpp_tokenizer_tool and args.token_ids:
        raise SystemExit("provide either --token-ids or --llamacpp-tokenizer-tool")
    if args.text_output and not args.llamacpp_tokenizer_tool:
        raise SystemExit("--text-output requires --llamacpp-tokenizer-tool")
    if args.apply_chat_template and not args.llamacpp_tokenizer_tool:
        raise SystemExit("--apply-chat-template requires --llamacpp-tokenizer-tool")
    if (
        args.compare_logits
        or args.compare_text
        or args.decode_logits_reference
        or args.llamacpp_decode_golden_tool
    ) and not args.capture_llamacpp:
        raise SystemExit("logits/text comparisons require --capture-llamacpp")
    if args.decode_logits_reference and args.llamacpp_decode_golden_tool:
        raise SystemExit("provide either --decode-logits-reference or --llamacpp-decode-golden-tool")
    if not args.token_ids and not args.capture_llamacpp and not args.llamacpp_tokenizer_tool:
        raise SystemExit("provide --token-ids, --llamacpp-tokenizer-tool, or enable --capture-llamacpp")
    if args.output is not None and not (args.token_ids or args.llamacpp_tokenizer_tool):
        raise SystemExit("--output is only used by the direct token-id decode path")
    if args.require_aot_cache_hit and args.aot_cache_dir is None:
        raise SystemExit("--require-aot-cache-hit requires --aot-cache-dir")

    root = repo_root()
    workdir: Path = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    litenn = discover_litenn(args.litenn)
    steps: list[dict[str, object]] = []
    def require_step_ok(step: dict[str, object]) -> None:
        require_ok(step, steps, workdir)

    litenn_decode_env = os.environ.copy()
    litenn_decode_env["LITENN_CPU_AOT_LLVM_OPT_LEVEL"] = str(args.llvm_opt_level)
    if args.aot_cache_dir is not None:
        litenn_decode_env["LITENN_GGUF_AOT_CACHE_DIR"] = str(args.aot_cache_dir)
    if args.require_aot_cache_hit:
        litenn_decode_env["LITENN_GGUF_AOT_CACHE_REQUIRE_HIT"] = "1"
    if args.no_aot_cache_write:
        litenn_decode_env["LITENN_GGUF_AOT_CACHE_WRITE"] = "0"
    if not args.no_compile_diagnostics:
        litenn_decode_env["LITENN_COMPILE_DIAGNOSTICS"] = "1"

    analyze = run_step("analyze", [str(litenn), "--analyze-llm", str(args.model), args.profile], workdir)
    steps.append(analyze)
    if not args.allow_analysis_failure:
        require_step_ok(analyze)

    token_ids = args.token_ids
    resolved_token_output: Path | None = None
    resolved_text_output: Path | None = None
    tokenizer_prompt_ids: list[int] | None = None
    if args.llamacpp_tokenizer_tool is not None:
        tokenizer_dir = workdir / "llamacpp_tokenizer"
        tokenizer_output = tokenizer_dir / "prompt_tokens.json"
        tokenizer_text = args.prompt
        if args.apply_chat_template:
            formatted_prompt = tokenizer_dir / "formatted_prompt.bin"
            template_cmd = [
                sys.executable,
                str(root / "scripts" / "gguf_tokenizer_adapter.py"),
                "chat-template",
                "--tool",
                str(args.llamacpp_tokenizer_tool),
                "--model",
                str(args.model),
                "--workdir",
                str(tokenizer_dir),
                "--output",
                str(formatted_prompt),
                "--text",
                args.prompt,
            ]
            apply_template = run_step("apply_chat_template", template_cmd, workdir)
            steps.append(apply_template)
            require_step_ok(apply_template)
            tokenizer_text = formatted_prompt.read_text(encoding="utf-8")
        tokenize_cmd = [
            sys.executable,
            str(root / "scripts" / "gguf_tokenizer_adapter.py"),
            "tokenize",
            "--tool",
            str(args.llamacpp_tokenizer_tool),
            "--model",
            str(args.model),
            "--workdir",
            str(tokenizer_dir),
            "--output",
            str(tokenizer_output),
            "--text",
            tokenizer_text,
        ]
        tokenize = run_step("tokenize_prompt", tokenize_cmd, workdir)
        steps.append(tokenize)
        require_step_ok(tokenize)
        token_document = json.loads(tokenizer_output.read_text(encoding="utf-8"))
        if token_document.get("schema") != "litenn.llamacpp_tokens.v1":
            raise SystemExit("llama.cpp tokenizer adapter returned an unsupported token schema")
        tokenizer_prompt_ids = token_document.get("tokenIds")
        if (
            not isinstance(tokenizer_prompt_ids, list)
            or not tokenizer_prompt_ids
            or any(not isinstance(token_id, int) or token_id < 0 for token_id in tokenizer_prompt_ids)
        ):
            raise SystemExit("llama.cpp tokenizer adapter returned invalid or empty token ids")
        token_ids = ",".join(str(token_id) for token_id in tokenizer_prompt_ids)
    capture_dir = workdir / "llamacpp_capture"
    if args.capture_llamacpp:
        capture_cmd = [
            sys.executable,
            str(root / "scripts" / "gguf_capture_llamacpp_golden.py"),
            "--model",
            str(args.model),
            "--prompt",
            args.prompt,
            "--out-dir",
            str(capture_dir),
            "--predict",
            str(args.steps),
            "--seed",
            str(args.seed),
        ]
        if args.llama_debug is not None:
            capture_cmd += ["--llama-debug", str(args.llama_debug)]
        if args.llama_cli is not None:
            capture_cmd += ["--llama-cli", str(args.llama_cli)]
        capture = run_step("capture_llamacpp", capture_cmd, workdir)
        steps.append(capture)
        require_step_ok(capture)

        replay_cmd = [
            sys.executable,
            str(root / "scripts" / "gguf_run_litenn_from_golden.py"),
            "--manifest",
            str(capture_dir / "manifest.json"),
            "--litenn",
            str(litenn),
            "--steps",
            str(args.steps),
            "--sample",
            args.sample,
            "--seed",
            str(args.seed),
        ]
        if args.decode_logits_reference is not None or args.llamacpp_decode_golden_tool is not None:
            replay_cmd.append("--capture-decode-logits")
        replay = run_step("litenn_replay_from_golden", replay_cmd, workdir, env=litenn_decode_env)
        steps.append(replay)
        require_step_ok(replay)

        if args.compare_logits:
            compare_cmd = [
                sys.executable,
                str(root / "scripts" / "gguf_compare_llamacpp_logits.py"),
                "--manifest",
                str(capture_dir / "manifest.json"),
                "--litenn",
                str(litenn),
            ]
            compare = run_step("compare_prefill_logits", compare_cmd, workdir)
            steps.append(compare)
            require_step_ok(compare)

        if args.compare_text:
            compare_text_cmd = [
                sys.executable,
                str(root / "scripts" / "gguf_compare_generation_text.py"),
                "--manifest",
                str(capture_dir / "manifest.json"),
                "--replay-manifest",
                str(capture_dir / "litenn_decode_manifest.json"),
            ]
            compare_text = run_step("compare_generation_text", compare_text_cmd, workdir)
            steps.append(compare_text)
            require_step_ok(compare_text)

        decode_logits_reference = args.decode_logits_reference
        if args.llamacpp_decode_golden_tool is not None:
            replay_manifest = json.loads((capture_dir / "litenn_decode_manifest.json").read_text(encoding="utf-8"))
            prompt_ids = replay_manifest.get("tokenIds")
            generated_ids = replay_manifest.get("generatedTokenIds")
            if not prompt_ids or not generated_ids:
                raise SystemExit("LiteNN replay did not produce prompt/generated token ids for decode-logits capture")
            reference_dir = capture_dir / "llamacpp_decode_logits"
            capture_decode_cmd = [
                sys.executable,
                str(root / "scripts" / "gguf_capture_llamacpp_decode_logits.py"),
                "--tool",
                str(args.llamacpp_decode_golden_tool),
                "--model",
                str(args.model),
                "--prompt-token-ids",
                ",".join(str(token_id) for token_id in prompt_ids),
                "--generated-token-ids",
                ",".join(str(token_id) for token_id in generated_ids),
                "--out-dir",
                str(reference_dir),
            ]
            capture_decode = run_step("capture_llamacpp_decode_logits", capture_decode_cmd, workdir)
            steps.append(capture_decode)
            require_step_ok(capture_decode)
            decode_logits_reference = reference_dir / "manifest.json"

        if decode_logits_reference is not None:
            compare_decode_cmd = [
                sys.executable,
                str(root / "scripts" / "gguf_compare_llamacpp_decode_logits.py"),
                "--reference-manifest",
                str(decode_logits_reference),
                "--replay-manifest",
                str(capture_dir / "litenn_decode_manifest.json"),
            ]
            compare_decode = run_step("compare_decode_logits", compare_decode_cmd, workdir)
            steps.append(compare_decode)
            require_step_ok(compare_decode)

    if token_ids:
        decode_output = args.output if args.output is not None else workdir / "litenn_decode_tokens.txt"
        resolved_token_output = decode_output
        decode_output.parent.mkdir(parents=True, exist_ok=True)
        decode_cmd = [
            str(litenn),
            "--run-llama-decode-loop-token-ids",
            str(args.model),
            token_ids,
            str(args.steps),
            "--output",
            str(decode_output),
            "--sample",
            args.sample,
            "--seed",
            str(args.seed),
        ]
        if args.stateful:
            decode_cmd.append("--stateful")
        if args.ignore_eos:
            decode_cmd.append("--ignore-eos")
        if args.stream_tokens:
            decode_cmd.append("--stream-tokens")
        if args.stream_stats:
            decode_cmd.append("--stream-stats")
        if args.compile_only:
            decode_cmd.append("--compile-only")
        decode = run_step(
            "litenn_decode_token_ids",
            decode_cmd,
            workdir,
            env=litenn_decode_env,
        )
        steps.append(decode)
        require_step_ok(decode)
        if args.llamacpp_tokenizer_tool is not None and not args.compile_only:
            replay_lines = decode_output.read_text(encoding="utf-8").splitlines()
            replay_ids = json.loads(replay_lines[0]) if replay_lines else []
            if not isinstance(replay_ids, list) or any(not isinstance(token_id, int) for token_id in replay_ids):
                raise SystemExit("LiteNN direct decode output contains an invalid token-id line")
            generated_ids = replay_ids[len(tokenizer_prompt_ids or []) :]
            text_output = args.text_output if args.text_output is not None else workdir / "generated_text.bin"
            resolved_text_output = text_output
            if generated_ids:
                detokenize_dir = workdir / "llamacpp_detokenizer"
                detokenize_cmd = [
                    sys.executable,
                    str(root / "scripts" / "gguf_tokenizer_adapter.py"),
                    "detokenize",
                    "--tool",
                    str(args.llamacpp_tokenizer_tool),
                    "--model",
                    str(args.model),
                    "--workdir",
                    str(detokenize_dir),
                    "--output",
                    str(text_output),
                    "--token-ids",
                    ",".join(str(token_id) for token_id in generated_ids),
                ]
                detokenize = run_step("detokenize_generation", detokenize_cmd, workdir)
                steps.append(detokenize)
                require_step_ok(detokenize)
            else:
                text_output.parent.mkdir(parents=True, exist_ok=True)
                text_output.write_bytes(b"")

    trace_path, waterfall_path = write_profile_artifacts(workdir, steps)

    report = {
        "schema": "litenn.gguf_qwen_smoke.v2",
        "model": str(args.model),
        "backend_policy": args.backend_policy,
        "decode_mode": "stateful" if args.stateful else "functional",
        "stop_mode": "ignore_eos" if args.ignore_eos else ("until_eos" if args.until_eos else "eos_or_token_cap"),
        "aot_cache_dir": str(args.aot_cache_dir) if args.aot_cache_dir is not None else None,
        "require_aot_cache_hit": args.require_aot_cache_hit,
        "aot_cache_write": not args.no_aot_cache_write,
        "stream_tokens": args.stream_tokens,
        "stream_stats": args.stream_stats,
        "compile_only": args.compile_only,
        "fallback_used": False,
        "production_candidate": args.backend_policy == "cuda-native",
        "workdir": str(workdir),
        "token_output": str(resolved_token_output) if resolved_token_output is not None else None,
        "text_output": str(resolved_text_output) if resolved_text_output is not None else None,
        "trace": str(trace_path),
        "waterfall": str(waterfall_path),
        "steps": steps,
    }
    report_path = workdir / "qwen_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
