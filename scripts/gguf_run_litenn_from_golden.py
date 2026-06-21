#!/usr/bin/env python3
"""Run LiteNN GGUF decode from a llama.cpp golden-capture manifest.

Examples:
  python311 scripts/gguf_run_litenn_from_golden.py \
    --manifest build/gguf_golden/hello/manifest.json \
    --litenn build-release/tools/gguf/litenn_gguf_convert.exe \
    --steps 16 --sample greedy
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


TOKEN_IDS_RE = re.compile(r"^token ids:\s*(?P<ids>[0-9,\s-]+)\s*$", re.MULTILINE)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover_litenn(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"LiteNN GGUF tool does not exist: {explicit}")
        return explicit
    for name in ("litenn_gguf_convert.exe", "litenn_gguf_convert"):
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)
    root = repo_root()
    for candidate in (
        root / "build-release" / "tools" / "gguf" / "litenn_gguf_convert.exe",
        root / "build-release" / "tools" / "gguf" / "litenn_gguf_convert",
        root / "build" / "tools" / "gguf" / "litenn_gguf_convert.exe",
        root / "build" / "tools" / "gguf" / "litenn_gguf_convert",
    ):
        if candidate.exists():
            return candidate
    raise SystemExit("litenn_gguf_convert executable was not found; pass --litenn")


def load_manifest(path: Path) -> tuple[dict[str, object], Path]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "litenn.llamacpp_golden_capture.v1":
        raise SystemExit(f"unsupported golden manifest schema: {data.get('schema')!r}")
    return data, path.parent


def find_prompt_file(manifest: dict[str, object], base: Path) -> Path:
    artifacts = manifest.get("debugArtifacts")
    if not isinstance(artifacts, list):
        raise SystemExit("golden manifest is missing debugArtifacts")
    prompt_files = [base / str(path) for path in artifacts if str(path).endswith("-prompt.txt")]
    if not prompt_files:
        raise SystemExit("golden manifest does not contain a llama-debug *-prompt.txt artifact")
    if len(prompt_files) > 1:
        names = ", ".join(str(path) for path in prompt_files)
        raise SystemExit(f"golden manifest contains multiple prompt artifacts; pass a trimmed manifest: {names}")
    if not prompt_files[0].exists():
        raise SystemExit(f"prompt artifact does not exist: {prompt_files[0]}")
    return prompt_files[0]


def parse_token_ids(prompt_file: Path) -> list[int]:
    text = prompt_file.read_text(encoding="utf-8")
    match = TOKEN_IDS_RE.search(text)
    if match is None:
        raise SystemExit(f"prompt artifact does not contain a 'token ids:' line: {prompt_file}")
    ids = [int(part.strip()) for part in match.group("ids").split(",") if part.strip()]
    if not ids:
        raise SystemExit(f"prompt artifact contains no token ids: {prompt_file}")
    if any(token_id < 0 for token_id in ids):
        raise SystemExit(f"prompt artifact contains negative token ids: {prompt_file}")
    return ids


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def parse_replay_token_ids(path: Path) -> list[int]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise SystemExit(f"LiteNN replay output is empty: {path}")
    values = json.loads(lines[0])
    if not isinstance(values, list) or any(not isinstance(value, int) for value in values):
        raise SystemExit(f"LiteNN replay output has an invalid token-id line: {path}")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="llama.cpp capture manifest.json")
    parser.add_argument("--litenn", type=Path, help="Path to litenn_gguf_convert")
    parser.add_argument("--steps", type=int, help="Generated token count; defaults to manifest predict or 1")
    parser.add_argument("--output", type=Path, help="LiteNN output file; defaults under the manifest directory")
    parser.add_argument(
        "--logits-output-dir",
        type=Path,
        help="Per-position logits directory; implies --capture-decode-logits",
    )
    parser.add_argument("--capture-decode-logits", action="store_true")
    parser.add_argument("--sample", choices=("greedy", "random"), default="greedy")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--repeat-penalty", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ignore-eos", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest, base = load_manifest(args.manifest)
    model = Path(str(manifest.get("model", "")))
    if not model:
        raise SystemExit("golden manifest is missing model")
    prompt_file = find_prompt_file(manifest, base)
    token_ids = parse_token_ids(prompt_file)
    steps = args.steps
    if steps is None:
        predict = manifest.get("predict", 0)
        steps = int(predict) if isinstance(predict, int) and predict > 0 else 1
    if steps <= 0:
        raise SystemExit("--steps must be positive")

    output = args.output if args.output is not None else base / "litenn_decode_tokens.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    capture_decode_logits = args.capture_decode_logits or args.logits_output_dir is not None
    logits_output_dir = args.logits_output_dir if args.logits_output_dir is not None else base / "litenn_decode_logits"
    if capture_decode_logits:
        logits_output_dir.mkdir(parents=True, exist_ok=True)
        for stale in logits_output_dir.glob("position-*.txt"):
            stale.unlink()
    litenn = discover_litenn(args.litenn)
    command = [
        str(litenn),
        "--run-llama-decode-loop-token-ids",
        str(model),
        ",".join(str(token_id) for token_id in token_ids),
        str(steps),
        "--output",
        str(output),
        "--sample",
        args.sample,
    ]
    if capture_decode_logits:
        command += ["--logits-output-dir", str(logits_output_dir)]
    if args.temperature is not None:
        command += ["--temperature", str(args.temperature)]
    if args.top_k is not None:
        command += ["--top-k", str(args.top_k)]
    if args.top_p is not None:
        command += ["--top-p", str(args.top_p)]
    if args.repeat_penalty is not None:
        command += ["--repeat-penalty", str(args.repeat_penalty)]
    if args.seed is not None:
        command += ["--seed", str(args.seed)]
    if args.ignore_eos:
        command.append("--ignore-eos")

    completed = subprocess.run(command, text=True, capture_output=True)
    stdout_path = base / "litenn_decode.stdout.txt"
    stderr_path = base / "litenn_decode.stderr.txt"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    replay_token_ids = parse_replay_token_ids(output) if completed.returncode == 0 else []
    logits_artifacts = []
    for path in sorted(logits_output_dir.glob("position-*.txt")) if capture_decode_logits else []:
        try:
            position = int(path.stem.removeprefix("position-"))
        except ValueError:
            continue
        phase = "prompt"
        decode_step = None
        if position == len(token_ids):
            phase = "prefill"
        elif position > len(token_ids):
            phase = "decode"
            decode_step = position - len(token_ids)
        logits_artifacts.append(
            {
                "position": position,
                "phase": phase,
                "decodeStep": decode_step,
                "path": relative_or_absolute(path, base),
            }
        )
    run_manifest = {
        "schema": "litenn.golden_replay.v2",
        "sourceManifest": str(args.manifest),
        "promptArtifact": str(prompt_file),
        "tokenIds": token_ids,
        "replayTokenIds": replay_token_ids,
        "generatedTokenIds": replay_token_ids[len(token_ids) :],
        "steps": steps,
        "command": command,
        "returncode": completed.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "output": str(output),
        "logitsOutputDirectory": str(logits_output_dir) if capture_decode_logits else None,
        "logitsArtifacts": logits_artifacts,
    }
    (base / "litenn_decode_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_manifest, indent=2))
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
