#!/usr/bin/env python3
"""Compare LiteNN replay text against a llama.cpp golden capture.

Examples:
  python311 scripts/gguf_compare_generation_text.py \
    --manifest build/gguf_golden/hello/manifest.json \
    --replay-manifest build/gguf_golden/hello/litenn_decode_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path, expected_schema: str) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != expected_schema:
        raise SystemExit(f"unsupported schema in {path}: {data.get('schema')!r}")
    return data


def rel_path(base: Path, raw: object) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return base / path


def find_llamacpp_stdout(manifest: dict[str, object], base: Path) -> Path:
    commands = manifest.get("commands")
    if not isinstance(commands, list):
        raise SystemExit("golden manifest is missing commands")
    matches = [
        rel_path(base, command.get("stdout"))
        for command in commands
        if isinstance(command, dict) and command.get("name") == "llama_cli"
    ]
    if not matches:
        raise SystemExit("golden manifest does not contain a llama_cli stdout capture")
    if len(matches) > 1:
        raise SystemExit("golden manifest contains multiple llama_cli stdout captures")
    if not matches[0].exists():
        raise SystemExit(f"llama_cli stdout capture does not exist: {matches[0]}")
    return matches[0]


def parse_litenn_output(path: Path, prompt_token_count: int) -> tuple[list[int], list[str], str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise SystemExit(f"LiteNN output is missing token/piece lines: {path}")
    token_ids = json.loads(lines[0])
    pieces = json.loads(lines[1])
    if not isinstance(token_ids, list) or not all(isinstance(item, int) for item in token_ids):
        raise SystemExit(f"LiteNN token-id line is invalid: {path}")
    if not isinstance(pieces, list) or not all(isinstance(item, str) for item in pieces):
        raise SystemExit(f"LiteNN piece line is invalid: {path}")
    if len(pieces) < prompt_token_count:
        raise SystemExit("LiteNN piece count is smaller than the prompt token count")
    generated = "".join(pieces[prompt_token_count:])
    return token_ids, pieces, generated


def normalize(text: str, strip: bool) -> str:
    text = text.replace("\r\n", "\n")
    return text.strip() if strip else text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="llama.cpp capture manifest.json")
    parser.add_argument("--replay-manifest", required=True, type=Path, help="LiteNN litenn_decode_manifest.json")
    parser.add_argument("--output", type=Path, help="Comparison report path")
    parser.add_argument("--no-strip", action="store_true", help="Compare text without stripping leading/trailing space")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    golden = load_json(args.manifest, "litenn.llamacpp_golden_capture.v1")
    replay = load_json(args.replay_manifest, "litenn.golden_replay.v1")
    golden_base = args.manifest.parent
    stdout_path = find_llamacpp_stdout(golden, golden_base)

    replay_output = Path(str(replay.get("output", "")))
    if not replay_output.exists():
        raise SystemExit(f"LiteNN replay output does not exist: {replay_output}")
    prompt_tokens = replay.get("tokenIds")
    if not isinstance(prompt_tokens, list) or not all(isinstance(item, int) for item in prompt_tokens):
        raise SystemExit("LiteNN replay manifest is missing tokenIds")

    strip = not args.no_strip
    reference_text = normalize(stdout_path.read_text(encoding="utf-8"), strip)
    token_ids, pieces, candidate_text = parse_litenn_output(replay_output, len(prompt_tokens))
    candidate_text = normalize(candidate_text, strip)
    passed = reference_text == candidate_text
    report = {
        "schema": "litenn.llamacpp_generation_text_compare.v1",
        "sourceManifest": str(args.manifest),
        "replayManifest": str(args.replay_manifest),
        "referenceStdout": str(stdout_path),
        "candidateOutput": str(replay_output),
        "strip": strip,
        "passed": passed,
        "promptTokenCount": len(prompt_tokens),
        "candidateTokenIds": token_ids,
        "candidatePieces": pieces,
        "referenceText": reference_text,
        "candidateText": candidate_text,
    }
    output = args.output if args.output is not None else args.replay_manifest.parent / "generation_text_compare.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
