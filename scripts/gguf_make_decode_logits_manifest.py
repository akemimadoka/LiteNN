#!/usr/bin/env python3
"""Create an exact-token llama.cpp decode-logits reference manifest.

Use this after a llama.cpp API-level capture tool has emitted one full-vocabulary
``index: value`` file per decode step. The manifest deliberately records prompt
and generated token ids so comparisons cannot silently cross tokenizer streams.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_token_ids(text: str, label: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as error:
        raise SystemExit(f"{label} must contain comma-separated integers") from error
    if not values or any(value < 0 for value in values):
        raise SystemExit(f"{label} must contain at least one non-negative token id")
    return values


def parse_artifact(text: str) -> tuple[int, Path]:
    step_text, separator, path_text = text.partition("=")
    if not separator:
        raise SystemExit("--logits must use STEP=PATH")
    try:
        step = int(step_text)
    except ValueError as error:
        raise SystemExit(f"invalid decode step: {step_text}") from error
    path = Path(path_text)
    if step <= 0 or not path.is_file():
        raise SystemExit(f"decode step must be positive and artifact must exist: {text}")
    return step, path


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prompt-token-ids", required=True)
    parser.add_argument("--generated-token-ids", required=True)
    parser.add_argument("--logits", action="append", required=True, metavar="STEP=PATH")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--producer", default="llama.cpp")
    args = parser.parse_args()

    prompt_ids = parse_token_ids(args.prompt_token_ids, "--prompt-token-ids")
    generated_ids = parse_token_ids(args.generated_token_ids, "--generated-token-ids")
    artifacts = [parse_artifact(value) for value in args.logits]
    steps = [step for step, _ in artifacts]
    if len(set(steps)) != len(steps):
        raise SystemExit("--logits contains duplicate decode steps")
    if max(steps) > len(generated_ids):
        raise SystemExit("a decode logits step exceeds the supplied generated-token sequence")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "litenn.llamacpp_decode_logits.v1",
        "producer": args.producer,
        "model": str(args.model),
        "promptTokenIds": prompt_ids,
        "generatedTokenIds": generated_ids,
        "logitsArtifacts": [
            {
                "decodeStep": step,
                "position": len(prompt_ids) + step,
                "path": relative_or_absolute(path, args.output.parent),
            }
            for step, path in sorted(artifacts)
        ],
    }
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
