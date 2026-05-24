#!/usr/bin/env python3
"""Write small safetensors input-binding files for LiteNN SDXL smoke graphs."""

from __future__ import annotations

import argparse
import json
import random
import struct
import sys
from array import array
from math import prod
from pathlib import Path


def parse_shape(text: str) -> list[int]:
    separators = "," if "," in text else "x"
    dims = [int(part) for part in text.split(separators) if part]
    if not dims or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError(f"invalid shape {text!r}")
    return dims


def parse_tensor(text: str) -> tuple[str, list[int]]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("--tensor must use name=dimxdim or name=dim,dim syntax")
    name, shape_text = text.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("--tensor name must not be empty")
    return name, parse_shape(shape_text)


def validate_image_shape(height: int, width: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError("--height and --width must be positive")
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError("--height and --width must be divisible by 8 for SDXL latent shapes")


def probe_tensors(args: argparse.Namespace) -> dict[str, list[int]]:
    if args.probe is None:
        return {}
    validate_image_shape(args.height, args.width)
    latent_h = args.height // 8
    latent_w = args.width // 8
    feature_h = max(args.height // 32, 1)
    feature_w = max(args.width // 32, 1)
    if args.probe == "unet-euler-smoke":
        return {
            "latent": [args.batch, args.latent_channels, latent_h, latent_w],
            "timestep": [args.batch],
        }
    if args.probe == "unet-conditioning-smoke":
        return {
            "latent": [args.batch, args.latent_channels, latent_h, latent_w],
            "timestep": [args.batch],
            "vector_cond": [args.batch, args.vector_width],
        }
    if args.probe == "unet-full-fixed":
        return {
            "latent": [args.batch, args.latent_channels, latent_h, latent_w],
            "timestep": [args.batch],
            "context": [args.context_tokens, args.context_width],
            "vector_cond": [args.batch, args.vector_width],
        }
    if args.probe == "spatial-transformer-smoke":
        return {
            "tokens": [args.tokens, args.channels],
            "context": [args.context_tokens, args.context_width],
        }
    if args.probe == "spatial-transformer-2d-smoke":
        return {
            "features": [args.batch, args.channels, feature_h, feature_w],
            "context": [args.context_tokens, args.context_width],
        }
    if args.probe == "vae-decode-full":
        return {
            "latent": [args.batch, args.latent_channels, latent_h, latent_w],
        }
    raise ValueError(f"unsupported probe {args.probe!r}")


def make_f32_payload(shape: list[int], fill: str, rng: random.Random) -> bytes:
    count = prod(shape)
    if fill == "zero":
        return b"\0" * (count * 4)
    values = array("f", (rng.gauss(0.0, 1.0) for _ in range(count)))
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def write_safetensors(path: Path, tensors: dict[str, list[int]], fill: str, seed: int) -> None:
    rng = random.Random(seed)
    header: dict[str, dict[str, object]] = {}
    payloads: list[bytes] = []
    offset = 0
    for name, shape in tensors.items():
        payload = make_f32_payload(shape, fill, rng)
        end = offset + len(payload)
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, end],
        }
        payloads.append(payload)
        offset = end

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(header_bytes)))
        stream.write(header_bytes)
        for payload in payloads:
            stream.write(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--probe",
        choices=[
            "unet-euler-smoke",
            "unet-conditioning-smoke",
            "unet-full-fixed",
            "spatial-transformer-smoke",
            "spatial-transformer-2d-smoke",
            "vae-decode-full",
        ],
    )
    parser.add_argument("--tensor", action="append", type=parse_tensor, default=[], help="Extra tensor as name=dimxdim")
    parser.add_argument("--fill", choices=["zero", "random"], default="zero")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--channels", type=int, default=1280)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--context-tokens", type=int, default=77)
    parser.add_argument("--context-width", type=int, default=2048)
    parser.add_argument("--vector-width", type=int, default=2816)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tensors = probe_tensors(args)
    for name, shape in args.tensor:
        tensors[name] = shape
    if not tensors:
        raise ValueError("provide --probe and/or one or more --tensor bindings")
    write_safetensors(args.output, tensors, args.fill, args.seed)
    total_bytes = sum(prod(shape) * 4 for shape in tensors.values())
    print(f"Wrote {args.output} with {len(tensors)} F32 tensor(s), payload_bytes={total_bytes}")
    for name, shape in tensors.items():
        print(f"  {name}: {shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
