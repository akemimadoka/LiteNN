#!/usr/bin/env python3
"""Compare F32 tensors from two safetensors files."""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TensorView:
    dtype: str
    shape: list[int]
    data: np.ndarray


def load_f32_safetensors(path: Path) -> dict[str, TensorView]:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"{path} is too small to be a safetensors file")
    header_size = struct.unpack("<Q", raw[:8])[0]
    header_start = 8
    header_end = header_start + header_size
    header = json.loads(raw[header_start:header_end])
    if not isinstance(header, dict):
        raise ValueError(f"{path} safetensors header must be a JSON object")
    result: dict[str, TensorView] = {}
    payload_start = header_end
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = meta.get("dtype")
        shape = meta.get("shape")
        offsets = meta.get("data_offsets")
        if dtype != "F32":
            raise ValueError(f"{path}:{name} uses dtype {dtype}, only F32 is supported by this comparator")
        if not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
            raise ValueError(f"{path}:{name} has invalid shape metadata")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"{path}:{name} has invalid data_offsets metadata")
        start = payload_start + int(offsets[0])
        end = payload_start + int(offsets[1])
        data = np.frombuffer(raw[start:end], dtype=np.dtype("<f4")).reshape(shape)
        result[name] = TensorView(dtype=dtype, shape=shape, data=data)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--tensor", action="append", default=[], help="Tensor to compare; default compares all expected tensors")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    actual = load_f32_safetensors(args.actual)
    expected = load_f32_safetensors(args.expected)
    names = args.tensor or sorted(expected)
    failed = False
    for name in names:
        if name not in expected:
            raise KeyError(f"expected file does not contain tensor {name!r}")
        if name not in actual:
            raise KeyError(f"actual file does not contain tensor {name!r}")
        lhs = actual[name]
        rhs = expected[name]
        if lhs.shape != rhs.shape:
            raise ValueError(f"{name}: shape mismatch actual={lhs.shape} expected={rhs.shape}")
        diff = np.abs(lhs.data.astype(np.float64) - rhs.data.astype(np.float64))
        max_abs = float(diff.max(initial=0.0))
        denom = np.maximum(np.abs(rhs.data.astype(np.float64)), args.atol)
        max_rel = float((diff / denom).max(initial=0.0))
        ok = bool(np.allclose(lhs.data, rhs.data, atol=args.atol, rtol=args.rtol))
        print(f"{name}: max_abs={max_abs:.6g} max_rel={max_rel:.6g} ok={ok}")
        failed = failed or not ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
