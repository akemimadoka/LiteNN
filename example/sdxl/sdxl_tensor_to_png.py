#!/usr/bin/env python3
"""Convert a LiteNN SDXL image safetensors tensor to PNG."""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
from pathlib import Path
from typing import Any


def load_safetensors(path: Path) -> tuple[dict[str, Any], memoryview]:
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError("safetensors file is too small")
    header_size = struct.unpack("<Q", data[:8])[0]
    header_end = 8 + header_size
    if header_end > len(data):
        raise ValueError("safetensors file is truncated before payload")
    header = json.loads(data[8:header_end])
    if not isinstance(header, dict):
        raise ValueError("safetensors header root must be an object")
    return header, memoryview(data)[header_end:]


def choose_tensor(header: dict[str, Any], name: str | None) -> tuple[str, dict[str, Any]]:
    if name is not None:
        raw = header.get(name)
        if not isinstance(raw, dict):
            raise KeyError(f"tensor {name!r} not found")
        return name, raw
    for candidate in ("image", "decoded", "output"):
        raw = header.get(candidate)
        if isinstance(raw, dict):
            return candidate, raw
    for key, raw in header.items():
        if key != "__metadata__" and isinstance(raw, dict):
            return key, raw
    raise ValueError("safetensors file contains no tensor")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tensor", help="Tensor name; defaults to image, decoded, output, or the first tensor")
    args = parser.parse_args()

    missing = [name for name in ("numpy", "PIL") if importlib.util.find_spec(name) is None]
    if missing:
        print("Cannot write PNG; missing Python modules:")
        for name in missing:
            print(f"  - {name}")
        return 2

    import numpy as np
    from PIL import Image

    header, payload = load_safetensors(args.input)
    name, tensor = choose_tensor(header, args.tensor)
    if tensor.get("dtype") != "F32":
        raise ValueError(f"tensor {name!r} must be F32, got {tensor.get('dtype')!r}")
    shape = tensor.get("shape")
    offsets = tensor.get("data_offsets")
    if not isinstance(shape, list) or not all(isinstance(dim, int) and dim > 0 for dim in shape):
        raise ValueError(f"tensor {name!r} has invalid shape")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError(f"tensor {name!r} has invalid data_offsets")
    begin, end = offsets
    array = np.frombuffer(payload[begin:end], dtype="<f4").reshape(shape)
    if array.ndim == 4:
        if array.shape[0] != 1 or array.shape[1] != 3:
            raise ValueError(f"rank-4 image tensor must have shape [1,3,H,W], got {shape}")
        image = np.transpose(array[0], (1, 2, 0))
    elif array.ndim == 3 and array.shape[0] == 3:
        image = np.transpose(array, (1, 2, 0))
    elif array.ndim == 3 and array.shape[2] == 3:
        image = array
    else:
        raise ValueError(f"image tensor must be [1,3,H,W], [3,H,W], or [H,W,3], got {shape}")

    pixels = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(args.output)
    print(f"Wrote {args.output} from tensor {name} shape={shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
