#!/usr/bin/env python3
"""Convert a LiteNN SDXL image safetensors tensor to PNG."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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


def tensor_payload_to_float32(payload: memoryview, begin: int, end: int, dtype: str, shape: list[int]) -> Any:
    import numpy as np

    if dtype == "F32":
        return np.frombuffer(payload[begin:end], dtype="<f4").reshape(shape)
    if dtype == "F16":
        return np.frombuffer(payload[begin:end], dtype="<f2").astype(np.float32).reshape(shape)
    if dtype == "BF16":
        raw = np.frombuffer(payload[begin:end], dtype="<u2").astype(np.uint32)
        return (raw << 16).view(np.float32).reshape(shape)
    raise ValueError(f"unsupported image tensor dtype {dtype!r}; expected F32, F16, or BF16")


def format_float_list(values: Any) -> str:
    return "[" + ", ".join(f"{float(value):.6g}" for value in values) + "]"


def choose_value_range(image: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    raw_min = float(image.min(initial=math.inf))
    raw_max = float(image.max(initial=-math.inf))
    if raw_min < -0.05 and raw_min >= -1.25 and raw_max <= 1.25:
        return "minus-one-one"
    return "zero-one"


def normalize_image(image: Any, value_range: str) -> Any:
    if value_range == "zero-one":
        return image
    if value_range == "minus-one-one":
        return (image + 1.0) * 0.5
    raise ValueError(f"unsupported image value range {value_range!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tensor", help="Tensor name; defaults to image, decoded, output, or the first tensor")
    parser.add_argument(
        "--range",
        choices=["auto", "zero-one", "minus-one-one"],
        default="auto",
        help="Input image value range; auto treats negative SDXL VAE outputs as [-1,1] and otherwise uses [0,1]",
    )
    parser.add_argument("--validate", action="store_true", help="Fail on non-finite, constant, or fully clipped images")
    parser.add_argument("--min-std", type=float, default=1.0e-4, help="Minimum normalized image stddev for --validate")
    parser.add_argument(
        "--max-hard-clip-ratio",
        type=float,
        default=0.995,
        help="Maximum fraction of values outside [0,1] before clipping for --validate",
    )
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
    dtype = tensor.get("dtype")
    if not isinstance(dtype, str):
        raise ValueError(f"tensor {name!r} has invalid dtype")
    shape = tensor.get("shape")
    offsets = tensor.get("data_offsets")
    if not isinstance(shape, list) or not all(isinstance(dim, int) and dim > 0 for dim in shape):
        raise ValueError(f"tensor {name!r} has invalid shape")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError(f"tensor {name!r} has invalid data_offsets")
    begin, end = offsets
    array = tensor_payload_to_float32(payload, begin, end, dtype, shape)
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

    if not np.all(np.isfinite(image)):
        raise ValueError(f"tensor {name!r} contains NaN or Inf values")

    value_range = choose_value_range(image, args.range)
    normalized = normalize_image(image, value_range)
    out_of_range = (normalized < 0.0) | (normalized > 1.0)
    hard_clip_ratio = float(np.mean(out_of_range))
    clipped_low_ratio = float(np.mean(normalized <= 0.0))
    clipped_high_ratio = float(np.mean(normalized >= 1.0))
    image_min = float(image.min(initial=math.inf))
    image_max = float(image.max(initial=-math.inf))
    image_mean = float(image.mean())
    image_std = float(image.std())
    normalized_mean = float(normalized.mean())
    normalized_std = float(normalized.std())
    channel_mean = normalized.reshape(-1, 3).mean(axis=0)
    channel_std = normalized.reshape(-1, 3).std(axis=0)

    print(
        "Image stats "
        f"tensor={name} shape={shape} dtype={dtype} range={value_range} "
        f"raw_min={image_min:.6g} raw_max={image_max:.6g} raw_mean={image_mean:.6g} raw_std={image_std:.6g} "
        f"norm_mean={normalized_mean:.6g} norm_std={normalized_std:.6g} "
        f"out_of_range={hard_clip_ratio:.6g} clip_low={clipped_low_ratio:.6g} clip_high={clipped_high_ratio:.6g} "
        f"channel_mean={format_float_list(channel_mean)} channel_std={format_float_list(channel_std)}",
        flush=True,
    )

    if args.validate:
        failures: list[str] = []
        if normalized_std < args.min_std:
            failures.append(f"normalized stddev {normalized_std:.6g} is below {args.min_std:.6g}")
        if hard_clip_ratio > args.max_hard_clip_ratio:
            failures.append(
                f"{hard_clip_ratio:.6g} of values are outside [0,1] before clipping, above {args.max_hard_clip_ratio:.6g}"
            )
        if failures:
            for failure in failures:
                print(f"Image validation failure: {failure}", flush=True)
            return 1
        warnings: list[str] = []
        if clipped_low_ratio + clipped_high_ratio > 0.85:
            warnings.append("most values are at the display clamp boundary")
        if normalized_std < 0.01:
            warnings.append("very low contrast; inspect the image and latent statistics")
        for warning in warnings:
            print(f"Image validation warning: {warning}", flush=True)
        print("Image validation: ok", flush=True)

    pixels = np.clip(normalized * 255.0 + 0.5, 0, 255).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(args.output)
    print(f"Wrote {args.output} from tensor {name} shape={shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
