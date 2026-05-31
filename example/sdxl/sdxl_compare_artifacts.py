#!/usr/bin/env python3
"""Compare SDXL safetensors artifacts and report tensor-level error metrics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import struct
from pathlib import Path
from typing import Any


TOLERANCE_PRESETS: dict[str, tuple[float, float]] = {
    "f32": (1.0e-5, 1.0e-4),
    "f16": (5.0e-3, 5.0e-2),
    "bf16": (1.0e-2, 1.0e-1),
    "image": (2.0 / 255.0, 1.0e-2),
}


def load_safetensors(path: Path) -> tuple[dict[str, Any], memoryview]:
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError(f"{path} is too small to be a safetensors file")
    header_size = struct.unpack("<Q", data[:8])[0]
    header_end = 8 + header_size
    if header_end > len(data):
        raise ValueError(f"{path} is truncated before payload")
    header = json.loads(data[8:header_end])
    if not isinstance(header, dict):
        raise ValueError(f"{path} safetensors header root must be an object")
    return header, memoryview(data)[header_end:]


def tensor_names(header: dict[str, Any]) -> list[str]:
    return [name for name, raw in header.items() if name != "__metadata__" and isinstance(raw, dict)]


def choose_tensor(header: dict[str, Any], requested: str | None, label: str) -> tuple[str, dict[str, Any]]:
    if requested is not None:
        raw = header.get(requested)
        if not isinstance(raw, dict):
            raise KeyError(f"{label} tensor {requested!r} not found")
        return requested, raw
    names = tensor_names(header)
    if len(names) != 1:
        raise ValueError(f"{label} tensor must be specified; found {names}")
    raw = header[names[0]]
    if not isinstance(raw, dict):
        raise ValueError(f"{label} tensor {names[0]!r} has invalid metadata")
    return names[0], raw


def tensor_to_numpy(payload: memoryview, tensor: dict[str, Any]) -> Any:
    import numpy as np

    dtype = tensor.get("dtype")
    shape = tensor.get("shape")
    offsets = tensor.get("data_offsets")
    if not isinstance(dtype, str):
        raise ValueError("tensor has invalid dtype")
    if not isinstance(shape, list) or not all(isinstance(dim, int) and dim >= 0 for dim in shape):
        raise ValueError("tensor has invalid shape")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError("tensor has invalid data_offsets")
    begin, end = offsets
    if not isinstance(begin, int) or not isinstance(end, int) or begin < 0 or end < begin or end > len(payload):
        raise ValueError("tensor has out-of-range data_offsets")

    raw = payload[begin:end]
    if dtype == "F64":
        return np.frombuffer(raw, dtype="<f8").reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(shape)
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape)
    if dtype == "BF16":
        values = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        return (values << 16).view(np.float32).reshape(shape)
    if dtype == "I64":
        return np.frombuffer(raw, dtype="<i8").reshape(shape)
    if dtype == "I32":
        return np.frombuffer(raw, dtype="<i4").reshape(shape)
    if dtype == "I8":
        return np.frombuffer(raw, dtype="<i1").reshape(shape)
    if dtype == "U8":
        return np.frombuffer(raw, dtype="<u1").reshape(shape)
    raise ValueError(f"unsupported tensor dtype {dtype!r}")


def compute_metrics(actual: Any, expected: Any, atol: float, rtol: float, allow_nonfinite: bool) -> dict[str, Any]:
    import numpy as np

    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: actual {actual.shape}, expected {expected.shape}")

    actual64 = actual.astype(np.float64, copy=False)
    expected64 = expected.astype(np.float64, copy=False)
    actual_finite = np.isfinite(actual64)
    expected_finite = np.isfinite(expected64)
    nonfinite_actual = int(actual64.size - np.count_nonzero(actual_finite))
    nonfinite_expected = int(expected64.size - np.count_nonzero(expected_finite))
    if not allow_nonfinite and (nonfinite_actual != 0 or nonfinite_expected != 0):
        raise ValueError(
            f"non-finite values found: actual={nonfinite_actual}, expected={nonfinite_expected}"
        )

    finite_pair = actual_finite & expected_finite
    if not np.any(finite_pair):
        raise ValueError("no finite value pairs available for comparison")

    diff = actual64[finite_pair] - expected64[finite_pair]
    abs_diff = np.abs(diff)
    expected_abs = np.abs(expected64[finite_pair])
    tolerance = atol + rtol * expected_abs
    violations = abs_diff > tolerance
    max_abs_index = int(np.argmax(abs_diff))
    denom = np.maximum(expected_abs, atol)
    rel_diff = abs_diff / denom
    return {
        "elements": int(actual64.size),
        "finite_pairs": int(np.count_nonzero(finite_pair)),
        "nonfinite_actual": nonfinite_actual,
        "nonfinite_expected": nonfinite_expected,
        "mean_abs": float(np.mean(abs_diff)),
        "max_abs": float(np.max(abs_diff)),
        "rmse": float(math.sqrt(float(np.mean(diff * diff)))),
        "max_rel": float(np.max(rel_diff)),
        "max_abs_actual": float(actual64[finite_pair][max_abs_index]),
        "max_abs_expected": float(expected64[finite_pair][max_abs_index]),
        "violations": int(np.count_nonzero(violations)),
        "atol": atol,
        "rtol": rtol,
        "passed": bool(not np.any(violations) and nonfinite_actual == 0 and nonfinite_expected == 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--tensor", help="Tensor name to use in both files")
    parser.add_argument("--actual-tensor", help="Tensor name in --actual; overrides --tensor")
    parser.add_argument("--expected-tensor", help="Tensor name in --expected; overrides --tensor")
    parser.add_argument("--preset", choices=sorted(TOLERANCE_PRESETS), default="f32")
    parser.add_argument("--atol", type=float, help="Absolute tolerance override")
    parser.add_argument("--rtol", type=float, help="Relative tolerance override")
    parser.add_argument("--allow-nonfinite", action="store_true")
    parser.add_argument("--json", type=Path, help="Optional JSON metrics output")
    args = parser.parse_args()

    if importlib.util.find_spec("numpy") is None:
        print("Cannot compare tensors; missing Python module: numpy")
        return 2

    default_atol, default_rtol = TOLERANCE_PRESETS[args.preset]
    atol = default_atol if args.atol is None else args.atol
    rtol = default_rtol if args.rtol is None else args.rtol

    actual_header, actual_payload = load_safetensors(args.actual)
    expected_header, expected_payload = load_safetensors(args.expected)
    actual_name, actual_info = choose_tensor(actual_header, args.actual_tensor or args.tensor, "actual")
    expected_name, expected_info = choose_tensor(expected_header, args.expected_tensor or args.tensor, "expected")
    actual = tensor_to_numpy(actual_payload, actual_info)
    expected = tensor_to_numpy(expected_payload, expected_info)
    metrics = compute_metrics(actual, expected, atol=atol, rtol=rtol, allow_nonfinite=args.allow_nonfinite)
    metrics.update(
        {
            "actual": str(args.actual),
            "expected": str(args.expected),
            "actual_tensor": actual_name,
            "expected_tensor": expected_name,
            "actual_dtype": actual_info.get("dtype"),
            "expected_dtype": expected_info.get("dtype"),
            "shape": list(actual.shape),
            "preset": args.preset,
        }
    )

    print(
        "compare "
        f"actual={args.actual}:{actual_name} expected={args.expected}:{expected_name} "
        f"preset={args.preset} atol={atol:g} rtol={rtol:g} "
        f"mean_abs={metrics['mean_abs']:.6g} max_abs={metrics['max_abs']:.6g} "
        f"rmse={metrics['rmse']:.6g} max_rel={metrics['max_rel']:.6g} "
        f"violations={metrics['violations']} passed={metrics['passed']}"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
