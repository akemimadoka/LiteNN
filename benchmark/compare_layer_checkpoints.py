#!/usr/bin/env python3
"""Compare layer-contiguous LiteNN-compatible hidden-state checkpoint bundles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path


DTYPE_BYTES = {
    "Float32": 4,
    "Float64": 8,
    "Float16": 2,
    "BFloat16": 2,
    "Float8E4M3": 1,
    "Float8E5M2": 1,
}


@dataclass(frozen=True)
class CheckpointRow:
    generated_index: int
    layer: int
    name: str
    dtype: str
    shape: tuple[int, ...]
    payload: Path
    byte_offset: int
    byte_size: int


def parse_shape(text: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(dim) for dim in text.split("x"))
    except ValueError as error:
        raise RuntimeError(f"invalid checkpoint shape: {text}") from error
    if not shape or any(dim <= 0 for dim in shape):
        raise RuntimeError(f"checkpoint shape must contain positive dimensions: {text}")
    return shape


def load_manifest(path: Path) -> dict[tuple[int, int], CheckpointRow]:
    with path.open(encoding="utf-8", newline="") as stream:
        schema = stream.readline().rstrip("\r\n")
        if schema != "# litenn-layer-checkpoints-v1":
            raise RuntimeError(f"unsupported layer checkpoint manifest schema in {path}: {schema}")
        reader = csv.DictReader(stream, delimiter="\t")
        required = {
            "generated_index",
            "layer",
            "name",
            "dtype",
            "shape",
            "file",
            "byte_offset",
            "byte_size",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"layer checkpoint manifest is missing required columns: {path}")
        rows: dict[tuple[int, int], CheckpointRow] = {}
        for source in reader:
            dtype = source["dtype"]
            if dtype not in DTYPE_BYTES:
                raise RuntimeError(f"unsupported checkpoint dtype {dtype!r} in {path}")
            shape = parse_shape(source["shape"])
            element_count = math.prod(shape)
            byte_size = int(source["byte_size"])
            expected_bytes = element_count * DTYPE_BYTES[dtype]
            if byte_size != expected_bytes:
                raise RuntimeError(
                    f"checkpoint byte size {byte_size} does not match {dtype} {shape} ({expected_bytes}) in {path}"
                )
            row = CheckpointRow(
                generated_index=int(source["generated_index"]),
                layer=int(source["layer"]),
                name=source["name"],
                dtype=dtype,
                shape=shape,
                payload=path.parent / source["file"],
                byte_offset=int(source["byte_offset"]),
                byte_size=byte_size,
            )
            key = (row.generated_index, row.layer)
            if key in rows:
                raise RuntimeError(f"duplicate generated-index/layer row {key} in {path}")
            if row.generated_index < 0 or row.layer < 0 or row.byte_offset < 0:
                raise RuntimeError(f"negative checkpoint coordinate in {path}: {key}")
            rows[key] = row
    if not rows:
        raise RuntimeError(f"layer checkpoint manifest contains no rows: {path}")
    return rows


def float8_to_float32(bits: int, exponent_bits: int, mantissa_bits: int, exponent_bias: int) -> float:
    sign = -1.0 if bits & 0x80 else 1.0
    exponent_mask = (1 << exponent_bits) - 1
    exponent = (bits >> mantissa_bits) & exponent_mask
    mantissa = bits & ((1 << mantissa_bits) - 1)
    scale = float(1 << mantissa_bits)
    if exponent == 0:
        return sign * math.ldexp(mantissa / scale, 1 - exponent_bias)
    if exponent == exponent_mask:
        return sign * math.inf if mantissa == 0 else math.nan
    return sign * math.ldexp(1.0 + mantissa / scale, exponent - exponent_bias)


def decode_values(dtype: str, payload: bytes) -> list[float]:
    if dtype == "Float32":
        return [value[0] for value in struct.iter_unpack("<f", payload)]
    if dtype == "Float64":
        return [value[0] for value in struct.iter_unpack("<d", payload)]
    if dtype == "Float16":
        return [value[0] for value in struct.iter_unpack("<e", payload)]
    if dtype == "BFloat16":
        return [struct.unpack("<f", struct.pack("<I", bits << 16))[0] for (bits,) in struct.iter_unpack("<H", payload)]
    if dtype == "Float8E4M3":
        return [float8_to_float32(bits, 4, 3, 7) for bits in payload]
    if dtype == "Float8E5M2":
        return [float8_to_float32(bits, 5, 2, 15) for bits in payload]
    raise RuntimeError(f"unsupported checkpoint dtype: {dtype}")


def read_row(row: CheckpointRow) -> list[float]:
    if not row.payload.is_file():
        raise RuntimeError(f"checkpoint payload does not exist: {row.payload}")
    with row.payload.open("rb") as stream:
        stream.seek(row.byte_offset)
        payload = stream.read(row.byte_size)
    if len(payload) != row.byte_size:
        raise RuntimeError(
            f"checkpoint payload range is truncated: {row.payload} offset={row.byte_offset} bytes={row.byte_size}"
        )
    return decode_values(row.dtype, payload)


def compare_values(reference: list[float], candidate: list[float], absolute: float, relative: float) -> dict[str, object]:
    if len(reference) != len(candidate):
        raise RuntimeError("checkpoint element counts do not match")
    max_absolute = 0.0
    max_relative = 0.0
    absolute_sum = 0.0
    square_sum = 0.0
    dot = 0.0
    reference_square = 0.0
    candidate_square = 0.0
    mismatch_count = 0
    non_finite_mismatch_count = 0
    max_index = 0
    for index, (expected, actual) in enumerate(zip(reference, candidate, strict=True)):
        if not math.isfinite(expected) or not math.isfinite(actual):
            equal_non_finite = expected == actual and not math.isnan(expected) and not math.isnan(actual)
            if not equal_non_finite:
                mismatch_count += 1
                non_finite_mismatch_count += 1
            continue
        difference = abs(actual - expected)
        relative_difference = difference / max(abs(expected), 1.0e-30)
        if difference > max_absolute:
            max_absolute = difference
            max_index = index
        max_relative = max(max_relative, relative_difference)
        absolute_sum += difference
        square_sum += difference * difference
        dot += expected * actual
        reference_square += expected * expected
        candidate_square += actual * actual
        if difference > absolute + relative * abs(expected):
            mismatch_count += 1
    count = len(reference)
    denominator = math.sqrt(reference_square * candidate_square)
    cosine_similarity = dot / denominator if denominator != 0.0 else (1.0 if reference == candidate else 0.0)
    return {
        "element_count": count,
        "mismatch_count": mismatch_count,
        "non_finite_mismatch_count": non_finite_mismatch_count,
        "max_absolute_error": max_absolute,
        "max_absolute_error_index": max_index,
        "max_relative_error": max_relative,
        "mean_absolute_error": absolute_sum / count,
        "rms_error": math.sqrt(square_sum / count),
        "cosine_similarity": cosine_similarity,
        "passed": mismatch_count == 0,
    }


def compare_manifests(
    reference_path: Path,
    candidate_path: Path,
    absolute_tolerance: float,
    relative_tolerance: float,
    generated_indices: set[int] | None = None,
) -> dict[str, object]:
    reference = load_manifest(reference_path)
    candidate = load_manifest(candidate_path)
    reference_keys = {key for key in reference if generated_indices is None or key[0] in generated_indices}
    candidate_keys = {key for key in candidate if generated_indices is None or key[0] in generated_indices}
    if reference_keys != candidate_keys:
        missing = sorted(reference_keys - candidate_keys)
        extra = sorted(candidate_keys - reference_keys)
        raise RuntimeError(f"checkpoint coordinate mismatch: missing={missing} extra={extra}")
    if not reference_keys:
        raise RuntimeError("no checkpoint rows match the selected generated indices")

    rows: list[dict[str, object]] = []
    for key in sorted(reference_keys):
        expected = reference[key]
        actual = candidate[key]
        if (expected.name, expected.dtype, expected.shape) != (actual.name, actual.dtype, actual.shape):
            raise RuntimeError(
                f"checkpoint metadata mismatch at generated_index={key[0]} layer={key[1]}: "
                f"reference={(expected.name, expected.dtype, expected.shape)} "
                f"candidate={(actual.name, actual.dtype, actual.shape)}"
            )
        metrics = compare_values(
            read_row(expected), read_row(actual), absolute_tolerance, relative_tolerance
        )
        rows.append(
            {
                "generated_index": key[0],
                "layer": key[1],
                "name": expected.name,
                "dtype": expected.dtype,
                "shape": list(expected.shape),
                **metrics,
            }
        )
    failing = [row for row in rows if not row["passed"]]
    first_failing_by_index: dict[str, int | None] = {}
    for generated_index in sorted({int(row["generated_index"]) for row in rows}):
        first = next(
            (int(row["layer"]) for row in rows if row["generated_index"] == generated_index and not row["passed"]),
            None,
        )
        first_failing_by_index[str(generated_index)] = first
    return {
        "schema": "litenn.layer_checkpoint_comparison.v1",
        "reference_manifest": str(reference_path),
        "candidate_manifest": str(candidate_path),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "passed": not failing,
        "compared_rows": len(rows),
        "failing_rows": len(failing),
        "first_failing_layer_by_generated_index": first_failing_by_index,
        "rows": rows,
    }


def markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Layer Checkpoint Comparison",
        "",
        f"- Result: `{'PASS' if report['passed'] else 'FAIL'}`",
        f"- Compared rows: `{report['compared_rows']}`",
        f"- Failing rows: `{report['failing_rows']}`",
        f"- Absolute/relative tolerance: `{report['absolute_tolerance']}` / `{report['relative_tolerance']}`",
        f"- First failing layers: `{report['first_failing_layer_by_generated_index']}`",
        "",
        "| Generated | Layer | Max abs | Mean abs | RMS | Max rel | Cosine | Mismatches | Result |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["rows"]:
        assert isinstance(row, dict)
        lines.append(
            f"| {row['generated_index']} | {row['layer']} | {row['max_absolute_error']:.6g} | "
            f"{row['mean_absolute_error']:.6g} | {row['rms_error']:.6g} | {row['max_relative_error']:.6g} | "
            f"{row['cosine_similarity']:.9f} | {row['mismatch_count']} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    return "\n".join(lines) + "\n"


def comma_indices(text: str) -> set[int]:
    try:
        values = {int(value) for value in text.split(",")}
    except ValueError as error:
        raise argparse.ArgumentTypeError("generated indices must be comma-separated integers") from error
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("generated indices must be non-empty and non-negative")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Reference manifest.tsv")
    parser.add_argument("--candidate", required=True, type=Path, help="Candidate manifest.tsv")
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--generated-indices", type=comma_indices)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.absolute_tolerance < 0 or args.relative_tolerance < 0:
        parser.error("tolerances must be non-negative")

    try:
        report = compare_manifests(
            args.reference,
            args.candidate,
            args.absolute_tolerance,
            args.relative_tolerance,
            args.generated_indices,
        )
    except RuntimeError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "layer_checkpoint_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "layer_checkpoint_comparison.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
