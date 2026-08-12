#!/usr/bin/env python3
"""Compare a directory suite of aligned decoder sub-layer checkpoint manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_layer_checkpoints import comma_indices, compare_manifests


BOUNDARY_ORDER = (
    "attention_norm",
    "query_rotated",
    "key_rotated",
    "value",
    "attention_context",
    "attention_output",
    "attention_residual",
    "ffn_norm",
    "ffn_gate",
    "ffn_up",
    "ffn_swiglu",
    "ffn_down",
    "post_ffn",
)


def discover_boundaries(reference_root: Path, candidate_root: Path) -> list[str]:
    reference = {path.parent.name for path in reference_root.glob("*/manifest.tsv")}
    candidate = {path.parent.name for path in candidate_root.glob("*/manifest.tsv")}
    if reference != candidate:
        raise RuntimeError(
            f"sub-layer boundary mismatch: missing={sorted(reference - candidate)} extra={sorted(candidate - reference)}"
        )
    unknown = reference - set(BOUNDARY_ORDER)
    if unknown:
        raise RuntimeError(f"unknown sub-layer checkpoint boundaries: {sorted(unknown)}")
    boundaries = [boundary for boundary in BOUNDARY_ORDER if boundary in reference]
    if not boundaries:
        raise RuntimeError("no sub-layer checkpoint manifests were found")
    return boundaries


def aggregate_reports(
    reports: dict[str, dict[str, object]], target_generated_index: int
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    control_indices: list[int] | None = None
    for boundary in BOUNDARY_ORDER:
        if boundary not in reports:
            continue
        analysis = reports[boundary].get("target_outlier_analysis")
        if not isinstance(analysis, dict):
            raise RuntimeError(f"boundary {boundary} is missing target outlier analysis")
        boundary_controls = [int(value) for value in analysis["control_generated_indices"]]
        if control_indices is None:
            control_indices = boundary_controls
        elif boundary_controls != control_indices:
            raise RuntimeError("sub-layer checkpoint boundaries use different control generated indices")
        for source in analysis["rows"]:
            normalized = source["normalized_rms_error"]
            cosine = source["cosine_distance"]
            normalized_ratio = normalized["ratio_to_control_median"]
            cosine_ratio = cosine["ratio_to_control_median"]
            normalized_z = normalized["modified_z_score"]
            cosine_z = cosine["modified_z_score"]
            rows.append(
                {
                    "boundary": boundary,
                    "boundary_index": BOUNDARY_ORDER.index(boundary),
                    "layer": int(source["layer"]),
                    "target_normalized_rms_error": normalized["target"],
                    "control_normalized_rms_error_median": normalized["control_median"],
                    "control_normalized_rms_error_maximum": normalized["control_maximum"],
                    "normalized_rms_error_ratio_to_control_median": normalized_ratio,
                    "normalized_rms_error_modified_z_score": normalized_z,
                    "normalized_rms_error_above_control_maximum": normalized["above_control_maximum"],
                    "target_cosine_distance": cosine["target"],
                    "control_cosine_distance_median": cosine["control_median"],
                    "control_cosine_distance_maximum": cosine["control_maximum"],
                    "cosine_distance_ratio_to_control_median": cosine_ratio,
                    "cosine_distance_modified_z_score": cosine_z,
                    "cosine_distance_above_control_maximum": cosine["above_control_maximum"],
                    "joint_above_control_maximum": (
                        normalized["above_control_maximum"] and cosine["above_control_maximum"]
                    ),
                    "joint_modified_z_score": (
                        min(float(normalized_z), float(cosine_z))
                        if normalized_z is not None and cosine_z is not None
                        else None
                    ),
                }
            )

    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["joint_modified_z_score"])
            if row["joint_modified_z_score"] is not None
            else float("-inf"),
            float(row["normalized_rms_error_ratio_to_control_median"] or float("-inf")),
        ),
        reverse=True,
    )
    layers = sorted({int(row["layer"]) for row in rows})
    first_joint_outlier_by_layer: dict[str, str | None] = {}
    for layer in layers:
        first_joint_outlier_by_layer[str(layer)] = next(
            (
                str(row["boundary"])
                for row in rows
                if row["layer"] == layer and row["joint_above_control_maximum"]
            ),
            None,
        )
    return {
        "schema": "litenn.sub_layer_checkpoint_comparison.v1",
        "target_generated_index": target_generated_index,
        "control_generated_indices": control_indices or [],
        "boundaries": [boundary for boundary in BOUNDARY_ORDER if boundary in reports],
        "layers": layers,
        "first_joint_outlier_boundary_by_layer": first_joint_outlier_by_layer,
        "ranked_coordinates_by_joint_modified_z": [
            {"boundary": row["boundary"], "layer": row["layer"]} for row in ranked
        ],
        "rows": rows,
    }


def markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Sub-Layer Checkpoint Comparison",
        "",
        f"- Target generated index: `{report['target_generated_index']}`",
        f"- Controls: `{report['control_generated_indices']}`",
        f"- Layers: `{report['layers']}`",
        f"- First joint outlier boundaries: `{report['first_joint_outlier_boundary_by_layer']}`",
        "",
        "| Boundary | Layer | Target NRMSE | Control median | Ratio | NRMSE z | Above max | Cosine z | Above max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in report["rows"]:
        normalized_z = row["normalized_rms_error_modified_z_score"]
        cosine_z = row["cosine_distance_modified_z_score"]
        lines.append(
            f"| {row['boundary']} | {row['layer']} | {row['target_normalized_rms_error']:.6g} | "
            f"{row['control_normalized_rms_error_median']:.6g} | "
            f"{row['normalized_rms_error_ratio_to_control_median']:.6g} | "
            f"{'n/a' if normalized_z is None else f'{normalized_z:.6g}'} | "
            f"{'yes' if row['normalized_rms_error_above_control_maximum'] else 'no'} | "
            f"{'n/a' if cosine_z is None else f'{cosine_z:.6g}'} | "
            f"{'yes' if row['cosine_distance_above_control_maximum'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--generated-indices", required=True, type=comma_indices)
    parser.add_argument("--target-generated-index", required=True, type=int)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.target_generated_index < 0:
        parser.error("target generated index must be non-negative")
    if args.absolute_tolerance < 0 or args.relative_tolerance < 0:
        parser.error("tolerances must be non-negative")

    try:
        boundaries = discover_boundaries(args.reference_root, args.candidate_root)
        reports = {
            boundary: compare_manifests(
                args.reference_root / boundary / "manifest.tsv",
                args.candidate_root / boundary / "manifest.tsv",
                args.absolute_tolerance,
                args.relative_tolerance,
                args.generated_indices,
                args.target_generated_index,
            )
            for boundary in boundaries
        }
        report = aggregate_reports(reports, args.target_generated_index)
    except RuntimeError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sub_layer_checkpoint_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "sub_layer_checkpoint_comparison.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
