#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GGUF_EXAMPLE = ROOT / "example" / "gguf"
if str(GGUF_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(GGUF_EXAMPLE))

import qwen_first_divergence_attribution as attribution


class QwenFirstDivergenceAttributionTest(unittest.TestCase):
    def test_prefill_divergence_has_an_empty_generated_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": attribution.GENERATION_SCHEMA,
                        "promptTokenIds": [10, 11],
                        "generatedTokenIds": [20, 21],
                    }
                ),
                encoding="utf-8",
            )
            quality = root / "quality.json"
            context = attribution.case_context(
                {
                    "name": "prefill",
                    "passedIntegrity": True,
                    "firstDivergenceDecisionStep": 0,
                    "referenceManifest": str(manifest),
                },
                quality,
            )
            self.assertEqual(context["generatedPrefix"], [])
            self.assertEqual(context["forcedTrajectory"], [20])

    def test_layer_summary_reports_thresholds_representatives_and_growth(self) -> None:
        rows = []
        for layer, nrmse in enumerate((1.0e-7, 2.0e-6, 3.0e-4, 2.0e-4, 2.0e-2)):
            rows.append(
                {
                    "generated_index": 3,
                    "layer": layer,
                    "normalized_rms_error": nrmse,
                    "cosine_similarity": 1.0 - nrmse * nrmse,
                    "max_absolute_error": nrmse * 2.0,
                    "mean_absolute_error": nrmse / 2.0,
                }
            )
        summary = attribution.summarize_comparison(
            {
                "rows": rows,
                "first_failing_layer_by_generated_index": {"3": 1},
            },
            3,
        )
        self.assertEqual(summary["firstExactFailingLayer"], 1)
        self.assertEqual(summary["firstLayerByNrmseThreshold"]["1e-06"], 1)
        self.assertEqual(summary["firstLayerByNrmseThreshold"]["1e-04"], 2)
        self.assertEqual(summary["firstLayerByNrmseThreshold"]["1e-02"], 4)
        self.assertEqual([row["layer"] for row in summary["representativeLayers"]], [0, 2, 4])
        self.assertEqual(summary["peakNrmse"]["layer"], 4)
        self.assertEqual(summary["largestPositiveNrmseIncrease"]["layer"], 4)

    def test_markdown_contains_cross_case_summary(self) -> None:
        summary = {
            "layerCount": 48,
            "firstExactFailingLayer": 0,
            "firstLayerByNrmseThreshold": {"1e-04": 12},
            "peakNrmse": {"layer": 47, "normalizedRmsError": 0.02},
            "largestPositiveNrmseIncrease": {"layer": 31},
        }
        markdown = attribution.markdown_report(
            {
                "cases": [
                    {
                        "name": "reasoning",
                        "firstDivergenceDecisionStep": 3,
                        "summary": summary,
                    }
                ]
            }
        )
        self.assertIn("| reasoning | 3 | 48 | 12 | 47 | 0.02 | n/a | n/a |", markdown)

    def test_sub_layer_summary_preserves_boundary_order(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference = root / "reference"
            candidate = root / "candidate"
            for boundary_index, boundary in enumerate(attribution.SUB_LAYER_BOUNDARIES):
                nrmse = 1.0e-6 * (boundary_index + 1)
                comparison = {
                    "rows": [
                        {
                            "generated_index": 0,
                            "layer": 0,
                            "normalized_rms_error": nrmse,
                            "cosine_similarity": 1.0 - nrmse,
                            "max_absolute_error": nrmse,
                            "mean_absolute_error": nrmse,
                        }
                    ],
                    "first_failing_layer_by_generated_index": {"0": 0},
                }
                boundary_dir = root / "fixtures" / boundary
                boundary_dir.mkdir(parents=True)
                (boundary_dir / "comparison.json").write_text(json.dumps(comparison), encoding="utf-8")

            original = attribution.layer_compare.compare_manifests
            try:
                attribution.layer_compare.compare_manifests = lambda reference_path, *_args, **_kwargs: json.loads(
                    (root / "fixtures" / reference_path.parent.name / "comparison.json").read_text(encoding="utf-8")
                )
                for base in (reference, candidate):
                    for boundary in attribution.SUB_LAYER_BOUNDARIES:
                        path = base / boundary / "manifest.tsv"
                        path.parent.mkdir(parents=True)
                        path.write_text("fixture", encoding="utf-8")
                summary = attribution.summarize_sub_layers(
                    reference, candidate, 0, [0], 1.0e-5, 1.0e-5, root / "output"
                )
            finally:
                attribution.layer_compare.compare_manifests = original
            block = summary["blocks"][0]
            self.assertEqual(block["boundaries"][0]["boundary"], "attention_norm")
            self.assertEqual(block["boundaries"][-1]["boundary"], "post_ffn")
            self.assertEqual(block["peakNrmse"]["boundary"], "post_ffn")


if __name__ == "__main__":
    unittest.main()
