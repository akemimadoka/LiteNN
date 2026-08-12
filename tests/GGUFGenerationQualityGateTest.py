#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gguf_generation_quality_gate as gate


def write_logits(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{index}: {value}\n" for index, value in enumerate(values)), encoding="utf-8")


def write_manifest(
    root: Path,
    producer: str,
    prompt: list[int],
    generated: list[int],
    logits: list[list[float]],
    fallback: bool = False,
) -> Path:
    artifacts = []
    for step, values in enumerate(logits):
        path = root / "logits" / f"decision-step-{step:06}.txt"
        write_logits(path, values)
        artifacts.append({"decisionStep": step, "position": len(prompt) + step, "path": str(path)})
    manifest = {
        "schema": gate.MANIFEST_SCHEMA,
        "producer": producer,
        "sampling": "greedy",
        "promptTokenIds": prompt,
        "generatedTokenIds": generated,
        "requestedTokenCount": len(generated),
        "stoppedOnEos": False,
        "fallbackUsed": fallback,
        "logitsArtifacts": artifacts,
    }
    path = root / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class GenerationQualityGateTest(unittest.TestCase):
    def test_first_divergence_is_same_context_and_later_steps_are_not(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference = write_manifest(
                root / "reference",
                "reference",
                [7, 8],
                [2, 3, 4],
                [[0.0, 1.0, 4.0, 2.0, 3.0], [0.0, 1.0, 2.0, 5.0, 3.0], [0.0, 1.0, 2.0, 3.0, 6.0]],
            )
            candidate = write_manifest(
                root / "candidate",
                "candidate",
                [7, 8],
                [2, 1, 0],
                [[0.0, 1.0, 4.0, 2.0, 3.0], [0.0, 5.0, 2.0, 4.0, 3.0], [6.0, 1.0, 2.0, 3.0, 4.0]],
            )
            report = gate.evaluate_case("divergence", reference, candidate, 2)

            self.assertTrue(report["passedIntegrity"])
            self.assertEqual(report["commonPrefixTokenCount"], 1)
            self.assertEqual(report["firstDivergenceDecisionStep"], 1)
            self.assertEqual([step["sameContext"] for step in report["steps"]], [True, True, False])
            self.assertEqual(report["disputedToken"]["referenceDistribution"]["candidateTokenRank"], 4)
            self.assertEqual(report["disputedToken"]["candidateDistribution"]["referenceTokenRank"], 2)
            self.assertGreater(report["disputedToken"]["referenceDistribution"]["preferenceMargin"], 0.0)
            self.assertGreater(report["disputedToken"]["candidateDistribution"]["preferenceMargin"], 0.0)

    def test_campaign_enforces_coverage_and_quality_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cases = []
            for index in range(2):
                reference = write_manifest(root / f"r{index}", "reference", [1], [2, 3], [[0, 1, 3, 2], [0, 1, 2, 3]])
                candidate = write_manifest(root / f"c{index}", "candidate", [1], [2, 3], [[0, 1, 3, 2], [0, 1, 2, 3]])
                cases.append({"name": f"case-{index}", "referenceManifest": str(reference), "candidateManifest": str(candidate)})
            campaign = {
                "schema": gate.CAMPAIGN_SCHEMA,
                "thresholds": {
                    "topK": 2,
                    "minimumCaseCount": 2,
                    "minimumTotalReferenceTokens": 4,
                    "minimumPrefixAgreement": 1.0,
                    "minimumSameContextTopKOverlap": 1.0,
                },
                "cases": cases,
            }
            campaign_path = root / "campaign.json"
            campaign_path.write_text(json.dumps(campaign), encoding="utf-8")

            report = gate.evaluate_campaign(campaign_path)
            self.assertTrue(report["passed"])
            self.assertEqual(report["summary"]["totalReferenceGeneratedTokens"], 4)
            self.assertEqual(report["summary"]["weightedPrefixAgreement"], 1.0)

            campaign["thresholds"]["minimumTotalReferenceTokens"] = 5
            campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
            report = gate.evaluate_campaign(campaign_path)
            self.assertFalse(report["passed"])
            self.assertFalse(next(check for check in report["checks"] if check["name"] == "tokenCoverage")["passed"])

    def test_non_finite_logits_and_fallback_fail_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference = write_manifest(root / "reference", "reference", [1], [2], [[0.0, 1.0, 3.0]])
            candidate = write_manifest(root / "candidate", "candidate", [1], [2], [[0.0, 1.0, math.nan]])
            report = gate.evaluate_case("non-finite", reference, candidate, 2)
            self.assertFalse(report["passedIntegrity"])
            self.assertFalse(report["finiteLogits"])

            candidate = write_manifest(root / "fallback", "candidate", [1], [2], [[0.0, 1.0, 3.0]], fallback=True)
            report = gate.evaluate_case("fallback", reference, candidate, 2)
            self.assertFalse(report["passedIntegrity"])
            self.assertIn("fallbackUsed=false", report["errors"][0])

    def test_cli_writes_json_and_markdown_and_returns_gate_status(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference = write_manifest(root / "reference", "reference", [1], [2], [[0.0, 1.0, 3.0]])
            candidate = write_manifest(root / "candidate", "candidate", [1], [2], [[0.0, 1.0, 3.0]])
            campaign_path = root / "campaign.json"
            campaign_path.write_text(
                json.dumps(
                    {
                        "schema": gate.CAMPAIGN_SCHEMA,
                        "thresholds": {
                            "minimumCaseCount": 1,
                            "minimumTotalReferenceTokens": 1,
                            "minimumPrefixAgreement": 1.0,
                            "minimumSameContextTopKOverlap": 1.0,
                        },
                        "cases": [
                            {
                                "name": "cli",
                                "referenceManifest": str(reference),
                                "candidateManifest": str(candidate),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "report.json"
            markdown = root / "report.md"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "gguf_generation_quality_gate.py"),
                "--campaign",
                str(campaign_path),
                "--output",
                str(output),
                "--markdown",
                str(markdown),
            ]
            completed = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(json.loads(output.read_text(encoding="utf-8"))["passed"])
            self.assertIn("Result: **PASS**", markdown.read_text(encoding="utf-8"))

            failed = subprocess.run(command + ["--minimum-total-reference-tokens", "2"], capture_output=True)
            self.assertEqual(failed.returncode, 1)


if __name__ == "__main__":
    unittest.main()
