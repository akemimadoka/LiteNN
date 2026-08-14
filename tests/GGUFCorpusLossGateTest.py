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

from scripts import gguf_corpus_loss_gate as gate


def write_logits(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{index}: {value}\n" for index, value in enumerate(values)), encoding="utf-8")


def write_manifest(
    root: Path,
    producer: str,
    prompt: list[int],
    targets: list[int],
    logits: list[list[float]],
    *,
    boundary: str = "pre-target",
    fallback: bool = False,
    target_offset: int = 0,
) -> Path:
    artifacts = []
    for step, values in enumerate(logits):
        path = root / "logits" / f"decision-step-{step:06}.txt"
        write_logits(path, values)
        target_index = (step + target_offset) % len(targets)
        artifacts.append(
            {
                "decisionStep": step,
                "position": len(prompt) + step,
                "targetTokenId": targets[target_index],
                "path": str(path),
            }
        )
    manifest = {
        "schema": gate.MANIFEST_SCHEMA,
        "producer": producer,
        "captureBoundary": boundary,
        "promptTokenIds": prompt,
        "targetTokenIds": targets,
        "fallbackUsed": fallback,
        "logitsArtifacts": artifacts,
    }
    path = root / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def write_campaign(root: Path, reference: Path, candidate: Path, **thresholds: float | int) -> Path:
    document = {
        "schema": gate.CAMPAIGN_SCHEMA,
        "thresholds": {
            "minimumCaseCount": 1,
            "minimumTokenCount": 2,
            "maximumCrossEntropyRegressionNats": 0.02,
            "maximumRelativePerplexityRegression": 0.02,
            "maximumWorstSampleCrossEntropyRegressionNats": 0.10,
            **thresholds,
        },
        "cases": [
            {
                "name": "alignment",
                "referenceManifest": str(reference),
                "candidateManifest": str(candidate),
            }
        ],
    }
    path = root / "campaign.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class CorpusLossGateTest(unittest.TestCase):
    def test_scores_each_target_at_its_pre_feed_decision_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            logits = [[0.0, 4.0, 1.0], [0.0, 1.0, 5.0]]
            reference = write_manifest(root / "reference", "reference", [9], [1, 2], logits)
            candidate = write_manifest(root / "candidate", "candidate", [9], [1, 2], logits)
            report = gate.evaluate_campaign(write_campaign(root, reference, candidate))

            expected_first = math.log(math.exp(-4.0) + 1.0 + math.exp(-3.0))
            expected_second = math.log(math.exp(-5.0) + math.exp(-4.0) + 1.0)
            self.assertTrue(report["passed"])
            self.assertAlmostEqual(report["cases"][0]["steps"][0]["referenceNegativeLogLikelihoodNats"], expected_first)
            self.assertAlmostEqual(report["cases"][0]["steps"][1]["referenceNegativeLogLikelihoodNats"], expected_second)
            self.assertEqual(report["summary"]["crossEntropyDeltaNats"], 0.0)
            self.assertEqual(report["summary"]["relativePerplexityDelta"], 0.0)

    def test_rejects_post_target_and_shifted_target_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            logits = [[0.0, 4.0, 1.0], [0.0, 1.0, 5.0]]
            reference = write_manifest(root / "reference", "reference", [9], [1, 2], logits)
            post_target = write_manifest(
                root / "post", "candidate", [9], [1, 2], logits, boundary="post-target"
            )
            report = gate.evaluate_campaign(write_campaign(root, reference, post_target))
            self.assertFalse(report["passed"])
            self.assertIn("captureBoundary=pre-target", report["cases"][0]["errors"][0])

            shifted = write_manifest(root / "shifted", "candidate", [9], [1, 2], logits, target_offset=1)
            report = gate.evaluate_campaign(write_campaign(root, reference, shifted))
            self.assertFalse(report["passed"])
            self.assertIn("shifted target token", report["cases"][0]["errors"][0])

    def test_enforces_aggregate_perplexity_and_worst_sample_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference_logits = [[0.0, 4.0, 1.0], [0.0, 1.0, 5.0]]
            candidate_logits = [[0.0, 3.0, 1.0], [0.0, 1.0, 3.0]]
            reference = write_manifest(root / "reference", "reference", [9], [1, 2], reference_logits)
            candidate = write_manifest(root / "candidate", "candidate", [9], [1, 2], candidate_logits)
            report = gate.evaluate_campaign(write_campaign(root, reference, candidate))

            self.assertFalse(report["passed"])
            self.assertGreater(report["summary"]["crossEntropyDeltaNats"], 0.02)
            self.assertGreater(report["summary"]["relativePerplexityDelta"], 0.02)
            self.assertFalse(next(check for check in report["checks"] if check["name"] == "crossEntropyRegression")["passed"])

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            logits = [[0.0, 4.0, 1.0], [0.0, 1.0, 5.0]]
            reference = write_manifest(root / "reference", "reference", [9], [1, 2], logits)
            candidate = write_manifest(root / "candidate", "candidate", [9], [1, 2], logits)
            campaign = write_campaign(root, reference, candidate)
            output = root / "report.json"
            markdown = root / "report.md"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "gguf_corpus_loss_gate.py"),
                    "--campaign",
                    str(campaign),
                    "--output",
                    str(output),
                    "--markdown",
                    str(markdown),
                ],
                text=True,
                capture_output=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(json.loads(output.read_text(encoding="utf-8"))["passed"])
            self.assertIn("Result: **PASS**", markdown.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
