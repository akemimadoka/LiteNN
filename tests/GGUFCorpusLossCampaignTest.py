#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from example.gguf import qwen_corpus_loss_campaign as campaign
from scripts import gguf_corpus_loss_gate as gate


def write_logits(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("0: 0.0\n1: 1.0\n2: 2.0\n", encoding="utf-8")


class CorpusLossCampaignTest(unittest.TestCase):
    def test_checked_in_corpus_has_valid_hashes_and_budget(self) -> None:
        path = ROOT / "example" / "gguf" / "qwen_corpus_loss_slice.json"
        document, samples, target_count = campaign.load_corpus_slice(path)

        self.assertEqual(document["license"], "MIT")
        self.assertEqual(target_count, 64)
        self.assertEqual(len(samples), 3)
        for sample in samples:
            self.assertEqual(hashlib.sha256(sample["text"].encode("utf-8")).hexdigest(), sample["sha256"])

    def test_normalizes_forced_candidate_without_shifting_targets(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source_dir = root / "source"
            logits = source_dir / "logits"
            first = logits / "position-000002.txt"
            second = logits / "position-000003.txt"
            write_logits(first)
            write_logits(second)
            source = {
                "schema": campaign.GENERATION_SCHEMA,
                "producer": "LiteNN",
                "sampling": "forced-reference-trajectory",
                "promptTokenIds": [7, 8],
                "generatedTokenIds": [1, 2],
                "fallbackUsed": False,
                "logitsArtifacts": [
                    {"decisionStep": 0, "position": 2, "selectedTokenId": 1, "path": str(first)},
                    {"decisionStep": 1, "position": 3, "selectedTokenId": 2, "path": str(second)},
                ],
            }
            source_path = source_dir / "manifest.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            output_path = root / "normalized" / "manifest.json"

            campaign.normalize_candidate_manifest(source_path, output_path, [1, 2])
            normalized = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(normalized["schema"], gate.MANIFEST_SCHEMA)
            self.assertEqual(normalized["captureBoundary"], "pre-target")
            self.assertEqual([item["targetTokenId"] for item in normalized["logitsArtifacts"]], [1, 2])

    def test_rejects_candidate_position_shift(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            artifact = root / "position.txt"
            write_logits(artifact)
            source = {
                "schema": campaign.GENERATION_SCHEMA,
                "sampling": "forced-reference-trajectory",
                "promptTokenIds": [7],
                "generatedTokenIds": [1],
                "fallbackUsed": False,
                "logitsArtifacts": [
                    {"decisionStep": 0, "position": 2, "selectedTokenId": 1, "path": str(artifact)}
                ],
            }
            source_path = root / "manifest.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "shifted logits positions"):
                campaign.normalize_candidate_manifest(source_path, root / "output" / "manifest.json", [1])


if __name__ == "__main__":
    unittest.main()
