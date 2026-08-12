import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_layer_checkpoints import compare_manifests, markdown_report  # noqa: E402


def write_fixture(root: Path, values: list[list[float]]) -> Path:
    payload = root / "generated-000023.bin"
    offsets: list[int] = []
    with payload.open("wb") as stream:
        for layer in values:
            offsets.append(stream.tell())
            stream.write(struct.pack(f"<{len(layer)}f", *layer))
    manifest = root / "manifest.tsv"
    lines = [
        "# litenn-layer-checkpoints-v1",
        "generated_index\tabsolute_step\tposition\tinput_token_id\tfile\tlayer\tname\tdtype\tshape\t"
        "byte_offset\tbyte_size\tminimum\tmaximum\tmean\trms\tnon_finite\tchecksum_fnv1a64",
    ]
    for layer, row in enumerate(values):
        lines.append(
            f"23\t32\t31\t7\t{payload.name}\t{layer}\tlayer_hidden_{layer}\tFloat32\t1x{len(row)}\t"
            f"{offsets[layer]}\t{len(row) * 4}\t0\t0\t0\t0\t0\t0"
        )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


class LayerCheckpointComparisonTest(unittest.TestCase):
    def test_reports_first_failing_layer_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            reference_manifest = write_fixture(reference, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            candidate_manifest = write_fixture(candidate, [[1.0, 2.0], [3.0, 4.25], [5.0, 6.5]])

            report = compare_manifests(reference_manifest, candidate_manifest, 1.0e-4, 1.0e-4)
            self.assertFalse(report["passed"])
            self.assertEqual(report["failing_rows"], 2)
            self.assertEqual(report["first_failing_layer_by_generated_index"], {"23": 1})
            self.assertAlmostEqual(report["rows"][1]["max_absolute_error"], 0.25)
            self.assertIn("| 23 | 1 |", markdown_report(report))

    def test_accepts_identical_bundles(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            values = [[-1.0, 0.0, 1.0], [2.0, 4.0, 8.0]]
            report = compare_manifests(
                write_fixture(reference, values), write_fixture(candidate, values), 0.0, 0.0, {23}
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["first_failing_layer_by_generated_index"], {"23": None})
            self.assertTrue(all(row["cosine_similarity"] == 1.0 for row in report["rows"]))

    def test_rejects_truncated_payload(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            reference_manifest = write_fixture(reference, [[1.0, 2.0]])
            candidate_manifest = write_fixture(candidate, [[1.0, 2.0]])
            (candidate / "generated-000023.bin").write_bytes(struct.pack("<f", 1.0))
            with self.assertRaisesRegex(RuntimeError, "truncated"):
                compare_manifests(reference_manifest, candidate_manifest, 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
